#!/usr/bin/env python3
"""命令行景点推荐 Agent（高德地图 MCP + 大模型 Function Calling）。

流程：
  1. 解析需求：自然语言 -> 结构化 JSON（城市 / 出发地 / 时长 / 偏好 / 约束）
  2. 工具循环：大模型通过 Function Calling 调用高德 MCP 工具（工具定义运行时从 MCP 读取）
  3. 筛选排序：不超过 3 个景点 + 游览顺序 + 交通方式
  4. 程序校验：所有名称 / 地址 / 距离 / 时间都回到工具原始返回里核对，核对不到的一律标"暂无数据"

大模型接口不可用时自动降级到"规则规划器"（同样只使用 MCP 工具返回的数据，同样过校验层）。
所有密钥只从环境变量读取；任何错误信息输出前都会把密钥脱敏。

文件结构（按顺序）：配置 -> 证据库 -> MCP 客户端 -> 大模型客户端 -> 需求解析 -> 大模型工具循环
                  -> 规则规划器 -> 校验层 -> 终端输出 -> 主流程
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import itertools
import json
import logging
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

try:  # 可选：本地 .env 方便开发，不写入代码
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"))
except Exception:  # pragma: no cover
    pass

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client

NA = "暂无数据"
MAX_RECOMMEND = 3
TOOL_TIMEOUT = 30  # 单次工具调用超时（秒）
WALK_LIMIT_WALK_LESS = 1200  # "少走路"时，超过这个距离不安排步行（米）
WALK_LIMIT_NORMAL = 2000
TRANSIT_WALK_TOO_LONG = 1500  # 公交方案里步行段超过这个距离视为"走太多"（米）

_SECRETS: list[str] = []  # 运行时收集的密钥，用于脱敏


def redact(s: Any) -> str:
    s = str(s)
    for k in _SECRETS:
        if k:
            s = s.replace(k, "***")
    return s


class FatalError(Exception):
    """带人话说明的致命错误：主流程捕获后只打印一行，不打 traceback。"""


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    amap_key: str
    amap_mode: str  # sse | stdio
    llm_key: str | None
    llm_base: str
    llm_path: str
    llm_model: str
    llm_timeout: float
    max_steps: int

    @classmethod
    def from_env(cls, max_steps: int) -> "Config":
        amap_key = os.environ.get("AMAP_MAPS_API_KEY", "").strip()
        if not amap_key:
            raise FatalError("缺少环境变量 AMAP_MAPS_API_KEY（高德地图 Key）。请先设置（或写入同目录 .env）后再运行。")
        llm_key = os.environ.get("LLM_API_KEY", "").strip() or None
        _SECRETS.extend([amap_key] + ([llm_key] if llm_key else []))
        mode = os.environ.get("AMAP_MCP_MODE", "sse").strip().lower() or "sse"
        if mode not in ("sse", "stdio"):
            raise FatalError(f"AMAP_MCP_MODE 只能是 sse 或 stdio，当前为 {mode!r}")
        try:
            timeout = float(os.environ.get("LLM_TIMEOUT", "60") or 60)
        except ValueError:
            raise FatalError("环境变量 LLM_TIMEOUT 必须是数字（秒）")
        return cls(
            amap_key=amap_key,
            amap_mode=mode,
            llm_key=llm_key,
            llm_base=os.environ.get("LLM_API_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            llm_path=os.environ.get("LLM_CHAT_PATH", "/chat/completions") or "/chat/completions",
            llm_model=os.environ.get("LLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
            llm_timeout=timeout,
            max_steps=max_steps,
        )


# --------------------------------------------------------------------------- #
# 工具调用记录 + 证据库（校验层的数据来源）
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    step: int
    tool: str
    args: dict
    ok: bool
    result: str
    seconds: float


@dataclass
class Evidence:
    """只存工具真实返回的数据；最终输出的每个字段都必须能在这里找到来源。"""

    pois: dict[str, dict] = field(default_factory=dict)  # poi_id -> 合并后的 POI 信息
    routes: list[dict] = field(default_factory=list)  # {origin, destination, mode, distance_m, duration_s, ...}
    geocodes: list[dict] = field(default_factory=list)  # {address, city, location, result_city}
    calls: list[ToolCall] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    # ---- 录入 ----
    def ingest(self, tool: str, args: dict, data: dict) -> None:
        if tool in ("maps_text_search", "maps_around_search"):
            for p in data.get("pois") or []:
                if isinstance(p, dict):
                    self._merge_poi(p)
        elif tool == "maps_search_detail":
            if data.get("id"):
                self._merge_poi(data)
        elif tool == "maps_geo":
            g = data.get("return") or data.get("geocodes") or data.get("results") or []
            if isinstance(g, list) and g and isinstance(g[0], dict) and g[0].get("location"):
                self.geocodes.append(
                    {
                        "address": str(args.get("address") or ""),
                        "city": str(args.get("city") or ""),
                        "location": g[0]["location"],
                        "result_city": str(g[0].get("city") or ""),
                        "result_province": str(g[0].get("province") or ""),
                    }
                )
        elif tool in ("maps_direction_walking", "maps_direction_driving", "maps_direction_bicycling"):
            route = data.get("route") if isinstance(data.get("route"), dict) else data  # 不同版本 server 包装层级不同
            paths = route.get("paths") or []
            if paths and isinstance(paths[0], dict):
                self.routes.append(
                    {
                        "origin": route.get("origin") or args.get("origin"),
                        "destination": route.get("destination") or args.get("destination"),
                        "mode": tool.rsplit("_", 1)[-1],
                        "distance_m": _to_int(paths[0].get("distance")),
                        "duration_s": _to_int(paths[0].get("duration")),
                        "detail": "",
                    }
                )
        elif tool == "maps_direction_transit_integrated":
            route = data.get("route") if isinstance(data.get("route"), dict) else data
            transits = route.get("transits") or []
            if transits and isinstance(transits[0], dict):
                t = transits[0]
                lines, dist = [], 0
                for seg in t.get("segments") or []:
                    if not isinstance(seg, dict):
                        continue
                    dist += _to_int((seg.get("walking") or {}).get("distance")) or 0
                    for bl in (seg.get("bus") or {}).get("buslines") or []:
                        dist += _to_int(bl.get("distance")) or 0
                        nm = bl.get("name")
                        if nm:
                            dep = (bl.get("departure_stop") or {}).get("name", "")
                            arr = (bl.get("arrival_stop") or {}).get("name", "")
                            lines.append(f"{nm}（{dep}→{arr}）" if dep or arr else nm)
                    rw = seg.get("railway") or {}
                    if rw.get("name"):
                        lines.append(rw["name"])
                self.routes.append(
                    {
                        "origin": route.get("origin") or args.get("origin"),
                        "destination": route.get("destination") or args.get("destination"),
                        "mode": "transit",
                        # 高德 route.distance 的语义是"起终点步行距离"，这里改用各段（步行+公交）距离之和作为全程距离
                        "distance_m": dist or _to_int(t.get("distance")) or _to_int(route.get("distance")),
                        "duration_s": _to_int(t.get("duration")),
                        "walking_m": _to_int(t.get("walking_distance")),
                        "detail": "；".join(lines),
                    }
                )
        elif tool == "maps_distance":
            origins = str(args.get("origins", "")).split("|")
            typ = str(args.get("type", "1"))
            mode = {"0": "straight", "1": "driving", "3": "walking"}.get(typ, "driving")
            for r in data.get("results") or []:
                if not isinstance(r, dict):
                    continue
                try:
                    o = origins[int(r.get("origin_id", "1")) - 1]
                except Exception:
                    o = None
                self.routes.append(
                    {
                        "origin": o,
                        "destination": args.get("destination"),
                        "mode": mode,
                        "distance_m": _to_int(r.get("distance")),
                        "duration_s": None if mode == "straight" else _to_int(r.get("duration")),  # 直线距离没有用时
                        "detail": "",
                    }
                )

    def _merge_poi(self, p: dict) -> None:
        pid = p.get("id")
        if not pid:
            return
        cur = self.pois.setdefault(str(pid), {})
        for k, v in p.items():
            if v not in (None, "", []):
                cur[k] = v

    # ---- 查询 ----
    def find_poi(self, ref: Any) -> dict | None:
        """按 id 或名称精确找 POI；找不到返回 None（=模型编造）。"""
        if not ref or not isinstance(ref, str):
            return None
        if ref in self.pois:
            return self.pois[ref]
        for p in self.pois.values():
            if p.get("name") == ref:
                return p
        return None

    def find_route(self, origin: str | None, dest: str | None, mode: str | None) -> dict | None:
        """找两点之间的路线；mode 为 None 时匹配任意"真正的路径规划"（不含直线距离）。"""
        if not origin or not dest:
            return None
        best = None
        for r in self.routes:
            if mode and r["mode"] != mode:
                continue
            if not mode and r["mode"] == "straight":
                continue
            if _same_point(r.get("origin"), origin) and _same_point(r.get("destination"), dest):
                best = r  # 取最后一条
        return best

    def origin_location(self, origin_name: str) -> str | None:
        """出发地坐标：优先取地址与出发地同名的 maps_geo 结果，其次第一条 geo，最后按名称找 POI。"""
        for g in self.geocodes:
            if g["address"] == origin_name or origin_name in g["address"] or g["address"] in origin_name:
                return g["location"]
        if self.geocodes:
            return self.geocodes[0]["location"]
        p = self.find_poi(origin_name)
        return p.get("location") if p else None


def _to_int(v: Any) -> int | None:
    try:
        return int(float(v))
    except Exception:
        return None


def _parse_loc(s: Any) -> tuple[float, float] | None:
    if not s:
        return None
    try:
        lon, lat = [float(x) for x in str(s).split(",")[:2]]
        return lon, lat
    except Exception:
        return None


def _same_point(a: Any, b: Any, tol_m: float = 150) -> bool:
    pa, pb = _parse_loc(a), _parse_loc(b)
    if not pa or not pb:
        return False
    return haversine_m(pa, pb) <= tol_m


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    d = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371000 * 2 * math.asin(math.sqrt(d))


# --------------------------------------------------------------------------- #
# 高德 MCP 客户端
# --------------------------------------------------------------------------- #
class AmapMCP:
    def __init__(self, cfg: Config, evidence: Evidence):
        self.cfg = cfg
        self.ev = evidence
        self.session: ClientSession | None = None
        self.tools: list[dict] = []  # OpenAI function-calling 格式
        self._stack = contextlib.AsyncExitStack()

    async def __aenter__(self) -> "AmapMCP":
        try:
            if self.cfg.amap_mode == "stdio":
                params = StdioServerParameters(
                    command="npx",
                    args=["-y", "@amap/amap-maps-mcp-server"],
                    env={**os.environ, "AMAP_MAPS_API_KEY": self.cfg.amap_key},
                )
                transport = stdio_client(params)
            else:
                transport = sse_client(f"https://mcp.amap.com/sse?key={self.cfg.amap_key}")
            read, write = await asyncio.wait_for(self._stack.enter_async_context(transport), timeout=30)
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(session.initialize(), timeout=30)
            self.session = session
            listed = await asyncio.wait_for(session.list_tools(), timeout=30)
        except BaseException as e:  # 连接阶段的任何错误（含 ExceptionGroup）都转成一句人话
            with contextlib.suppress(BaseException):
                await self._stack.aclose()
            if isinstance(e, (KeyboardInterrupt, asyncio.CancelledError)):
                raise
            raise FatalError(
                "无法连接高德 MCP。请检查 AMAP_MAPS_API_KEY 是否有效、网络是否可达"
                + ("、Node/npx 是否可用" if self.cfg.amap_mode == "stdio" else "")
                + f"。底层错误：{_brief_exc(e)}"
            ) from None
        for t in listed.tools:
            schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {"type": "object"}
            self.tools.append(
                {
                    "type": "function",
                    "function": {"name": t.name, "description": (t.description or "")[:1000], "parameters": schema},
                }
            )
        if not self.tools:
            raise FatalError("高德 MCP 已连接但没有返回任何工具，无法继续。")
        return self

    async def __aexit__(self, *exc) -> None:
        with contextlib.suppress(BaseException):
            await self._stack.aclose()

    def tool_names(self) -> list[str]:
        return [t["function"]["name"] for t in self.tools]

    async def call(self, name: str, args: dict) -> tuple[bool, str]:
        """调用工具。返回 (ok, 原始文本)。失败不抛异常，只记录。"""
        ok, text, _ = await self._call(name, args)
        return ok, text

    async def call_json(self, name: str, args: dict) -> tuple[bool, dict]:
        """调用工具并解析 JSON。失败或返回不是 JSON 对象时 ok=False，data={}。"""
        ok, _, data = await self._call(name, args)
        return ok, data

    async def _call(self, name: str, args: dict) -> tuple[bool, str, dict]:
        assert self.session
        step = len(self.ev.calls) + 1
        if not isinstance(args, dict):
            args = {}
        t0 = time.time()
        data: dict = {}
        try:
            res = await asyncio.wait_for(self.session.call_tool(name, args), timeout=TOOL_TIMEOUT)
            text = "".join(getattr(c, "text", "") for c in res.content) or ""
            is_err = bool(getattr(res, "is_error", False) or getattr(res, "isError", False))
            parsed = _try_json(text)
            data = parsed if isinstance(parsed, dict) else {}
            ok = (not is_err) and bool(data) and not _looks_like_error(data)
            if not ok and not text:
                text = json.dumps({"error": "工具返回为空"}, ensure_ascii=False)
        except asyncio.TimeoutError:
            ok, text = False, json.dumps({"error": f"工具调用超时（>{TOOL_TIMEOUT}s）"}, ensure_ascii=False)
        except Exception as e:  # 网络 / 参数错误
            ok, text = False, json.dumps({"error": _brief_exc(e)}, ensure_ascii=False)
        dt = time.time() - t0
        text = redact(text)
        self.ev.calls.append(ToolCall(step, name, args, ok, text, dt))
        if ok:
            self.ev.ingest(name, args, data)
            data = data
        else:
            data = {}
            self.ev.failures.append(f"第{step}步 {name}({json.dumps(args, ensure_ascii=False)[:80]}) 失败：{text[:160]}")
        log(f"  [工具 {step}] {name} {json.dumps(args, ensure_ascii=False)[:120]} -> {'OK' if ok else 'FAIL'} ({dt:.1f}s, {len(text)} 字)")
        return ok, text, data


def _try_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _looks_like_error(d: dict) -> bool:
    if "error" in d or "errcode" in d:
        return True
    if str(d.get("status", "")) == "0":
        return True
    info = str(d.get("info", "")).upper()
    return info.startswith("INVALID") or info.startswith("DAILY_QUERY_OVER") or "ERROR" in info


def _brief_exc(e: BaseException) -> str:
    """把异常（含 ExceptionGroup）压成一行、脱敏。"""
    subs = getattr(e, "exceptions", None)
    if subs:
        inner = "; ".join(_brief_exc(x) for x in list(subs)[:3])
        return redact(f"{type(e).__name__}[{inner}]")
    msg = str(e).strip().splitlines()[0] if str(e).strip() else ""
    return redact(f"{type(e).__name__}: {msg}" if msg else type(e).__name__)[:300]


# --------------------------------------------------------------------------- #
# 大模型客户端（OpenAI 兼容接口，60s 超时，失败重试一次）
# --------------------------------------------------------------------------- #
class LLMUnavailable(Exception):
    pass


class LLM:
    def __init__(self, cfg: Config):
        if not cfg.llm_key:
            raise LLMUnavailable("未设置 LLM_API_KEY")
        self.cfg = cfg
        self.url = cfg.llm_base + cfg.llm_path
        self.client = httpx.AsyncClient(timeout=cfg.llm_timeout)

    async def chat(self, messages: list[dict], tools: list[dict] | None = None, json_mode: bool = False) -> dict:
        body: dict[str, Any] = {"model": self.cfg.llm_model, "messages": messages, "temperature": 0.2}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        last_err = ""
        for attempt in (1, 2):
            try:
                r = await self.client.post(self.url, json=body, headers={"Authorization": f"Bearer {self.cfg.llm_key}"})
                data = _try_json(r.text)
                err_msg = None
                if isinstance(data, dict) and data.get("error"):
                    err = data["error"]
                    err_msg = err.get("message") if isinstance(err, dict) else str(err)
                if r.status_code >= 400 or err_msg:
                    err_msg = err_msg or r.text[:200]
                    low = err_msg.lower()
                    if r.status_code in (400, 401, 403) and ("key" in low or "auth" in low or "expired" in low):
                        raise LLMUnavailable(f"大模型接口拒绝访问（HTTP {r.status_code}）：{redact(err_msg)}")
                    raise RuntimeError(f"HTTP {r.status_code}: {redact(err_msg)}")
                if not isinstance(data, dict) or not data.get("choices"):
                    raise RuntimeError(f"响应格式异常：{redact(r.text[:200])}")
                msg = data["choices"][0].get("message")
                if not isinstance(msg, dict):
                    raise RuntimeError("响应缺少 message 字段")
                return msg
            except LLMUnavailable:
                raise
            except Exception as e:
                last_err = _brief_exc(e)
                log(f"  [LLM] 第{attempt}次调用失败：{last_err}")
                if attempt == 1:
                    await asyncio.sleep(1)
        raise LLMUnavailable(f"大模型调用两次均失败（超时 {self.cfg.llm_timeout:.0f}s）：{last_err}")


def extract_json(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    cand = m.group(1) if m else text
    d = _try_json(cand)
    if isinstance(d, dict):
        return d
    s, e = text.find("{"), text.rfind("}")
    if s >= 0 and e > s:
        d = _try_json(text[s : e + 1])
        if isinstance(d, dict):
            return d
    return None


# --------------------------------------------------------------------------- #
# 第一步：解析需求
# --------------------------------------------------------------------------- #
PARSE_PROMPT = """你是旅游需求解析器。把用户的一句话解析成 JSON，字段：
city(城市名，如"杭州"；用户没说城市但说了出发地，就根据出发地推断城市；推断不出则 null)、
origin(出发地名称，如"杭州东站"，没有则 null)、
duration_hours(游玩小时数，数字："半天"=4，"一天"=8，"两天"=16；没说则 null)、
interests(兴趣偏好数组，如["历史文化"]；用户提到的具体事物也算，如["熊猫"])、
constraints(其他约束数组，如["少走路","带老人","带小孩"])、date_hint(日期提示字符串，如"周六"，没有则 null)。
只输出 JSON，不要解释。"""

INTEREST_WORDS = [
    "历史文化", "历史", "文化", "博物馆", "寺庙", "古镇", "古街", "自然风光", "公园", "湖", "山", "美食", "购物",
    "亲子", "网红", "夜景", "艺术", "园林", "动物园", "熊猫", "海洋馆", "科技馆", "红色", "遗址", "古建筑",
]
CONSTRAINT_WORDS = ["少走路", "不想走太多", "不想走太远", "带老人", "带小孩", "带孩子", "轻松", "省钱", "免费", "不爬山"]


def parse_request_rules(text: str) -> dict:
    """无大模型时的兜底解析（只做关键词匹配，识别不到的交给默认值并提示）。"""
    req: dict[str, Any] = {"city": None, "origin": None, "duration_hours": None, "interests": [], "constraints": [], "date_hint": None}
    m = re.search(r"从(.{2,20}?)(出发|开始|走)", text)
    if m:
        req["origin"] = m.group(1).strip().strip("，,。 ")
    m = re.search(r"在([一-龥]{2,6}?)(市)?(游玩|玩|旅游|逛|游|待|呆)", text) or re.search(r"([一-龥]{2,6}?)(一日游|半日游|两日游|市区游)", text)
    if m:
        req["city"] = m.group(1)
    if "半天" in text:
        req["duration_hours"] = 4
    elif re.search(r"两天|2天|二日", text):
        req["duration_hours"] = 16
    elif re.search(r"一天|全天|一整天|一日", text):
        req["duration_hours"] = 8
    else:
        m = re.search(r"(\d+(?:\.\d+)?|[一二两三四五六七八九十]+)\s*(个)?(小时|钟头)", text)
        if m:
            req["duration_hours"] = _cn_num(m.group(1))
    for kw in INTEREST_WORDS:
        if kw in text and not any(kw in i or i in kw for i in req["interests"]):
            req["interests"].append(kw)
    m = re.search(r"(?:想看|想去|看看|喜欢|想逛)([一-龥]{2,8}?)(?:[，。,、；\s]|$|景点|和|或)", text)
    if m and not req["interests"]:
        req["interests"].append(m.group(1))
    for kw in CONSTRAINT_WORDS:
        if kw in text:
            req["constraints"].append(kw)
    m = re.search(r"(周[一二三四五六日天]|星期[一二三四五六日天]|明天|后天|今天)", text)
    if m:
        req["date_hint"] = m.group(1)
    return req


def _cn_num(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        pass
    table = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    if s == "十":
        return 10
    if len(s) == 1:
        return float(table.get(s, 0)) or None
    if len(s) == 2 and s[0] == "十":
        return 10 + table.get(s[1], 0)
    if len(s) == 2 and s[1] == "十":
        return table.get(s[0], 0) * 10
    return None


async def parse_request(text: str, llm: LLM | None) -> tuple[dict, str]:
    if llm:
        try:
            msg = await llm.chat([{"role": "system", "content": PARSE_PROMPT}, {"role": "user", "content": text}], json_mode=True)
            data = extract_json(msg.get("content") or "")
            if isinstance(data, dict):
                return data, "llm"
            log("  [解析] 大模型没有返回合法 JSON，改用规则解析")
        except LLMUnavailable:
            raise
        except Exception as e:
            log(f"  [解析] 大模型解析失败，改用规则解析：{_brief_exc(e)}")
    return parse_request_rules(text), "rules"


def apply_defaults(req: dict) -> list[str]:
    """类型归一 + 补默认值，返回说明列表。"""
    notes = []
    req["city"] = str(req.get("city") or "").strip().rstrip("市") or None
    req["origin"] = str(req.get("origin") or "").strip() or None
    try:
        req["duration_hours"] = float(req.get("duration_hours") or 0) or None
    except (TypeError, ValueError):
        req["duration_hours"] = None
    req["interests"] = [str(x).strip() for x in (req.get("interests") or []) if x and str(x).strip()] if isinstance(req.get("interests"), list) else []
    req["constraints"] = [str(x).strip() for x in (req.get("constraints") or []) if x and str(x).strip()] if isinstance(req.get("constraints"), list) else []
    if not req["city"] and not req["origin"]:
        notes.append("未识别到城市和出发地，无法规划。请在需求里写明城市，例如“在杭州玩半天”")
    elif not req["city"]:
        notes.append(f"未识别到城市，将按出发地“{req['origin']}”所在城市处理")
    if not req["origin"]:
        req["origin"] = f"{req['city']}市中心" if req["city"] else None
        if req["origin"]:
            notes.append(f"未识别到出发地，默认从“{req['origin']}”出发")
    if not req["duration_hours"]:
        req["duration_hours"] = 4
        notes.append("未识别到游玩时长，默认按半天（约 4 小时）安排")
    if req["duration_hours"] > 48:
        req["duration_hours"] = 48
        notes.append("游玩时长按最多两天（48 小时）处理")
    if not req["interests"]:
        req["interests"] = ["景点"]
        notes.append("未识别到兴趣偏好，默认搜索热门景点")
    req["walk_less"] = any(("少走" in c or "不想走" in c or "轻松" in c or "老人" in c or "不爬" in c) for c in req["constraints"])
    return notes


# --------------------------------------------------------------------------- #
# 第二 / 三步（大模型路径）：Function Calling 工具循环
# --------------------------------------------------------------------------- #
AGENT_SYSTEM = """你是景点推荐 Agent，只能依据高德地图工具的真实返回给建议，绝不编造。
建议步骤（可按情况调整）：
1. 用 maps_geo 把出发地转成坐标（city 参数填城市）。如果用户没给城市，用出发地解析出的城市。
2. 用 maps_text_search（citylimit=true）按用户兴趣搜候选景点，可换 2~3 组关键词；必要时用 maps_around_search 以出发地为中心搜周边。
   如果搜索结果为空，先放宽关键词（例如只用"景点"或城市名+"景区"）再搜，仍然为空则在 notes 里说明。
3. 对 5~8 个最可能的候选调用 maps_search_detail 拿到 location、开放时间、评分等。
4. 选出不超过 3 个景点。策略：时长按 {hours} 小时算，每个景点预留 60~90 分钟游玩；{walk_rule}
   顺序按从出发地由近到远排列，不走回头路。时长 ≥4 小时且候选足够时应推荐 3 个；推荐少于 3 个时必须在 notes 里写明原因。
   如果工具返回了开放时间，优先避开在用户出行日期闭馆的景点。
5. 对"出发地→第1个景点"以及相邻景点之间做路径规划：距离 ≤{walk_limit_km}km 用 maps_direction_walking，
   否则用 maps_direction_transit_integrated（city 和 cityd 都填城市名），公交无结果再用 maps_direction_driving。
   每一段都必须实际调用一次工具。{walk_less_leg_rule}
最后（不再调用工具时）只输出一个 JSON，格式：
{{
  "recommendations": [{{"poi_id": "工具返回的POI id", "name": "名称", "reason": "推荐理由(结合用户偏好和约束)", "stay_minutes": 60}}],
  "order": ["poi_id 按游览顺序"],
  "legs": [{{"from": "origin 或 poi_id", "to": "poi_id", "mode": "walking|transit|driving"}}],
  "summary": "一两句总体说明",
  "notes": ["搜索无结果 / 工具失败 / 字段缺失 / 少于3个的原因等需要告诉用户的情况"]
}}
硬性规则：poi_id 必须来自工具返回；reason、summary、notes 里只能引用工具返回里确实存在的信息（类型、评分、开放时间、地址、距离），
不要加入工具没有返回的评价（例如"AAAA 级""网红""必去"）；距离和时间不要写进 JSON，程序会从工具返回里取。"""


async def run_llm_agent(req: dict, llm: LLM, mcp: AmapMCP, max_steps: int) -> dict:
    walk_less = bool(req.get("walk_less"))
    walk_limit = WALK_LIMIT_WALK_LESS if walk_less else WALK_LIMIT_NORMAL
    walk_rule = (
        "用户希望少走路（或带老人/小孩）：优先选彼此距离近的一簇景点，景点之间只有很近才步行，远的用公交/地铁或打车。"
        if walk_less
        else "景点之间距离适中即可，兼顾知名度与用户偏好。"
    )
    leg_rule = (
        f"用户要少走路：任何一段步行不得超过 {walk_limit/1000:.1f}km；公交方案里如果步行段超过 {TRANSIT_WALK_TOO_LONG/1000:.1f}km，改调 maps_direction_driving 用打车方案。"
        if walk_less
        else ""
    )
    system = AGENT_SYSTEM.format(hours=req["duration_hours"], walk_rule=walk_rule, walk_limit_km=f"{walk_limit/1000:.1f}", walk_less_leg_rule=leg_rule)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "用户需求（已解析）：" + json.dumps(req, ensure_ascii=False)},
    ]
    names = set(mcp.tool_names())
    for _ in range(max_steps):
        msg = await llm.chat(messages, tools=mcp.tools)
        assistant: dict = {"role": "assistant", "content": msg.get("content")}
        calls = msg.get("tool_calls") if isinstance(msg.get("tool_calls"), list) else []
        if calls:
            assistant["tool_calls"] = calls
        messages.append(assistant)
        if not calls:
            data = extract_json(msg.get("content") or "")
            if isinstance(data, dict):
                return data
            messages.append({"role": "user", "content": "请只输出符合要求的 JSON 对象，不要其他文字。"})
            continue
        for tc in calls:
            fn_info = tc.get("function") if isinstance(tc, dict) else None
            fn = (fn_info or {}).get("name") or ""
            args = _try_json((fn_info or {}).get("arguments") or "{}")
            if not isinstance(args, dict):
                args = {}
            if fn not in names:
                ok, text = False, json.dumps({"error": f"未知工具 {fn}，可用工具：{sorted(names)}"}, ensure_ascii=False)
                mcp.ev.failures.append(f"模型调用了不存在的工具 {fn}")
            else:
                ok, text = await mcp.call(fn, args)
            messages.append({"role": "tool", "tool_call_id": str(tc.get("id") or fn), "content": text[:8000]})
    # 达到上限：强制收尾（不再给工具）
    log(f"  [LLM] 工具循环达到上限 {max_steps} 步，要求模型直接收尾")
    messages.append({"role": "user", "content": "工具调用次数已到上限，请基于已有结果直接输出最终 JSON，不要再调用工具。"})
    msg = await llm.chat(messages)
    data = extract_json(msg.get("content") or "")
    if isinstance(data, dict):
        data.setdefault("notes", [])
        if isinstance(data["notes"], list):
            data["notes"].append(f"工具调用达到上限（{max_steps} 步），结果基于已获得的数据")
        return data
    return {"recommendations": [], "order": [], "legs": [], "notes": ["模型未能给出有效 JSON 结果"]}


# --------------------------------------------------------------------------- #
# 第二 / 三步（规则路径）：无大模型时的确定性规划器
# --------------------------------------------------------------------------- #
INTEREST_KEYWORDS = {
    "历史文化": ["历史文化景点", "博物馆", "古迹", "寺庙"],
    "历史": ["历史文化景点", "古迹", "博物馆"],
    "文化": ["文化景点", "博物馆"],
    "博物馆": ["博物馆", "纪念馆"],
    "寺庙": ["寺庙", "寺"],
    "古镇": ["古镇", "历史文化街区"],
    "古街": ["历史文化街区", "古街"],
    "自然风光": ["风景区", "公园", "湖"],
    "公园": ["公园"],
    "亲子": ["动物园", "科技馆", "游乐园"],
    "艺术": ["美术馆", "艺术馆"],
    "园林": ["园林", "公园"],
    "动物园": ["动物园"],
    "熊猫": ["熊猫", "动物园"],
    "海洋馆": ["海洋公园", "海洋馆"],
    "科技馆": ["科技馆", "博物馆"],
    "红色": ["红色景区", "纪念馆"],
    "遗址": ["遗址公园", "遗址博物馆"],
    "古建筑": ["古建筑", "古迹"],
}
SCENIC_TYPECODES = ("11", "1401", "1402", "1403")  # 风景名胜 / 博物馆 / 展览馆 / 会展中心
SUB_POI_MARKERS = ("住室", "旧址", "纪念碑", "纪念亭", "纪念园", "殡葬", "-", "驿站", "停车", "售票", "入口", "出口")


def _wanted(poi: dict) -> bool:
    tc = str(poi.get("typecode") or "")
    if not tc.startswith(SCENIC_TYPECODES):
        return False
    name = str(poi.get("name") or "")
    return not any(m in name for m in SUB_POI_MARKERS)


def _rating(d: dict) -> float:
    try:
        return float(d.get("rating") or 0)
    except (TypeError, ValueError):
        return 0.0


async def run_rules_agent(req: dict, mcp: AmapMCP) -> dict:
    notes: list[str] = []
    city = req.get("city")
    walk_less = bool(req.get("walk_less"))
    # 1) 出发地坐标（顺便确认城市）
    origin_loc = None
    if req.get("origin"):
        geo_args = {"address": req["origin"]}
        if city:
            geo_args["city"] = city
        ok, d = await mcp.call_json("maps_geo", geo_args)
        if ok:
            g = d.get("return") or d.get("geocodes") or d.get("results") or []
            if isinstance(g, list) and g and isinstance(g[0], dict):
                origin_loc = g[0].get("location")
                rc = str(g[0].get("city") or "").rstrip("市")
                if rc and not city:
                    city = req["city"] = rc
                    notes.append(f"根据出发地判断城市为“{rc}”")
                elif rc and city and city not in rc and rc not in city:
                    notes.append(f"出发地“{req['origin']}”定位到“{rc}”，与需求里的城市“{city}”不一致，路线按定位结果计算")
        if not origin_loc:
            ok, d = await mcp.call_json("maps_text_search", {"keywords": req["origin"], "city": city or "", "citylimit": bool(city)})
            pois = d.get("pois") or [] if ok else []
            if pois and isinstance(pois[0], dict) and pois[0].get("id"):
                ok2, d2 = await mcp.call_json("maps_search_detail", {"id": pois[0]["id"]})
                if ok2:
                    origin_loc = d2.get("location")
                    rc = str(d2.get("city") or "").rstrip("市")
                    if rc and not city:
                        city = req["city"] = rc
    if not city:
        return {"recommendations": [], "order": [], "legs": [], "summary": "", "notes": notes + ["无法确定城市，请在需求里写明城市名"]}
    if not origin_loc:
        notes.append(f"无法定位出发地“{req.get('origin')}”，出发地到第一个景点的路线无法计算，只给出景点之间的交通")
    # 2) 关键词搜索：城市级关键词搜（知名度）+ 出发地周边搜（就近），轮询合并
    keywords: list[str] = []
    for it in req["interests"]:
        hit = False
        for k, v in INTEREST_KEYWORDS.items():
            if k in it:
                keywords += v
                hit = True
        if not hit:
            keywords.append(it if ("景" in it or len(it) >= 3) else f"{it}景点")
    keywords = list(dict.fromkeys(keywords))[:4]
    radius_m = 15000 if req["duration_hours"] <= 4 else 30000
    buckets: list[list[dict]] = []
    for kw in keywords:
        ok, d = await mcp.call_json("maps_text_search", {"keywords": kw, "city": city, "citylimit": True})
        if ok:
            buckets.append([p for p in d.get("pois") or [] if isinstance(p, dict) and p.get("id") and _wanted(p)][:8])
        if origin_loc:
            ok, d = await mcp.call_json("maps_around_search", {"keywords": kw, "location": origin_loc, "radius": str(radius_m)})
            if ok:
                buckets.append([p for p in d.get("pois") or [] if isinstance(p, dict) and p.get("id") and _wanted(p)][:4])
    candidates: dict[str, dict] = {}
    for row in itertools.zip_longest(*buckets):
        for p in row:
            if p and p["id"] not in candidates:
                candidates[p["id"]] = p
    if not candidates:
        notes.append(f"按“{'、'.join(keywords)}”在{city}搜索无符合条件的景点，已放宽为“景点”重试")
        ok, d = await mcp.call_json("maps_text_search", {"keywords": "景点", "city": city, "citylimit": True})
        if ok:
            for p in d.get("pois") or []:
                if isinstance(p, dict) and p.get("id") and _wanted(p):
                    candidates.setdefault(p["id"], p)
    if not candidates:
        return {"recommendations": [], "order": [], "legs": [], "summary": "", "notes": notes + [f"在{city}未搜索到任何景点（可能是城市名无法识别或工具调用失败）"]}
    # 3) 详情（拿 location / 评分 / 开放时间）
    detailed: list[dict] = []
    for pid in list(candidates)[:18]:
        ok, d = await mcp.call_json("maps_search_detail", {"id": pid})
        if ok and d.get("location"):
            detailed.append(d)
    if not detailed:
        return {"recommendations": [], "order": [], "legs": [], "summary": "", "notes": notes + ["候选景点详情查询全部失败，无法确定位置"]}
    # 4) 选簇 + 排序
    o = _parse_loc(origin_loc)
    if o:
        for d in detailed:
            d["_d_origin"] = haversine_m(o, _parse_loc(d["location"]))
        pool = [d for d in detailed if d["_d_origin"] <= radius_m] or sorted(detailed, key=lambda d: d["_d_origin"])[:6]
    else:
        pool = detailed
    good = [d for d in pool if _rating(d) >= 4.0]
    if len(good) >= MAX_RECOMMEND:
        pool = good  # 有足够高分候选时不考虑低分景点
    pool = sorted(pool, key=lambda d: (-_rating(d), d.get("_d_origin", 0)))[:8]
    k = min(MAX_RECOMMEND, len(pool))
    near_w = 0.5 if req["duration_hours"] >= 8 else 1.0  # 全天行程对"离出发地远近"不敏感
    best, best_score = None, float("inf")
    for combo in itertools.combinations(pool, k):
        pts = [_parse_loc(d["location"]) for d in combo]
        pair = sum(haversine_m(a, b) for a, b in itertools.combinations(pts, 2))  # 簇内两两距离
        near = min(d.get("_d_origin", 0) for d in combo)  # 离出发地最近的一个
        bonus = sum((_rating(d) - 4.0) * 6000 for d in combo)  # 评分每高 0.1 相当于近 600m
        score = pair * (2.0 if walk_less else 1.0) + near * near_w - bonus
        if score < best_score:
            best, best_score = combo, score
    chosen = list(best or [])
    ordered: list[dict] = []  # 最近邻排序，不走回头路
    cur = o
    rest = chosen[:]
    while rest:
        if cur:
            rest.sort(key=lambda d: haversine_m(cur, _parse_loc(d["location"])))
        nxt = rest.pop(0)
        ordered.append(nxt)
        cur = _parse_loc(nxt["location"])
    # 5) 路线规划：近的步行，远的公交；少走路时公交步行段过长则改打车
    walk_limit = WALK_LIMIT_WALK_LESS if walk_less else WALK_LIMIT_NORMAL
    legs: list[dict] = []  # 与 ordered 一一对应；出发地未定位时第一段为 None
    prev_loc, prev_ref, prev_name = origin_loc, "origin", req.get("origin") or "出发地"
    for d in ordered:
        if not prev_loc:
            legs.append(None)
            prev_loc, prev_ref, prev_name = d["location"], d["id"], d["name"]
            continue
        dist = haversine_m(_parse_loc(prev_loc), _parse_loc(d["location"]))
        mode = "walking" if dist <= walk_limit else "transit"
        if mode == "walking":
            ok, _ = await mcp.call("maps_direction_walking", {"origin": prev_loc, "destination": d["location"]})
            if not ok:
                mode = "driving"
        else:
            ok, _ = await mcp.call("maps_direction_transit_integrated", {"origin": prev_loc, "destination": d["location"], "city": city, "cityd": city})
            r = mcp.ev.find_route(prev_loc, d["location"], "transit") if ok else None
            if not r:
                notes.append(f"{prev_name}→{d['name']} 公交规划无结果，改用驾车/打车方案")
                mode = "driving"
            elif walk_less and (r.get("walking_m") or 0) > TRANSIT_WALK_TOO_LONG:
                notes.append(f"{prev_name}→{d['name']} 公交方案需步行 {fmt_dist(r.get('walking_m'))}，考虑少走路改为打车")
                mode = "driving"
        if mode == "driving":
            await mcp.call("maps_direction_driving", {"origin": prev_loc, "destination": d["location"]})
        legs.append({"from": prev_ref, "to": d["id"], "mode": mode, "_from_loc": prev_loc, "_to_loc": d["location"]})
        prev_loc, prev_ref, prev_name = d["location"], d["id"], d["name"]
    # 6) 时间预算：先按预算压缩停留时长（45~90 分钟），仍装不下才舍弃末尾景点
    budget_min = req["duration_hours"] * 60
    kept = ordered[:]
    travel = 0.0
    while kept:
        travel = 0.0
        for lg in legs[: len(kept)]:
            if not lg:
                continue
            r = mcp.ev.find_route(lg["_from_loc"], lg["_to_loc"], lg["mode"])
            travel += (r["duration_s"] / 60) if r and r.get("duration_s") else 20
        per = (budget_min - travel) / len(kept)
        if per >= 45 or len(kept) == 1:
            break
        notes.append(f"“{kept[-1]['name']}”因超出 {req['duration_hours']:g} 小时预算而未纳入")
        kept.pop()
    per = max(30, min(90, int((budget_min - travel) / max(1, len(kept)))))

    def stay(d: dict) -> int:
        museum = "博物" in str(d.get("type") or "") + str(d.get("name") or "")
        return min(90, per + 15) if museum and (per + 15) * len(kept) + travel <= budget_min + 15 else per

    recs = []
    for d in kept:
        why = []
        if req["interests"] and req["interests"] != ["景点"]:
            why.append(f"类型“{d.get('type', '')}”符合“{'、'.join(req['interests'])}”偏好")
        if _rating(d):
            why.append(f"高德评分 {d['rating']}")
        if "_d_origin" in d:
            why.append(f"距出发地直线约 {d['_d_origin']/1000:.1f} km")
        if walk_less and len(kept) > 1:
            why.append("与其他推荐景点相距较近，减少步行")
        recs.append({"poi_id": d["id"], "name": d["name"], "reason": "；".join(why) or "热门景点", "stay_minutes": stay(d)})
    total_h = (travel + sum(stay(d) for d in kept)) / 60
    summary = f"按 {req['duration_hours']:g} 小时预算，从“{req.get('origin') or '出发地'}”出发，共安排 {len(kept)} 个景点，预计总用时约 {total_h:.1f} 小时（含交通）。"
    legs_out = [{"from": lg["from"], "to": lg["to"], "mode": lg["mode"]} for lg in legs[: len(kept)] if lg]
    return {"recommendations": recs, "order": [d["id"] for d in kept], "legs": legs_out, "summary": summary, "notes": notes}


# --------------------------------------------------------------------------- #
# 第四步：校验（一切以工具返回为准）
# --------------------------------------------------------------------------- #
def verify(answer: Any, ev: Evidence, req: dict) -> dict:
    if not isinstance(answer, dict):
        answer = {}
    origin_name = str(req.get("origin") or "出发地")
    dropped: list[str] = []
    recs_out: list[dict] = []
    id_map: dict[str, str] = {}  # 模型给的引用 -> 真实 id
    raw_recs = [r for r in (answer.get("recommendations") or []) if isinstance(r, dict)] if isinstance(answer.get("recommendations"), list) else []
    for rec in raw_recs[: MAX_RECOMMEND + 3]:
        ref = rec.get("poi_id") or rec.get("name")
        poi = ev.find_poi(ref) or ev.find_poi(rec.get("name"))
        if not poi:
            dropped.append(str(rec.get("name") or ref or "?"))
            continue
        if isinstance(rec.get("poi_id"), str):
            id_map[rec["poi_id"]] = poi["id"]
        id_map[str(poi.get("name", ""))] = poi["id"]
        if any(r["id"] == poi["id"] for r in recs_out):
            continue
        recs_out.append(
            {
                "id": poi["id"],
                "name": poi.get("name") or NA,  # 名称/地址等只从证据库取
                "address": poi.get("address") or NA,
                "type": poi.get("type") or NA,
                "open_time": poi.get("opentime2") or poi.get("open_time") or NA,
                "rating": poi.get("rating") or NA,
                "cost": poi.get("cost") or NA,
                "business_area": poi.get("business_area") or NA,
                "location": poi.get("location") or NA,
                "reason": str(rec.get("reason") or NA),
                "stay_minutes": _to_int(rec.get("stay_minutes")),
            }
        )
        if len(recs_out) == MAX_RECOMMEND:
            break
    valid_ids = {r["id"] for r in recs_out}
    model_legs: dict[tuple[str, str], str | None] = {}
    raw_legs = [lg for lg in (answer.get("legs") or []) if isinstance(lg, dict)] if isinstance(answer.get("legs"), list) else []
    for lg in raw_legs:
        f = id_map.get(str(lg.get("from")), str(lg.get("from")))
        t = id_map.get(str(lg.get("to")), str(lg.get("to")))
        mode = lg.get("mode") if lg.get("mode") in ("walking", "transit", "driving", "bicycling") else None
        model_legs[(f, t)] = mode
    # 顺序优先按 legs 链（每段都对应过工具调用）；legs 不成链时才用模型给的 order
    chain_order: list[str] = []
    cur = "origin"
    nxt = {f: t for (f, t) in model_legs}
    while cur in nxt and nxt[cur] in valid_ids and nxt[cur] not in chain_order:
        cur = nxt[cur]
        chain_order.append(cur)
    raw_order = [str(x) for x in (answer.get("order") or []) if isinstance(x, (str, int))] if isinstance(answer.get("order"), list) else []
    order = chain_order if len(chain_order) == len(valid_ids) else [id_map.get(x, x) for x in raw_order]
    order = list(dict.fromkeys(x for x in order if x in valid_ids))
    for r in recs_out:  # 模型漏排的补到末尾
        if r["id"] not in order:
            order.append(r["id"])
    by_id = {r["id"]: r for r in recs_out}

    origin_loc = ev.origin_location(origin_name)
    legs_out: list[dict] = []
    chain = ["origin"] + order
    for a, b in zip(chain, chain[1:]):
        a_name = origin_name if a == "origin" else by_id[a]["name"]
        a_loc = origin_loc if a == "origin" else by_id[a]["location"]
        b_loc = by_id[b]["location"]
        mode = model_legs.get((a, b))
        route = ev.find_route(a_loc, b_loc, mode) or ev.find_route(a_loc, b_loc, None)
        legs_out.append(
            {
                "from": a_name,
                "to": by_id[b]["name"],
                "mode": (route["mode"] if route else mode) or NA,
                "distance_m": route.get("distance_m") if route else None,
                "duration_s": route.get("duration_s") if route else None,
                "walking_m": route.get("walking_m") if route else None,
                "detail": route.get("detail", "") if route else "",
                "verified": bool(route),
            }
        )
    raw_notes = answer.get("notes")
    notes = [raw_notes] if isinstance(raw_notes, str) else [str(n) for n in (raw_notes or []) if n] if isinstance(raw_notes, list) else []
    if dropped:
        notes.append("以下景点在工具返回中找不到，已剔除：" + "、".join(dropped))
    # 程序级复核：需求里的城市是否被高德识别
    want_city = str(req.get("city") or "").rstrip("市")
    if want_city and ev.geocodes:
        got = ev.geocodes[0]
        got_city = (got.get("result_city") or "") + (got.get("result_province") or "")
        if got_city and want_city not in got_city:
            notes.append(f"需求中的城市“{want_city}”未能在高德识别，出发地实际定位到“{got.get('result_city') or got.get('result_province')}”，请核对城市名")
    # 程序级复核：少走路约束
    if req.get("walk_less"):
        for lg in legs_out:
            walk = lg["distance_m"] if lg["mode"] == "walking" else lg["walking_m"]
            if walk and walk > TRANSIT_WALK_TOO_LONG:
                notes.append(f"{lg['from']}→{lg['to']} 这一段需步行 {fmt_dist(walk)}，与“少走路”不完全相符，可考虑打车")
    # 程序级复核：数量少于 3 个而时长充足时，要有解释
    if 0 < len(recs_out) < MAX_RECOMMEND and (req.get("duration_hours") or 0) >= 4 and not any(("仅" in n or "只" in n or "少于" in n or "个" in n) for n in notes):
        notes.append(f"本次只推荐了 {len(recs_out)} 个景点（模型未说明原因，可能是符合偏好且彼此邻近的候选不足）")
    unverified = [f"{lg['from']}→{lg['to']}" for lg in legs_out if not lg["verified"]]
    if unverified:
        notes.append("以下路段没有获得路径规划工具的返回，距离和时间标为暂无数据：" + "、".join(unverified))
    return {"recommendations": [by_id[i] for i in order], "legs": legs_out, "summary": str(answer.get("summary") or ""), "notes": notes}


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
MODE_CN = {"walking": "步行", "transit": "公交/地铁", "driving": "驾车/打车", "straight": "直线距离", "bicycling": "骑行"}


def fmt_dist(m: int | None) -> str:
    if m is None:
        return NA
    return f"{m/1000:.1f} km" if m >= 1000 else f"{m} m"


def fmt_dur(s: int | None) -> str:
    if s is None:
        return NA
    mins = round(s / 60)
    return f"约 {mins//60} 小时 {mins%60} 分钟" if mins >= 60 else f"约 {mins} 分钟"


def render(req: dict, result: dict, ev: Evidence, planner: str) -> str:
    L: list[str] = []
    L.append("=" * 64)
    L.append("需求解析")
    hours = req.get("duration_hours")
    L.append(f"  城市：{req.get('city') or NA}   出发地：{req.get('origin') or NA}   时长：约 {hours:g} 小时" if hours else f"  城市：{req.get('city') or NA}   出发地：{req.get('origin') or NA}   时长：{NA}")
    L.append(f"  偏好：{'、'.join(req.get('interests') or []) or NA}   约束：{'、'.join(req.get('constraints') or []) or '无'}   规划器：{planner}")
    L.append("-" * 64)
    recs = result["recommendations"]
    if not recs:
        L.append("未能给出推荐（原因见下方提示）。")
    else:
        L.append(f"推荐景点（{len(recs)} 个，按游览顺序）")
        for i, r in enumerate(recs, 1):
            L.append(f"  {i}. {r['name']}")
            L.append(f"     地址：{r['address']}")
            L.append(f"     类型：{r['type']}    评分：{r['rating']}    门票/人均：{r['cost']}")
            L.append(f"     开放时间：{r['open_time']}")
            if r.get("stay_minutes"):
                L.append(f"     建议停留：约 {r['stay_minutes']} 分钟")
            L.append(f"     推荐理由：{r['reason']}")
        L.append("-" * 64)
        L.append("游览顺序与交通（距离/用时来自高德路径规划工具的实际返回）")
        for lg in result["legs"]:
            tag = "" if lg["verified"] else "（该段未获得工具返回）"
            L.append(f"  {lg['from']} → {lg['to']}{tag}")
            L.append(f"     方式：{MODE_CN.get(lg['mode'], lg['mode'])}    距离：{fmt_dist(lg['distance_m'])}    预计用时：{fmt_dur(lg['duration_s'])}")
            if lg["mode"] == "transit":
                L.append(f"     线路：{lg['detail'] or NA}    其中步行：{fmt_dist(lg['walking_m'])}")
        if not result["legs"] and len(recs) > 1:
            L.append(f"  （未获得路径规划结果，景点间距离和时间{NA}）")
    if result.get("summary"):
        L.append("-" * 64)
        L.append(f"总体说明：{result['summary']}")
    notes = list(result.get("notes") or [])
    if ev.failures:
        notes.append(f"共 {len(ev.failures)} 次工具调用失败：" + " | ".join(ev.failures))
    if notes:
        L.append("-" * 64)
        L.append("提示")
        for n in notes:
            L.append(f"  ⚠ {n}")
    L.append("-" * 64)
    L.append(f"工具调用记录（{len(ev.calls)} 次）")
    for c in ev.calls:
        L.append(f"  {c.step:>2}. {c.tool} {json.dumps(c.args, ensure_ascii=False)[:90]} -> {'OK' if c.ok else 'FAIL'}")
    L.append("=" * 64)
    return "\n".join(L)


VERBOSE = True


def log(msg: str) -> None:
    if VERBOSE:
        print(redact(msg), file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
async def main_async(text: str, planner_choice: str, max_steps: int, save_log: bool, as_json: bool) -> int:
    cfg = Config.from_env(max_steps)
    ev = Evidence()
    llm: LLM | None = None
    planner = "rules"
    pre_notes: list[str] = []
    if planner_choice in ("auto", "llm"):
        try:
            llm = LLM(cfg)
            planner = "llm"
        except LLMUnavailable as e:
            if planner_choice == "llm":
                raise FatalError(f"大模型不可用：{e}")
            log(f"  [降级] {e}，改用规则规划器")
            pre_notes.append(f"大模型不可用（{e}），本次使用规则规划器")
    else:
        pre_notes.append("按 --planner rules 指定使用规则规划器")

    log("连接高德 MCP ...")
    async with AmapMCP(cfg, ev) as mcp:
        log(f"  已读取 {len(mcp.tools)} 个工具：{', '.join(mcp.tool_names())}")
        # 1) 解析
        try:
            req, how = await parse_request(text, llm)
        except LLMUnavailable as e:
            if planner_choice == "llm":
                raise FatalError(f"大模型不可用：{e}")
            log(f"  [降级] {e}，改用规则规划器")
            pre_notes.append(f"大模型不可用（{e}），本次使用规则规划器")
            llm, planner = None, "rules(fallback)"
            req, how = parse_request_rules(text), "rules"
        default_notes = pre_notes + apply_defaults(req)
        log(f"  需求解析({how})：{json.dumps(req, ensure_ascii=False)}")
        # 2/3) 规划
        answer: dict
        if not req.get("city") and not req.get("origin"):
            answer = {"recommendations": [], "order": [], "legs": [], "notes": []}
        elif llm:
            try:
                answer = await run_llm_agent(req, llm, mcp, cfg.max_steps)
                planner = f"llm({cfg.llm_model})"
            except LLMUnavailable as e:
                if planner_choice == "llm":
                    raise FatalError(f"大模型不可用：{e}")
                log(f"  [降级] {e}，改用规则规划器（保留已获得的工具结果）")
                default_notes.append(f"大模型中途不可用（{e}），改用规则规划器完成")
                answer = await run_rules_agent(req, mcp)
                planner = "rules(fallback)"
        else:
            answer = await run_rules_agent(req, mcp)
        # 4) 校验
        result = verify(answer, ev, req)
        result["notes"] = default_notes + result["notes"]

    report = render(req, result, ev, planner)
    if as_json:
        print(json.dumps({"request": req, "planner": planner, "result": result, "failures": ev.failures}, ensure_ascii=False, indent=2))
    else:
        print(report)
    if save_log:
        out = Path(__file__).with_name("run_logs")
        out.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        (out / f"{stamp}.txt").write_text(f"输入：{text}\n\n{report}\n", encoding="utf-8")
        (out / f"{stamp}.trace.json").write_text(
            json.dumps(
                {"input": text, "request": req, "planner": planner, "result": result,
                 "calls": [{"step": c.step, "tool": c.tool, "args": c.args, "ok": c.ok, "seconds": round(c.seconds, 2), "result": c.result[:4000]} for c in ev.calls]},
                ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log(f"  运行记录已保存到 {out / stamp}.txt / .trace.json")
    return 0 if result["recommendations"] else 2


def main() -> None:
    global VERBOSE
    ap = argparse.ArgumentParser(description="命令行景点推荐 Agent（高德地图 MCP）")
    ap.add_argument("query", nargs="?", help="自然语言需求，不填则交互输入")
    ap.add_argument("--planner", choices=["auto", "llm", "rules"], default="auto", help="auto: 优先大模型，不可用时降级规则")
    ap.add_argument("--max-steps", type=int, default=12, help="大模型工具循环步数上限")
    ap.add_argument("--save-log", action="store_true", help="把本次运行记录保存到 run_logs/")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    ap.add_argument("--quiet", action="store_true", help="不打印过程日志")
    ap.add_argument("--debug", action="store_true", help="出错时打印完整 traceback（已脱敏）")
    a = ap.parse_args()
    VERBOSE = not a.quiet
    if not a.debug:  # 第三方库（mcp/httpx/anyio）的异常日志只在 --debug 时显示，避免刷屏和泄露 URL
        logging.basicConfig(level=logging.CRITICAL)
        for name in ("mcp", "httpx", "httpcore", "anyio"):
            logging.getLogger(name).setLevel(logging.CRITICAL)
    if a.max_steps < 1:
        sys.exit("--max-steps 必须 ≥ 1")
    try:
        text = a.query or input("请输入需求（城市/出发地、时长、偏好）：").strip()
    except (EOFError, KeyboardInterrupt):
        sys.exit("\n未输入需求，退出。")
    if not text.strip():
        sys.exit("需求不能为空")
    try:
        code = asyncio.run(main_async(text, a.planner, a.max_steps, a.save_log, a.json))
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        code = 130
    except FatalError as e:
        print(f"错误：{redact(e)}", file=sys.stderr)
        code = 1
    except BaseException as e:  # 兜底：绝不把带密钥的 traceback 直接打出来
        if a.debug:
            print(redact(traceback.format_exc()), file=sys.stderr)
        print(f"程序异常：{_brief_exc(e)}（加 --debug 查看详情）", file=sys.stderr)
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
