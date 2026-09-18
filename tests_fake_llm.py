"""用脚本化的假大模型测试 run_llm_agent + verify：模型按固定顺序发工具调用，最后输出的 JSON 里
故意包含一个不存在的景点和一个编造的地址，校验层应剔除前者、覆盖后者。"""
import asyncio, json, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import main as M

class FakeLLM:
    def __init__(self):
        self.n = 0
        self.ctx = {}
    async def chat(self, messages, tools=None, json_mode=False):
        self.n += 1
        last_tool = [m for m in messages if m.get("role") == "tool"]
        def tc(name, args):
            return {"role": "assistant", "content": None, "tool_calls": [{"id": f"call{self.n}", "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]}
        if self.n == 1:
            return tc("maps_geo", {"address": "杭州东站", "city": "杭州"})
        if self.n == 2:
            self.ctx["origin"] = json.loads(last_tool[-1]["content"])["results"][0]["location"]
            return tc("maps_text_search", {"keywords": "博物馆", "city": "杭州", "citylimit": True})
        if self.n == 3:
            pois = json.loads(last_tool[-1]["content"])["pois"]
            self.ctx["pois"] = pois[:2]
            return tc("maps_search_detail", {"id": pois[0]["id"]})
        if self.n == 4:
            self.ctx["d0"] = json.loads(last_tool[-1]["content"])
            return tc("maps_search_detail", {"id": self.ctx["pois"][1]["id"]})
        if self.n == 5:
            self.ctx["d1"] = json.loads(last_tool[-1]["content"])
            return tc("maps_direction_transit_integrated", {"origin": self.ctx["origin"], "destination": self.ctx["d0"]["location"], "city": "杭州", "cityd": "杭州"})
        if self.n == 6:
            return tc("maps_direction_driving", {"origin": self.ctx["d0"]["location"], "destination": self.ctx["d1"]["location"]})
        if self.n == 7:
            return tc("maps_direction_walking", {"origin": "1,1", "destination": "2,2"})  # 故意的坏调用
        d0, d1 = self.ctx["d0"], self.ctx["d1"]
        ans = {
            "recommendations": [
                {"poi_id": d0["id"], "name": d0["name"], "reason": "理由A", "stay_minutes": 90},
                {"poi_id": d1["id"], "name": "编造的名字", "reason": "理由B", "stay_minutes": 60},
                {"poi_id": "B0FAKE0000", "name": "凭空捏造博物馆", "reason": "不存在", "stay_minutes": 60},
            ],
            "order": [d1["id"], d0["id"]],
            "legs": [{"from": "origin", "to": d0["id"], "mode": "transit"}, {"from": d0["id"], "to": d1["id"], "mode": "driving"}],
            "summary": "假模型总结",
            "notes": [],
        }
        return {"role": "assistant", "content": "```json\n" + json.dumps(ans, ensure_ascii=False) + "\n```"}

async def run():
    cfg = M.Config.from_env(12)
    ev = M.Evidence()
    M.VERBOSE = True
    async with M.AmapMCP(cfg, ev) as mcp:
        req = {"city": "杭州", "origin": "杭州东站", "duration_hours": 4, "interests": ["博物馆"], "constraints": [], "walk_less": False}
        ans = await M.run_llm_agent(req, FakeLLM(), mcp, 12)
        res = M.verify(ans, ev, req)
    print(M.render(req, res, ev, "fake-llm"))
    names = [r["name"] for r in res["recommendations"]]
    assert "凭空捏造博物馆" not in names, "编造景点未被剔除"
    assert "编造的名字" not in names, "编造名称未被覆盖"
    assert len(res["legs"]) == 2 and all(l["verified"] for l in res["legs"]), res["legs"]
    assert res["legs"][0]["mode"] == "transit" and res["legs"][1]["mode"] == "driving"
    assert any("失败" in f for f in ev.failures), "坏调用应记录失败"
    print("\nLLM 路径 + 校验层测试通过")
asyncio.run(run())
