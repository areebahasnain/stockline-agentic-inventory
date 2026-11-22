"""LangGraph multi-agent workflow. Agents share a typed state, call real tools and append to an audit trail."""
import json, logging, operator, os, time
from typing import Annotated, TypedDict
import pandas as pd
from langgraph.graph import StateGraph, END
from . import config as C, data, forecast as F, tools as T
log = logging.getLogger("agents")

class State(TypedDict, total=False):
    question: str; sku: str; error: str; product: dict; forecast: dict; risk: dict
    suppliers: list; order: dict; report: str; audit: Annotated[list, operator.add]

def _a(agent, tool, inp, out, t0, why):
    return [{"agent": agent, "tool": tool, "input": inp, "output": out, "reasoning": why, "ms": round((time.time()-t0)*1000, 1)}]

def orchestrator(s):
    t0 = time.time(); prods = data.get_products(); q = (s.get("question") or "").lower(); sku = s.get("sku")
    if not sku:
        for _, r in prods.iterrows():
            if r.sku.lower() in q or r["name"].lower() in q: sku = r.sku; break
    if sku not in set(prods.sku):
        return {"error": "Could not identify a known product. Mention a SKU or product name (e.g. 'SKU-003' or 'Wireless Mouse').",
                "audit": _a("Orchestrator", "resolve_product", {"question": q}, None, t0, "No product resolved; aborting.")}
    prod = prods[prods.sku == sku].iloc[0].to_dict()
    return {"sku": sku, "product": prod, "audit": _a("Orchestrator", "resolve_product", {"question": q}, {"sku": sku}, t0,
            "Plan: forecast -> risk -> (supplier -> optimizer if risk requires action) -> explanation.")}

def forecaster(s):
    t0 = time.time(); f = F.predict(s["sku"]); f["mean_daily"] = sum(f["daily"])/len(f["daily"])
    return {"forecast": f, "audit": _a("Demand Forecasting Agent", "forecast.predict", {"sku": s["sku"]},
            {"mean_daily": round(f["mean_daily"], 1), "sigma": round(f["sigma"], 2), "holdout_wape": round(f["wape"], 3)}, t0,
            f"Applied {f['model']} 28-day forecast; uncertainty from holdout residuals.")}

def risk_agent(s):
    t0 = time.time(); sup = data.get_suppliers(s["sku"]); p = s["product"]; f = s["forecast"]
    lead, ls = float(sup.lead_days.median()), float(sup.lead_std.mean())
    r = T.assess_risk(f["daily"], f["sigma"], p["on_hand"], p["on_order"], lead, ls)
    return {"risk": r, "audit": _a("Inventory Risk Agent", "tools.assess_risk", {"on_hand": p["on_hand"], "on_order": p["on_order"], "lead": lead}, r, t0,
            f"Stock-out probability over lead time {r['p_stockout_in_lead_time']:.0%}, cover {r['cover_days']}d -> {r['level']}.")}

def supplier_agent(s):
    t0 = time.time(); f, p = s["forecast"], s["product"]
    need = round(f["mean_daily"]*C.HORIZON - p["on_hand"] - p["on_order"]); ranked = T.rank_suppliers(data.get_suppliers(s["sku"]), max(need, 1))
    top = ranked.iloc[0]
    return {"suppliers": ranked.to_dict("records"), "audit": _a("Supplier Analysis Agent", "tools.rank_suppliers", {"nominal_need": need, "weights": C.WEIGHTS},
            {"best": top.supplier, "score": float(top.score)}, t0, f"{top.supplier} ranks first on weighted price/lead-time/reliability with an MOQ penalty.")}

def optimizer(s):
    t0 = time.time(); top, p, f = s["suppliers"][0], s["product"], s["forecast"]
    o = T.plan_order(f["daily"], f["sigma"], p["on_hand"], p["on_order"], top["lead_days"], top["lead_std"], top["moq"], f["wape"])
    o.update(supplier=top["supplier"], unit_price=top["unit_price"], lead_days=top["lead_days"], moq=top["moq"], est_cost=round(o["order_qty"]*top["unit_price"], 2))
    if o["order_in_days"] is not None: o["order_date"] = str((data.as_of() + pd.Timedelta(days=1+o["order_in_days"])).date())
    return {"order": o, "audit": _a("Replenishment Optimization Agent", "tools.plan_order", {"supplier": top["supplier"], "service_level": C.SERVICE_LEVEL}, o, t0,
            f"Order-up-to policy: demand over lead+review plus safety stock, rounded to MOQ {top['moq']}.")}

def _narrate(facts):
    if not os.getenv("ANTHROPIC_API_KEY"): return None
    try:
        import anthropic
        r = anthropic.Anthropic().messages.create(model=C.LLM_MODEL, max_tokens=400, messages=[{"role": "user",
            "content": "Write a concise inventory planner briefing using ONLY these facts; do not invent numbers:\n" + json.dumps(facts, default=str)}])
        return r.content[0].text
    except Exception as e: log.warning("LLM narration failed, using template: %s", e); return None

def explainer(s):
    t0 = time.time(); p, r, f = s["product"], s["risk"], s["forecast"]
    o = s.get("order") or {"action": "HOLD", "order_qty": 0, "confidence_label": "n/a"}
    txt = (f"**{p['name']} ({p['sku']})** - risk **{r['level']}**. Forecast demand is {f['mean_daily']:.1f} units/day; stock position is "
           f"{p['on_hand']+p['on_order']} units ({r['cover_days']} days of cover) versus a {r['lead_time_used']}-day lead time "
           f"(stock-out probability {r['p_stockout_in_lead_time']:.0%}). ")
    if o["action"] == "HOLD": txt += "Recommendation: **no order now**." + (f" About {r['excess_units']} units are above 60 days of demand - avoid reordering." if r["excess_units"] else "")
    else: txt += (f"Recommendation: **order {o['order_qty']} units from {o['supplier']}** ({'today' if o['order_in_days']==0 else 'in '+str(o['order_in_days'])+' days, on '+o['order_date']}), "
                  f"est. cost ${o['est_cost']:,.2f}, safety stock {o['safety_stock']}. Confidence: {o['confidence_label']} ({o['confidence']}).")
    narrated = _narrate({"product": p["name"], "risk": r, "order": o, "mean_daily_forecast": f["mean_daily"]})
    return {"report": narrated or txt, "audit": _a("Explanation Agent", "llm.narrate" if narrated else "template", {}, {"chars": len(narrated or txt)}, t0,
            "Summarised verified numbers from shared state; no new figures introduced.")}

def build():
    g = StateGraph(State)
    for n, f in [("orchestrator", orchestrator), ("forecast_agent", forecaster), ("risk_agent", risk_agent),
                 ("supplier_agent", supplier_agent), ("optimizer_agent", optimizer), ("explainer", explainer)]: g.add_node(n, f)
    g.set_entry_point("orchestrator")
    g.add_conditional_edges("orchestrator", lambda s: END if s.get("error") else "forecast_agent")
    g.add_edge("forecast_agent", "risk_agent")
    g.add_conditional_edges("risk_agent", lambda s: "supplier_agent" if s["risk"]["level"] in ("CRITICAL", "HIGH", "WATCH") else "explainer")
    g.add_edge("supplier_agent", "optimizer_agent"); g.add_edge("optimizer_agent", "explainer"); g.add_edge("explainer", END)
    return g.compile()

_graph = None
def bootstrap(force=False):
    if force or not C.DB_PATH.exists(): data.init_db()
    if force or not C.MODEL_PATH.exists() or "per_sku" not in F.bundle()["metrics"]: F.train()

def run_plan(question=None, sku=None, save=True):
    global _graph; _graph = _graph or build()
    try: out = _graph.invoke({"question": question or "", "sku": sku, "audit": []})
    except Exception as e:
        log.exception("workflow failed"); return {"error": f"Internal error: {e}", "audit": []}
    if save and not out.get("error"): data.save_decision(out["sku"], {k: out.get(k) for k in ("risk", "order", "report", "audit")})
    log.info("plan sku=%s risk=%s", out.get("sku"), (out.get("risk") or {}).get("level")); return out

if __name__ == "__main__":
    bootstrap(force=True); print(json.dumps(F.bundle()["metrics"], indent=1)); print(run_plan("Should I reorder SKU-001?")["report"])
