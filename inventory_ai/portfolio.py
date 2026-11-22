"""Portfolio analytics, what-if scenarios and the Decision/Orchestrator graph.
Deterministic tools produce every number; the optional LLM only narrates, and its text is rejected if it contains ungrounded numbers."""
import json, logging, operator, re, time
from typing import Annotated, TypedDict
import numpy as np
from langgraph.graph import StateGraph, END
from . import agents as A, data, forecast as F, tools as T
log = logging.getLogger("portfolio")
RANK = {"CRITICAL": 0, "HIGH": 1, "WATCH": 2, "OK": 3, "OVERSTOCK": 4}
_fc = {}

def _forecast(sku):
    k = (sku, id(F.bundle()["model"]))
    if k not in _fc: _fc[k] = F.predict(sku)
    return _fc[k]

# ---------------- deterministic tools ----------------
def analyze_sku(sku, demand_mult=1.0, lead_add=0, exclude=()):
    p = data.get_products().set_index("sku").loc[sku].to_dict(); p["sku"] = sku; pos = p["on_hand"] + p["on_order"]
    f = _forecast(sku); d = [x*demand_mult for x in f["daily"]]; sg = f["sigma"]*demand_mult
    allsup = data.get_suppliers(sku); ex = {e.lower() for e in exclude}
    sup = allsup[~allsup.supplier.str.lower().isin(ex)].copy()
    risk = T.assess_risk(d, sg, p["on_hand"], p["on_order"], float(allsup.lead_days.median()) + lead_add, float(allsup.lead_std.mean()))
    hold = {"action": "HOLD", "order_qty": 0, "est_cost": 0.0, "unit_price": 0.0, "moq": 1, "supplier": None}
    ranked, order = [], hold
    if sup.empty: order = {**hold, "action": "NO_SUPPLIER"}
    else:
        sup["lead_days"] = sup.lead_days + lead_add
        ranked = T.rank_suppliers(sup, max(round(sum(d) - pos), 1)); top = ranked.iloc[0]
        if risk["level"] not in ("OK", "OVERSTOCK"):
            order = T.plan_order(d, sg, p["on_hand"], p["on_order"], float(top.lead_days), float(top.lead_std), int(top.moq), f["wape"])
            order.update(supplier=str(top.supplier), unit_price=float(top.unit_price), moq=int(top.moq), lead_days=int(top.lead_days),
                         est_cost=round(order["order_qty"]*float(top.unit_price), 2))
    return {"sku": sku, "name": p["name"], "on_hand": int(p["on_hand"]), "on_order": int(p["on_order"]), "mean_daily": round(float(np.mean(d)), 1),
            "risk": risk, "order": order, "suppliers": ranked.to_dict("records") if len(ranked) else []}

def portfolio_scan(budget=None, **scn):
    rows = [analyze_sku(s, **scn) for s in data.get_products().sku]
    act = sorted([r for r in rows if r["order"]["order_qty"] > 0], key=lambda r: (RANK[r["risk"]["level"]], -r["risk"]["p_stockout_in_lead_time"], r["risk"]["cover_days"]))
    left = budget
    for r in rows: r["priority"] = None; r["order"].update(funded_qty=0, funded_cost=0.0)
    for i, r in enumerate(act, 1):
        o = r["order"]; r["priority"] = i; q = o["order_qty"]
        if budget is not None: q = min(q, int(left // (o["unit_price"]*o["moq"]))*o["moq"]); left -= q*o["unit_price"]
        o["funded_qty"], o["funded_cost"] = q, round(q*o["unit_price"], 2)
    lv = lambda *L: [r["sku"] for r in rows if r["risk"]["level"] in L]
    summary = {"n_products": len(rows), "at_risk": lv("CRITICAL", "HIGH", "WATCH"), "critical": lv("CRITICAL"), "overstock": lv("OVERSTOCK"),
               "required_spend": round(sum(r["order"]["order_qty"]*r["order"]["unit_price"] for r in rows), 2),
               "funded_spend": round(sum(r["order"]["funded_cost"] for r in rows), 2), "budget": budget,
               "unfunded": [r["sku"] for r in act if r["order"]["funded_qty"] < r["order"]["order_qty"]],
               "no_supplier": [r["sku"] for r in rows if r["order"]["action"] == "NO_SUPPLIER"]}
    for r in rows: r.pop("suppliers")
    return {"summary": summary, "items": rows}

def flat(items):
    return [{"priority": r["priority"], "sku": r["sku"], "product": r["name"], "risk": r["risk"]["level"], "cover_days": r["risk"]["cover_days"],
             "p_stockout": r["risk"]["p_stockout_in_lead_time"], "action": r["order"]["action"], "supplier": r["order"].get("supplier"),
             "order_qty": r["order"]["order_qty"], "funded_qty": r["order"]["funded_qty"], "est_cost": r["order"]["est_cost"]} for r in items]

def _num(d): return {k: v for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
def what_if(sku=None, budget=None, demand_mult=1.0, lead_add=0, exclude=()):
    scn = dict(demand_mult=demand_mult, lead_add=lead_add, exclude=tuple(exclude)); desc = {"demand_mult": demand_mult, "lead_add": lead_add, "exclude": list(exclude), "budget": budget}
    if sku:
        sl = lambda a: {"level": a["risk"]["level"], "p_stockout": a["risk"]["p_stockout_in_lead_time"], "cover_days": a["risk"]["cover_days"], "action": a["order"]["action"],
                        "supplier": a["order"].get("supplier"), "order_qty": a["order"]["order_qty"], "est_cost": a["order"]["est_cost"]}
        b, s = sl(analyze_sku(sku)), sl(analyze_sku(sku, **scn))
    else:
        ps = lambda r: {**{k: len(r["summary"][k]) for k in ("at_risk", "critical", "overstock", "unfunded", "no_supplier")},
                        "required_spend": r["summary"]["required_spend"], "funded_spend": r["summary"]["funded_spend"]}
        b, s = ps(portfolio_scan(budget)), ps(portfolio_scan(budget, **scn))
    return {"scope": sku or "PORTFOLIO", "scenario": desc, "baseline": b, "projected": s, "delta": {k: round(s[k] - b[k], 3) for k in _num(b)}}

def parse_request(q, sku=None, budget=None, modifiers=None):
    t = (q or "").lower(); prods = data.get_products(); mod = {}
    m = re.search(r"demand([^%\d]{0,25})(\d+(?:\.\d+)?)\s*%", t)
    if m: mod["demand_mult"] = round(1 + (-1 if re.search(r"drop|fall|decreas|down|reduc|-", m.group(1)) else 1)*float(m.group(2))/100, 4)
    m = re.search(r"lead[\s-]*time([^\d]{0,25})(\d+)\s*day", t)
    if m: mod["lead_add"] = (-1 if re.search(r"shorter|reduc|decreas|down|-", m.group(1)) else 1)*int(m.group(2))
    if re.search(r"unavailable|without|exclude|no longer|out of stock|not available|cancel", t):
        ex = tuple(n for n in data._q("SELECT DISTINCT supplier FROM suppliers").supplier if n.lower() in t)
        if ex: mod["exclude"] = ex
    m = re.search(r"budget[^\d]{0,20}([\d,]+(?:\.\d+)?)\s*(k)?", t) or re.search(r"([\d,]+(?:\.\d+)?)\s*(k)?\s*(?:dollar )?budget", t)
    if budget is None and m: budget = float(m.group(1).replace(",", ""))*(1000 if m.group(2) else 1)
    if not sku: sku = next((r.sku for _, r in prods.iterrows() if r.sku.lower() in t or r["name"].lower() in t), None)
    if modifiers is not None: mod = {"demand_mult": 1.0, "lead_add": 0, "exclude": (), **modifiers}
    if mod: mod = {"demand_mult": 1.0, "lead_add": 0, "exclude": (), **mod}; mod["exclude"] = tuple(mod["exclude"])
    port = re.search(r"\b(all|portfolio|products|skus|which|total|overall|spend|priorit\w*|overstock\w*|budget|critical|at risk|at-risk)\b", t)
    intent = "scenario" if mod else "product" if sku else "portfolio" if port else None
    return {"intent": intent, "sku": sku, "budget": budget, "modifiers": mod}

def grounded(text, facts):
    blob = json.dumps(facts, default=str); pool = [float(x) for x in re.findall(r"\d+\.?\d*", blob)]; pool += [x*100 for x in pool if x <= 1]
    for n in re.findall(r"\d[\d,]*\.?\d*", text):
        n = n.replace(",", "").rstrip(".")
        if n and not any(abs(float(n) - x) <= max(.51, abs(float(n))*.005) for x in pool): return False
    return True

# ---------------- agents (LangGraph nodes) ----------------
class PState(TypedDict, total=False):
    question: str; sku: str; budget: float; modifiers: dict; intent: str; plan: list; result: dict; report: str; error: str
    audit: Annotated[list, operator.add]

def _call(au, agent, name, fn, why, **kw):
    t0 = time.time(); out = fn(**kw)
    brief = {k: v for k, v in out.items() if k not in ("items", "suppliers", "forecast", "audit")} if isinstance(out, dict) else str(out)[:200]
    au.append({"agent": agent, "tool": name, "input": {k: v for k, v in kw.items() if k != "save"}, "output": brief, "reasoning": why, "ms": round((time.time()-t0)*1000, 1)}); return out

def decision_agent(s):
    au = []; r = _call(au, "Decision Agent", "portfolio.parse_request", parse_request, "Classify the request and extract sku/budget/what-if modifiers.",
                       q=s.get("question", ""), sku=s.get("sku"), budget=s.get("budget"), modifiers=s.get("modifiers"))
    if r["budget"] is not None and r["budget"] < 0: r["intent"] = None
    plan = {"product": ["Product Planning sub-graph (forecast->risk->[supplier->optimizer])", "Report Agent"], "portfolio": ["Portfolio Agent", "Report Agent"],
            "scenario": ["Scenario Agent", "Report Agent"], None: []}[r["intent"]]
    if r["sku"] and r["sku"] not in set(data.get_products().sku): r["intent"], plan = None, []
    why = {"product": "Single product named: run the full per-product pipeline; portfolio tools not needed.",
           "portfolio": "No single product and a cross-product question: scan all SKUs; per-product forecasting graph skipped.",
           "scenario": f"What-if modifiers {r['modifiers']} detected: compare baseline vs projected with the Scenario Agent only.",
           None: "Request not understood (or invalid budget/SKU); stopping."}[r["intent"]]
    au.append({"agent": "Decision Agent", "tool": "route", "input": {}, "output": {"intent": r["intent"], "plan": plan}, "reasoning": why, "ms": 0})
    out = {**r, "plan": plan, "audit": au}
    if r["intent"] is None: out["error"] = "Could not route the request. Name a product (SKU or name), ask a portfolio question (e.g. 'which products are at risk?'), or give a what-if (e.g. 'demand +20%'); budgets must be >= 0."
    return out

def product_agent(s):
    au = []; o = _call(au, "Product Planning Agent", "agents.run_plan", A.run_plan, "Run existing LangGraph per-product pipeline as a tool.", question=s.get("question", ""), sku=s["sku"], save=False)
    if o.get("error"): return {"error": o["error"], "audit": au}
    return {"result": {"kind": "product", **{k: o.get(k) for k in ("product", "forecast", "risk", "suppliers", "order", "report")}}, "audit": au + o["audit"]}

def portfolio_agent(s):
    au = []; r = _call(au, "Portfolio Agent", "portfolio.portfolio_scan", portfolio_scan, "Scan all SKUs with risk/supplier/optimizer tools; rank by severity; apply budget.", budget=s.get("budget"))
    return {"result": {"kind": "portfolio", **r}, "audit": au}

def scenario_agent(s):
    au = []; r = _call(au, "Scenario Agent", "portfolio.what_if", what_if, "Re-run tools under modified assumptions and diff against baseline.", sku=s.get("sku"), budget=s.get("budget"), **s["modifiers"])
    return {"result": {"kind": "scenario", **r}, "audit": au}

def _template(q, r):
    if r["kind"] == "product": return r["report"]
    if r["kind"] == "scenario":
        return (f"**What-if ({r['scope']})** under {r['scenario']}: baseline -> projected {json.dumps(r['baseline'])} -> {json.dumps(r['projected'])}. Change: {json.dumps(r['delta'])}.")
    s, t = r["summary"], (q or "").lower(); L = [f"**Portfolio:** {len(s['at_risk'])} of {s['n_products']} products at risk ({len(s['critical'])} critical)."]
    for x in sorted([i for i in flat(r["items"]) if i["priority"]], key=lambda i: i["priority"])[:5]: L.append(f"- #{x['priority']} {x['sku']} {x['product']}: {x['risk']}, order {x['funded_qty']}/{x['order_qty']} from {x['supplier']} (${x['est_cost']:,.0f})")
    L.append(f"Required spend ${s['required_spend']:,.2f}; funded ${s['funded_spend']:,.2f}" + (f" under a ${s['budget']:,.0f} budget." if s["budget"] is not None else "."))
    if s["unfunded"]: L.append(f"Not fully funded: {', '.join(s['unfunded'])}.")
    if s["overstock"]: L.append(f"Overstock (do not reorder): {', '.join(s['overstock'])}.")
    return "\n".join(L)

def report_agent(s):
    au = []; r = s["result"]; txt = _template(s.get("question"), r); src = "template"
    facts = {"kind": r["kind"], **({"summary": r["summary"], "top": flat(r["items"])[:6]} if r["kind"] == "portfolio" else {"delta": r.get("delta"), "baseline": r.get("baseline"), "projected": r.get("projected")} if r["kind"] == "scenario" else {"report": txt})}
    if r["kind"] != "product":
        n = A._narrate(facts)
        if n and grounded(n, facts): txt, src = n, "llm (numbers verified against tool output)"
        elif n: src = "template (LLM text rejected: ungrounded numbers)"
    au.append({"agent": "Report Agent", "tool": "grounded_narration", "input": {"kind": r["kind"]}, "output": {"source": src}, "reasoning": "Numbers come only from tool output; any LLM text is verified against it.", "ms": 0})
    return {"report": txt, "audit": au}

def build_platform():
    g = StateGraph(PState)
    for n, f in [("decision_agent", decision_agent), ("product_agent", product_agent), ("portfolio_agent", portfolio_agent), ("scenario_agent", scenario_agent), ("report_agent", report_agent)]: g.add_node(n, f)
    g.set_entry_point("decision_agent")
    g.add_conditional_edges("decision_agent", lambda s: END if s.get("error") else s["intent"] + "_agent")
    g.add_conditional_edges("product_agent", lambda s: END if s.get("error") else "report_agent")
    g.add_edge("portfolio_agent", "report_agent"); g.add_edge("scenario_agent", "report_agent"); g.add_edge("report_agent", END)
    return g.compile()

_g = None
def run_request(question="", sku=None, budget=None, modifiers=None, save=True):
    global _g; _g = _g or build_platform()
    try: out = _g.invoke({"question": question or "", "sku": sku, "budget": budget, "modifiers": modifiers, "audit": []})
    except Exception as e: log.exception("platform failed"); return {"error": f"Internal error: {e}", "audit": []}
    if save and not out.get("error"): data.save_decision(out.get("sku") or "PORTFOLIO", {"intent": out["intent"], "question": question, "report": out["report"], "audit": out["audit"]})
    log.info("request intent=%s plan=%s", out.get("intent"), out.get("plan")); return out
