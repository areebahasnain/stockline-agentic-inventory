import pytest
from fastapi.testclient import TestClient
from inventory_ai import agents as A, portfolio as P, forecast as F
from inventory_ai.api import app
@pytest.fixture(scope="session", autouse=True)
def boot(): A.bootstrap()
cl = TestClient(app)
used = lambda o: [a["agent"] for a in o["audit"]]

def test_parse_scenario():
    r = P.parse_request("What if demand rises 20% and lead time +5 days, without Apex Supply for SKU-003? budget $5k")
    assert r["modifiers"] == {"demand_mult": 1.2, "lead_add": 5, "exclude": ("Apex Supply",)} and r["sku"] == "SKU-003" and r["budget"] == 5000
    assert P.parse_request("demand drops 10%")["modifiers"]["demand_mult"] == 0.9
def test_routing_portfolio_skips_product_graph():
    o = P.run_request("Which products are at risk and what is the total spend?", save=False)
    assert o["intent"] == "portfolio" and "Portfolio Agent" in used(o) and not any("Forecasting" in a or "Product Planning" in a for a in used(o))
def test_routing_product_uses_subgraph():
    o = P.run_request("Should I reorder SKU-001?", save=False); assert o["intent"] == "product" and "Demand Forecasting Agent" in used(o) and "Supplier Analysis Agent" in used(o)
def test_routing_low_risk_skips_supplier():
    o = P.run_request(sku="SKU-008", save=False); assert "Supplier Analysis Agent" not in used(o)
def test_routing_scenario():
    o = P.run_request("What if demand +20% for SKU-001", save=False); assert o["intent"] == "scenario" and "Scenario Agent" in used(o) and "Portfolio Agent" not in used(o)
@pytest.mark.parametrize("q,kw", [("hello", {}), ("", {}), ("anything", {"sku": "SKU-999"}), ("which products?", {"budget": -5})])
def test_unroutable_or_invalid(q, kw): assert "error" in P.run_request(q, save=False, **kw)
def test_portfolio_priority_and_budget():
    full = P.portfolio_scan(); s = full["summary"]; assert s["required_spend"] > 0 and len(s["at_risk"]) >= 2
    pr = sorted(i["priority"] for i in full["items"] if i["priority"]); assert pr == list(range(1, len(pr) + 1))
    assert P.portfolio_scan(budget=2000)["summary"]["funded_spend"] <= 2000 and P.portfolio_scan(budget=0)["summary"]["funded_spend"] == 0
    first = min((i for i in full["items"] if i["priority"]), key=lambda i: i["priority"]); assert first["risk"]["level"] in ("CRITICAL", "HIGH")
def test_tool_output_shapes():
    a = P.analyze_sku("SKU-001"); assert {"risk", "order", "suppliers"} <= set(a) and a["order"]["order_qty"] % a["order"]["moq"] == 0
    assert a["suppliers"][0]["score"] >= a["suppliers"][-1]["score"]
def test_demand_up_not_safer():
    b, s = P.analyze_sku("SKU-003"), P.analyze_sku("SKU-003", demand_mult=1.5)
    assert s["order"]["order_qty"] >= b["order"]["order_qty"] and s["risk"]["p_stockout_in_lead_time"] >= b["risk"]["p_stockout_in_lead_time"]
def test_lead_time_up_not_safer():
    assert P.analyze_sku("SKU-011", lead_add=5)["risk"]["p_stockout_in_lead_time"] >= P.analyze_sku("SKU-011")["risk"]["p_stockout_in_lead_time"]
def test_supplier_exclusion_and_unavailability():
    top = P.analyze_sku("SKU-001")["order"]["supplier"]; assert P.analyze_sku("SKU-001", exclude=(top,))["order"]["supplier"] != top
    assert P.analyze_sku("SKU-001", exclude=("Apex Supply", "GlobalParts", "QuickShip Co"))["order"]["action"] == "NO_SUPPLIER"
def test_whatif_shapes():
    w = P.what_if(demand_mult=1.2); assert w["delta"]["required_spend"] >= 0 and w["scope"] == "PORTFOLIO"
    assert {"baseline", "projected"} <= set(P.what_if(sku="SKU-001", lead_add=5))
def test_grounding_guard(monkeypatch):
    assert P.grounded("Order 1,200 units, 85% risk", {"a": 1200, "p": 0.85}) and not P.grounded("Order 5000 units", {"a": 1200})
    monkeypatch.setattr(A, "_narrate", lambda f: "Order 99999 units now")
    assert "99999" not in P.run_request("Which products are at risk?", save=False)["report"]
def test_per_sku_metrics_and_baselines():
    m = F.bundle()["metrics"]; assert len(m["per_sku"]) == 12 and {"coverage_80", "bias", "naive_WAPE", "ma28_WAPE", "blend_WAPE"} <= set(m["per_sku"]["SKU-001"])
    assert sum(v["WAPE"] < v["naive_WAPE"] for v in m["per_sku"].values()) >= 8 and "blend_lag28_35" in m
def test_api():
    assert cl.get("/portfolio?budget=3000").json()["summary"]["funded_spend"] <= 3000 and cl.get("/portfolio?budget=-1").status_code == 422
    assert cl.post("/ask", json={"question": "Which products need reordering?"}).status_code == 200 and cl.post("/ask", json={"question": "hello"}).status_code == 422
    assert cl.post("/whatif", json={"sku": "SKU-001", "demand_mult": 1.2}).json()["scope"] == "SKU-001" and cl.post("/whatif", json={"demand_mult": -1}).status_code == 422
    assert cl.get("/suppliers/SKU-001").status_code == 200 and cl.get("/suppliers/NOPE").status_code == 404
    assert cl.get("/metrics/SKU-001").status_code == 200 and cl.get("/holdout/SKU-001").status_code == 200 and cl.get("/decisions").status_code == 200
    assert cl.post("/plan", json={"sku": "SKU-001"}).status_code == 200
def test_budget_phrasings_and_priority_order():
    assert P.parse_request("which products with a $8k budget")["budget"] == 8000 and P.parse_request("budget of $2,500 please")["budget"] == 2500
    o = P.run_request("Which products are at risk with a $3k budget?", save=False); assert o["result"]["summary"]["funded_spend"] <= 3000
    ids = [int(l.split("#")[1].split()[0]) for l in o["report"].splitlines() if l.startswith("- #")]; assert ids == sorted(ids)
def test_cors_preflight():
    r = cl.options("/ask", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
