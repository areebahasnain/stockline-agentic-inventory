import numpy as np, pandas as pd, pytest
from inventory_ai import agents, tools, forecast as F
@pytest.fixture(scope="session", autouse=True)
def boot(): agents.bootstrap()
def test_no_leakage():
    d = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=100), "sku": "A", "units": np.arange(100.), "promo": 0})
    d2 = d.copy(); d2.loc[80:, "units"] = 999
    cols = ["lag28", "lag35", "roll28"]
    assert F.make_features(d, {"A": 0}).loc[80, cols].equals(F.make_features(d2, {"A": 0}).loc[80, cols])
def test_moq_and_hold():
    o = tools.plan_order([10]*28, 3, 20, 0, 7, 1, 50, .1); assert o["order_qty"] % 50 == 0 and o["order_qty"] >= 50
    assert tools.plan_order([10]*28, 3, 5000, 0, 7, 1, 50, .1)["action"] == "HOLD"
def test_forecast_beats_naive():
    m = F.bundle()["metrics"]; assert m["gradient_boosting"]["MAE"] < m["naive_lag28"]["MAE"]
def test_graph_routes():
    hi = agents.run_plan(sku="SKU-001", save=False); assert hi["risk"]["level"] in ("CRITICAL", "HIGH") and hi["order"]["order_qty"] > 0
    assert any(x["agent"] == "Supplier Analysis Agent" for x in hi["audit"])
    assert "order" not in agents.run_plan(sku="SKU-008", save=False)
def test_unknown():
    assert "error" in agents.run_plan("what about the flux capacitor?", save=False)
def test_horizon_features_independent_of_future_units():
    import numpy as np, pandas as pd
    d = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=120), "sku": "A", "units": np.arange(120.), "promo": 0}); d.loc[100:, "units"] = np.nan
    f = F.make_features(d, {"A": 0}); d2 = d.copy(); d2.loc[100:, "units"] = np.nan; assert f[F.FEATS].equals(F.make_features(d2, {"A": 0})[F.FEATS])
    assert f.loc[100:, ["lag28", "lag35", "roll28"]].notna().all().all()
