"""Global gradient-boosting demand forecaster (28-day direct horizon, leak-free lag features)."""
import logging, joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from . import config as C, data
log = logging.getLogger("forecast")
FEATS = ["dow", "month", "doy_sin", "doy_cos", "promo", "lag28", "lag35", "roll28", "sku_code"]
_bundle = {}

def make_features(df, codes):
    """All demand-derived features are lagged >=28 days, so they are known for any day in a 28-day horizon."""
    df = df.sort_values(["sku", "date"]).reset_index(drop=True).copy()
    g = df.groupby("sku")["units"]
    df["lag28"], df["lag35"] = g.shift(28), g.shift(35)
    df["roll28"] = g.transform(lambda s: s.shift(28).rolling(28).mean())
    d = pd.to_datetime(df["date"]); a = 2*np.pi*d.dt.dayofyear/365.25
    df["dow"], df["month"], df["doy_sin"], df["doy_cos"] = d.dt.dayofweek, d.dt.month, np.sin(a), np.cos(a)
    df["sku_code"] = df["sku"].map(codes); return df

def _new(): return HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=.05, max_leaf_nodes=15, random_state=C.SEED)
def _m(y, p): return {"MAE": float(np.mean(abs(y-p))), "RMSE": float(np.sqrt(np.mean((y-p)**2))), "WAPE": float(abs(y-p).sum()/max(y.sum(), 1))}

def train():
    s = data.load_sales(); codes = {k: i for i, k in enumerate(sorted(s.sku.unique()))}
    f = make_features(s, codes).dropna(subset=FEATS)
    cut = f.date.max() - pd.Timedelta(days=C.HOLDOUT); tr, te = f[f.date <= cut], f[f.date > cut].copy()
    te["pred"] = _new().fit(tr[FEATS], tr.units).predict(te[FEATS]); y, p = te.units.values, te.pred.values
    te["res"] = y - p
    metrics = {"gradient_boosting": _m(y, p), "naive_lag28": _m(y, te.lag28.values), "moving_avg_28": _m(y, te.roll28.values),
               "holdout_days": C.HOLDOUT, "per_sku_wape": {k: _m(g.units.values, g.pred.values)["WAPE"] for k, g in te.groupby("sku")}}
    te["blend"] = (te.lag28 + te.lag35) / 2; mid = cut + pd.Timedelta(days=C.HOLDOUT // 2)
    metrics["blend_lag28_35"] = _m(y, te.blend.values); metrics["per_sku"] = {}
    for k, g in te.groupby("sku"):   # sigma from 1st half of holdout, coverage tested on the 2nd (out-of-sample)
        sg = g[g.date <= mid].res.std(); cov = float((g[g.date > mid].res.abs() <= 1.28*sg).mean()); mm = _m(g.units.values, g.pred.values)
        metrics["per_sku"][k] = {"MAE": mm["MAE"], "RMSE": mm["RMSE"], "WAPE": mm["WAPE"], "bias": float(g.res.mean()),
            "naive_WAPE": _m(g.units.values, g.lag28.values)["WAPE"], "ma28_WAPE": _m(g.units.values, g.roll28.values)["WAPE"],
            "blend_WAPE": _m(g.units.values, g.blend.values)["WAPE"], "coverage_80": cov}
    sigma = te.groupby("sku").res.std().to_dict()
    joblib.dump({"model": _new().fit(f[FEATS], f.units), "codes": codes, "sigma": sigma, "metrics": metrics,
                 "holdout": te[["date", "sku", "units", "pred"]].reset_index(drop=True)}, C.MODEL_PATH)
    log.info("trained: %s", metrics["gradient_boosting"]); _bundle.clear(); return metrics

def bundle():
    if not _bundle: _bundle.update(joblib.load(C.MODEL_PATH))
    return _bundle

def predict(sku, horizon=C.HORIZON):
    b = bundle(); s = data.load_sales(sku); last = s.date.max()
    fut = pd.DataFrame({"date": pd.date_range(last + pd.Timedelta(days=1), periods=horizon), "sku": sku, "units": np.nan, "promo": 0})
    f = make_features(pd.concat([s.tail(120)[["date", "sku", "units", "promo"]], fut]), b["codes"])
    f = f[f.units.isna()]
    yhat = np.clip(b["model"].predict(f[FEATS]), 0, None)
    return {"dates": [str(d.date()) for d in f.date], "daily": yhat.tolist(), "sigma": float(b["sigma"][sku]),
            "wape": float(b["metrics"]["per_sku_wape"][sku]), "model": "HistGradientBoosting(poisson)"}
