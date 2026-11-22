"""Pure, unit-testable tools the agents call."""
import numpy as np, pandas as pd
from scipy.stats import norm
from . import config as C

def _L(lead, H): return int(min(max(round(lead), 1), H - 1))

def assess_risk(daily, sigma, on_hand, on_order, lead, lead_std):
    d = np.asarray(daily); L = _L(lead, len(d)); mu = d[:L].sum(); pos = on_hand + on_order
    sd = np.sqrt(sigma**2*L + (d.mean()*lead_std)**2)
    p = float(1 - norm.cdf((pos - mu)/sd)); cover = float(pos/max(d.mean(), 1e-9))
    if p >= .5 or cover < lead: lvl = "CRITICAL"
    elif p >= .2: lvl = "HIGH"
    elif p >= .05 or cover < lead + C.REVIEW_DAYS: lvl = "WATCH"
    elif cover > 90: lvl = "OVERSTOCK"
    else: lvl = "OK"
    return {"level": lvl, "p_stockout_in_lead_time": round(p, 3), "cover_days": round(cover, 1), "lead_time_used": L,
            "lead_time_demand": round(float(mu), 1), "excess_units": int(max(0, pos - d.mean()*60)) if lvl == "OVERSTOCK" else 0}

def rank_suppliers(sup, need, w=C.WEIGHTS):
    d = sup.copy(); need = max(int(need), 1)
    d["order_qty"] = d.moq.apply(lambda m: int(np.ceil(max(need, m)/m)*m)); d["total_cost"] = d.order_qty*d.unit_price
    n = lambda x: (x-x.min())/(x.max()-x.min()) if x.max() > x.min() else x*0
    d["score"] = w["cost"]*(1-n(d.unit_price)) + w["lead"]*(1-n(d.lead_days)) + w["reliability"]*n(d.reliability)
    d["score"] -= .1*np.minimum(1, (d.order_qty-need)/need)   # penalise MOQ-driven over-ordering
    return d.sort_values("score", ascending=False).round(3).reset_index(drop=True)

def plan_order(daily, sigma, on_hand, on_order, lead, lead_std, moq, wape, z=None, review=C.REVIEW_DAYS):
    z = z or float(norm.ppf(C.SERVICE_LEVEL)); d = np.asarray(daily); H = len(d)
    L = _L(lead, H); R = min(review, H - L); pos = on_hand + on_order
    sd = lambda n: np.sqrt(sigma**2*n + (d.mean()*lead_std)**2)
    target = d[:L+R].sum() + z*sd(L+R); rop = d[:L].sum() + z*sd(L)
    raw = max(0.0, target - pos); qty = int(np.ceil(max(raw, moq)/moq)*moq) if raw > 0 else 0
    cum = np.cumsum(d); hit = np.where(pos - cum <= rop)[0]; t_order = int(hit[0]) if len(hit) else None
    t_out = int(np.where(cum >= pos)[0][0]) if (cum >= pos).any() else None
    if t_order is None or qty == 0: qty, t_order, action = 0, None, "HOLD"
    else: action = "ORDER_NOW" if t_order == 0 else "ORDER_SCHEDULED"
    cv = float(sd(L+R)/max(d[:L+R].sum(), 1)); conf = float(np.clip((1-wape)*(1-min(.5, cv)), .05, .99))
    return {"action": action, "order_qty": qty, "order_in_days": t_order, "stockout_in_days_if_no_order": t_out,
            "safety_stock": int(round(z*sd(L+R))), "reorder_point": int(round(rop)), "target_level": int(round(target)),
            "confidence": round(conf, 2), "confidence_label": "High" if conf > .75 else "Medium" if conf > .55 else "Low",
            "service_level": C.SERVICE_LEVEL}
