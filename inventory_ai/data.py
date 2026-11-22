"""Synthetic-but-realistic data generator + SQLite data layer (data is simulated and labelled as such)."""
import json, sqlite3
from contextlib import closing
import numpy as np, pandas as pd
from . import config as C

END = pd.Timestamp("2026-11-21")
# sku, name, category, unit_cost, base daily demand, trend/day, seasonal amp, phase, days of cover, days on order
SKUS = [
 ("SKU-001","Wireless Mouse","Electronics",11,42,3e-4,.10,0,4,0),
 ("SKU-002","USB-C Cable","Electronics",3,90,2e-4,.08,1,9,0),
 ("SKU-003","Mechanical Keyboard","Electronics",38,18,5e-4,.15,2,12,5),
 ("SKU-004","Desk Lamp","Home",14,25,1e-4,.25,3,22,0),
 ("SKU-005","Water Bottle","Home",6,60,4e-4,.30,4,35,0),
 ("SKU-006","Notebook Pack","Stationery",4,75,0,.20,5,48,0),
 ("SKU-007","Office Chair","Furniture",85,7,3e-4,.10,0,60,0),
 ("SKU-008","Standing Desk","Furniture",190,4,6e-4,.12,1,150,0),
 ("SKU-009","Gel Pens (12)","Stationery",2.5,110,0,.35,2,18,10),
 ("SKU-010","Laptop Stand","Electronics",16,30,4e-4,.10,3,28,0),
 ("SKU-011","Phone Charger","Electronics",7,65,2e-4,.12,4,7,0),
 ("SKU-012","Backpack","Accessories",21,20,2e-4,.40,5,95,0),
]

def init_db():
    rng = np.random.default_rng(C.SEED)
    days = pd.date_range(END - pd.Timedelta(days=899), END); t = np.arange(len(days))
    dow_f = np.array([.9, .95, 1, 1, 1.1, 1.25, 1.05])[days.dayofweek]
    sales, prods, sups = [], [], []
    for sku, name, cat, cost, base, tr, amp, ph, cover, oo in SKUS:
        lam = base*(1+tr*t)*(1+amp*np.sin(2*np.pi*days.dayofyear/365.25+ph))*dow_f
        promo = (rng.random(len(t)) < .05).astype(int)
        units = rng.poisson(lam*(1+.4*promo))
        sales.append(pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "sku": sku, "units": units, "promo": promo}))
        recent = units[-28:].mean()
        prods.append(dict(sku=sku, name=name, category=cat, unit_cost=cost, on_hand=int(recent*cover), on_order=int(recent*oo)))
        for sn in ["Apex Supply", "GlobalParts", "QuickShip Co"]:
            sups.append(dict(sku=sku, supplier=sn, unit_price=round(cost*rng.uniform(.9, 1.12), 2),
                             lead_days=int(rng.integers(4, 19)), lead_std=round(float(rng.uniform(.5, 3)), 1),
                             moq=int(rng.choice([24, 50, 100, 200])), reliability=round(float(rng.uniform(.82, .99)), 2)))
    with closing(sqlite3.connect(C.DB_PATH)) as c:
        pd.concat(sales).to_sql("sales", c, if_exists="replace", index=False)
        pd.DataFrame(prods).to_sql("products", c, if_exists="replace", index=False)
        pd.DataFrame(sups).to_sql("suppliers", c, if_exists="replace", index=False)
        pd.DataFrame([{"as_of": str(END.date())}]).to_sql("meta", c, if_exists="replace", index=False)
        c.execute("CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY, ts TEXT DEFAULT CURRENT_TIMESTAMP, sku TEXT, payload TEXT)")
        c.commit()

def _q(sql, params=()):
    with closing(sqlite3.connect(C.DB_PATH)) as c:
        return pd.read_sql(sql, c, params=params)

def load_sales(sku=None):
    df = _q("SELECT * FROM sales" + (" WHERE sku=?" if sku else "") + " ORDER BY date", (sku,) if sku else ())
    df["date"] = pd.to_datetime(df["date"]); return df
def get_products(): return _q("SELECT * FROM products")
def get_suppliers(sku): return _q("SELECT * FROM suppliers WHERE sku=?", (sku,))
def as_of(): return pd.Timestamp(_q("SELECT as_of FROM meta").iloc[0, 0])

def save_decision(sku, payload):
    with closing(sqlite3.connect(C.DB_PATH)) as c:
        c.execute("CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY, ts TEXT DEFAULT CURRENT_TIMESTAMP, sku TEXT, payload TEXT)")
        c.execute("INSERT INTO decisions(sku,payload) VALUES(?,?)", (sku, json.dumps(payload, default=str))); c.commit()
def recent_decisions(n=20):
    return _q("SELECT id, ts, sku, payload FROM decisions ORDER BY id DESC LIMIT ?", (n,))
