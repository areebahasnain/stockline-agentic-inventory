from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from . import agents, data, forecast as F
import os
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="AI Inventory Intelligence API")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","), allow_methods=["*"], allow_headers=["*"])
agents.bootstrap()
class PlanReq(BaseModel): sku: str | None = None; question: str | None = None
@app.get("/health")
def health(): return {"ok": True}
@app.get("/products")
def products(): return data.get_products().to_dict("records")
@app.get("/forecast/{sku}")
def forecast(sku: str):
    try: return F.predict(sku)
    except KeyError: raise HTTPException(404, "unknown sku")
@app.post("/plan")
def plan(r: PlanReq):
    out = agents.run_plan(r.question, r.sku)
    if out.get("error"): raise HTTPException(422, out["error"])
    return out
@app.get("/metrics")
def metrics(): return F.bundle()["metrics"]
@app.get("/decisions")
def decisions(n: int = 20): return data.recent_decisions(n).to_dict("records")

# ---- platform endpoints (portfolio, what-if, suppliers, evaluation) ----
import json
from pydantic import Field
from . import portfolio as P
def _clean(x): return json.loads(json.dumps(x, default=str))
class AskReq(BaseModel): question: str = ""; sku: str | None = None; budget: float | None = Field(None, ge=0)
class WhatIfReq(BaseModel):
    sku: str | None = None; budget: float | None = Field(None, ge=0); demand_mult: float = Field(1.0, gt=0, le=5)
    lead_add: int = Field(0, ge=-10, le=30); exclude: list[str] = []
@app.post("/ask")
def ask(r: AskReq):
    o = P.run_request(r.question, r.sku, r.budget)
    if o.get("error"): raise HTTPException(422, o["error"])
    return _clean({k: o.get(k) for k in ("intent", "plan", "result", "report", "audit")})
@app.get("/portfolio")
def portfolio(budget: float | None = None):
    if budget is not None and budget < 0: raise HTTPException(422, "budget must be >= 0")
    return _clean(P.portfolio_scan(budget))
@app.post("/whatif")
def whatif(r: WhatIfReq):
    o = P.run_request("", r.sku, r.budget, {"demand_mult": r.demand_mult, "lead_add": r.lead_add, "exclude": r.exclude})
    if o.get("error"): raise HTTPException(422, o["error"])
    return _clean({**o["result"], "report": o["report"], "audit": o["audit"]})
@app.get("/suppliers/{sku}")
def suppliers(sku: str):
    if sku not in set(data.get_products().sku): raise HTTPException(404, "unknown sku")
    return _clean(P.analyze_sku(sku)["suppliers"])
@app.get("/metrics/{sku}")
def sku_metrics(sku: str):
    m = F.bundle()["metrics"]["per_sku"]
    if sku not in m: raise HTTPException(404, "unknown sku")
    return m[sku]
@app.get("/holdout/{sku}")
def holdout(sku: str):
    h = F.bundle()["holdout"]; h = h[h.sku == sku]
    if h.empty: raise HTTPException(404, "unknown sku")
    return _clean(h.assign(date=h.date.astype(str)).to_dict("records"))
