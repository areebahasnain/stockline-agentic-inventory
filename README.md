# AI Inventory Intelligence & Autonomous Replenishment Platform
FAST-NUCES - one project for **AI Engineering** and **Agentic AI**. All data is **SIMULATED** (900 days x 12 SKUs x 3 suppliers). Planning support only: no purchase orders are placed.

## Architecture
```
React + TypeScript SPA (frontend/, 7 views)  --REST-->  FastAPI (inventory_ai/api.py, CORS)
            |
 Platform graph (portfolio.py, LangGraph, shared PState + audit)
   Decision Agent --route--> Product Planning Agent --> Report Agent
                  \--------> Portfolio Agent --------> Report Agent
                   \-------> Scenario Agent ---------> Report Agent
   Product Planning Agent calls the per-product graph (agents.py) as a tool:
   Orchestrator -> Forecasting -> Risk --(CRITICAL/HIGH/WATCH)--> Supplier -> Optimizer -> Explainer
            |
 Deterministic tools: tools.py (risk, supplier ranking, order policy) | portfolio.py (analyze_sku, portfolio_scan, what_if, parse_request)
 ML: forecast.py (HistGradientBoosting, Poisson loss) | Data: data.py (SQLite) | Config: config.py
```
**Deterministic vs LLM.** Every number (forecast, risk, quantity, spend, scenario delta) comes from tools. The LLM is optional (`ANTHROPIC_API_KEY`), is used only in the Report Agent to narrate, and its text is discarded if it contains any number not present in the tool output (`portfolio.grounded`). Routing is rule-based and deterministic.

## Agentic AI requirements -> where
| Requirement | Implementation |
|---|---|
| Multiple specialised agents | Decision, Product Planning (Forecast/Risk/Supplier/Optimizer/Explainer), Portfolio, Scenario, Report |
| Tool use | each agent calls named tools via `_call`/node functions; inputs/outputs logged |
| Shared state | LangGraph `State` and `PState` (typed, audit reduced with `operator.add`) |
| Conditional routing / autonomy | Decision Agent picks the agent set per request; risk gate skips supplier/optimizer for OK/OVERSTOCK |
| Planning & orchestration | Decision Agent emits an explicit `plan`; product graph is invoked as a sub-graph tool |
| Grounding / safety | numeric guard on LLM text, no ordering, invalid input -> error path |
| Observability | per-step audit (agent, tool, input, output, reasoning, ms) returned by API, shown in dashboard, persisted in SQLite |

## AI Engineering requirements -> where
| Requirement | Implementation |
|---|---|
| Data pipeline / reproducibility | seeded generator, fixed as-of date, SQLite, config via env |
| Leak-free ML pipeline | features lagged >=28d, tested (`test_core`) |
| Evaluation | 56-day holdout, 28-day-ahead; baselines: naive lag28, 28-day MA, lag28/35 blend; global + per-SKU MAE/RMSE/WAPE/bias; 80% interval coverage (sigma fitted on 1st half, tested on 2nd half) |
| Modularity | data / forecast / tools / agents / portfolio / api / app separated |
| APIs | `/ask /plan /portfolio /whatif /suppliers/{sku} /forecast/{sku} /metrics /metrics/{sku} /holdout/{sku} /decisions /products /health` (validated inputs, 404/422) |
| Testing | 24 tests: leakage, routing, tools, scenarios, budget, grounding guard, API, edge cases |
| Logging / errors | `logging` throughout; graph errors caught and returned as structured errors |

## Run
```
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m inventory_ai.agents            # generate data, train, print metrics
cd frontend && cp .env.example .env && npm install && npm run dev   # UI at http://localhost:5173 (needs the API running)
uvicorn inventory_ai.api:app --reload    # http://127.0.0.1:8000/docs
python -m pytest -q
```
Example questions: "Which products are at risk and what is the total spend with a $8k budget?", "What if demand +20% and lead time +5 days without Apex Supply?", "Should I reorder Wireless Mouse?"

## Limitations
Synthetic data; Normal-approximation uncertainty; 28-day horizon (lead time + review is clipped to it); suppliers/prices static; budget allocation is greedy by priority (not an optimal knapsack); rule-based request parsing (no LLM routing); one scenario at a time; frontend is a thin client (no auth, no pagination).
