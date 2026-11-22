import { useState } from "react";
import { api, Ask, Hold, Item, Order, Portfolio, Product, ProductResult, Step, WhatIf } from "./api";
import { Card, Cover, Err, Kpi, LineChart, Loading, Pill, Rich, money, pct, useAsync } from "./ui";
type Log = (label: string, audit: Step[]) => void;
const Report = ({ text }: { text: string }) => <div className="report">{text.split("\n").map((l, i) => <p key={i}><Rich text={l} /></p>)}</div>;
const Trace = ({ audit }: { audit: Step[] }) => <table className="tbl"><thead><tr><th>#</th><th>Agent</th><th>Tool</th><th>Reasoning</th><th className="num">ms</th></tr></thead>
  <tbody>{audit.map((s, i) => <tr key={i}><td>{i + 1}</td><td>{s.agent}</td><td><code>{s.tool}</code></td><td style={{ whiteSpace: "normal" }}>{s.reasoning}</td><td className="num">{s.ms}</td></tr>)}</tbody></table>;
function ActionTable({ items, budgeted }: { items: Item[]; budgeted?: boolean }) {
  return <table className="tbl"><thead><tr><th>#</th><th>Product</th><th>Risk</th><th>Cover</th><th className="num">P(stock-out)</th><th>Action</th><th>Supplier</th><th className="num">Order qty</th>{budgeted && <th className="num">Funded</th>}<th className="num">Est. cost</th></tr></thead>
    <tbody>{items.map(i => <tr key={i.sku}><td>{i.priority ?? "–"}</td><td><strong>{i.name}</strong> <span className="muted small">{i.sku}</span></td><td><Pill level={i.risk.level} /></td>
      <td><Cover days={i.risk.cover_days} level={i.risk.level} /></td><td className="num">{pct(i.risk.p_stockout_in_lead_time)}</td><td><Pill level={i.order.action} /></td><td>{i.order.supplier ?? "–"}</td>
      <td className="num">{i.order.order_qty.toLocaleString()}</td>{budgeted && <td className="num">{(i.order.funded_qty ?? 0).toLocaleString()}</td>}<td className="num">{money(i.order.est_cost)}</td></tr>)}</tbody></table>;
}
const sorted = (p: Portfolio) => [...p.items].sort((a, b) => (a.priority ?? 99) - (b.priority ?? 99));

export function Overview({ log }: { log: Log }) {
  const { data, error, loading } = useAsync(() => api.portfolio(), []);
  const [q, setQ] = useState("Which products are at risk and how much should I spend with an $8k budget?");
  const [ans, setAns] = useState<Ask | null>(null); const [err, setErr] = useState(""); const [busy, setBusy] = useState(false);
  async function ask() { setBusy(true); setErr(""); try { const a = await api.ask({ question: q }); setAns(a); log(q, a.audit); } catch (e: any) { setErr(e.message); } setBusy(false); }
  if (loading) return <Loading />; if (!data) return <Err msg={error} />; const s = data.summary;
  return <div className="page">
    <div className="kpis"><Kpi label="Products monitored" value={s.n_products} /><Kpi label="At risk" value={s.at_risk.length} sub={`${s.critical.length} critical`} tone={s.critical.length ? "crit" : ""} />
      <Kpi label="Required replenishment" value={money(s.required_spend)} sub="all recommended orders" /><Kpi label="Overstocked" value={s.overstock.length} sub="do not reorder" /></div>
    <div className="grid2">
      <Card title="Replenishment queue" right={<span className="muted small">ranked by urgency</span>}><ActionTable items={sorted(data).filter(i => i.priority)} /></Card>
      <Card title="Ask the planner"><div className="row" style={{ flexDirection: "column", alignItems: "stretch" }}>
        <textarea value={q} onChange={e => setQ(e.target.value)} rows={3} style={{ font: "inherit", padding: 8, border: "1px solid #cfd5dc", borderRadius: 7 }} />
        <button className="btn" onClick={ask} disabled={busy || !q.trim()}>{busy ? "Working…" : "Ask"}</button></div><Err msg={err} />
        {ans && <><div className="chips">{ans.plan.map(p => <span className="chip" key={p}>{p}</span>)}</div><Report text={ans.report} /></>}</Card>
    </div></div>;
}

export function Planner({ products, log }: { products: Product[]; log: Log }) {
  const [sku, setSku] = useState(products[0].sku); const [res, setRes] = useState<{ ask: Ask; hold: Hold[] } | null>(null); const [err, setErr] = useState(""); const [busy, setBusy] = useState(false);
  async function run() { setBusy(true); setErr(""); try { const [ask, hold] = await Promise.all([api.ask({ sku }), api.holdout(sku)]); setRes({ ask, hold }); log(`Plan ${sku}`, ask.audit); } catch (e: any) { setErr(e.message); } setBusy(false); }
  const r: ProductResult | undefined = res?.ask.result; const o: Order | null | undefined = r?.order;
  let chart = null;
  if (r && res) { const h = res.hold.slice(-42), f = r.forecast, nh = h.length, pad = (a: number[]) => [...Array(nh).fill(null), ...a] as (number | null)[];
    chart = <LineChart labels={[...h.map(x => x.date), ...f.dates]} band={{ lo: pad(f.daily.map(v => Math.max(0, v - 1.28 * f.sigma))), hi: pad(f.daily.map(v => v + 1.28 * f.sigma)) }}
      series={[{ name: "Actual sales", color: "#15202b", vals: [...h.map(x => x.units), ...Array(f.dates.length).fill(null)] }, { name: "Model fit (holdout)", color: "#9aa5b1", dash: true, vals: [...h.map(x => x.pred), ...Array(f.dates.length).fill(null)] },
        { name: "28-day forecast", color: "#1f5f8b", vals: pad(f.daily) }]} />; }
  return <div className="page">
    <Card><div className="row"><label className="field">Product<select value={sku} onChange={e => setSku(e.target.value)}>{products.map(p => <option key={p.sku} value={p.sku}>{p.sku} – {p.name}</option>)}</select></label>
      <button className="btn" onClick={run} disabled={busy}>{busy ? "Planning…" : "Generate plan"}</button></div></Card><Err msg={err} />
    {r && res && <>
      <Card><div className="rec-card"><div><div className="muted small">Recommendation · {r.product.name}</div>
        <div className="rec-t">{o && o.order_qty > 0 ? `Order ${o.order_qty.toLocaleString()} units from ${o.supplier}` : "No order required"}</div>
        <div className="muted">{o && o.order_qty > 0 ? `${o.order_in_days === 0 ? "Place today" : `Place on ${o.order_date}`} · est. ${money(o.est_cost)} · confidence ${o.confidence_label} (${o.confidence})` : "Stock position covers forecast demand over the lead time."}</div></div>
        <div style={{ display: "flex", gap: 8 }}><Pill level={r.risk.level} /><Pill level={o?.action ?? "HOLD"} /></div></div></Card>
      <div className="kpis"><Kpi label="Days of cover" value={r.risk.cover_days} sub={`lead time ${r.risk.lead_time_used}d`} /><Kpi label="P(stock-out in lead time)" value={pct(r.risk.p_stockout_in_lead_time)} />
        <Kpi label="Forecast / day" value={r.forecast.mean_daily.toFixed(1)} sub={`holdout WAPE ${pct(r.forecast.wape)}`} /><Kpi label="Safety stock" value={o?.safety_stock ?? "–"} sub={o ? `reorder point ${o.reorder_point}` : ""} /></div>
      <Card title="Demand: history and 28-day forecast">{chart}</Card>
      <div className="grid2"><Card title="Supplier options">{r.suppliers && r.suppliers.length ? <SupplierTable rows={r.suppliers} /> : <span className="muted">Supplier analysis was skipped: risk did not require action.</span>}</Card>
        <Card title="Briefing"><Report text={res.ask.report} /></Card></div>
      <Card title="Decision trace"><Trace audit={res.ask.audit} /></Card></>}
  </div>;
}

function SupplierTable({ rows }: { rows: any[] }) {
  return <table className="tbl"><thead><tr><th>Supplier</th><th className="num">Unit price</th><th className="num">Lead</th><th className="num">MOQ</th><th className="num">Reliability</th><th className="num">Order qty</th><th className="num">Total</th><th>Score</th></tr></thead>
    <tbody>{rows.map((s, i) => <tr key={s.supplier} className={i === 0 ? "rec" : ""}><td><strong>{s.supplier}</strong> {i === 0 && <span className="pill ok">Recommended</span>}</td><td className="num">${s.unit_price.toFixed(2)}</td>
      <td className="num">{s.lead_days}d ±{s.lead_std}</td><td className="num">{s.moq}</td><td className="num">{pct(s.reliability)}</td><td className="num">{s.order_qty}</td><td className="num">{money(s.total_cost)}</td>
      <td><span className="scorebar" style={{ width: Math.max(4, s.score * 70) }} />{s.score.toFixed(2)}</td></tr>)}</tbody></table>;
}

export function PortfolioPage() {
  const [b, setB] = useState(""); const [applied, setApplied] = useState<number | undefined>(); const { data, error, loading } = useAsync(() => api.portfolio(applied), [applied]);
  return <div className="page"><Card><div className="row"><label className="field">Budget ($, blank = unconstrained)<input type="number" min={0} step={500} value={b} onChange={e => setB(e.target.value)} /></label>
    <button className="btn" onClick={() => setApplied(Number(b) || undefined)}>Apply</button></div></Card><Err msg={error} />
    {loading && <Loading />}{data && <><div className="kpis"><Kpi label="Required spend" value={money(data.summary.required_spend)} /><Kpi label="Funded spend" value={money(data.summary.funded_spend)} sub={data.summary.budget != null ? `budget ${money(data.summary.budget)}` : "no budget cap"} />
      <Kpi label="Not fully funded" value={data.summary.unfunded.length} sub={data.summary.unfunded.join(", ")} tone={data.summary.unfunded.length ? "crit" : ""} /><Kpi label="No supplier available" value={data.summary.no_supplier.length} /></div>
      <Card title="All products"><ActionTable items={sorted(data)} budgeted /></Card></>}</div>;
}

export function SuppliersPage({ products }: { products: Product[] }) {
  const [sku, setSku] = useState(products[0].sku); const { data, error, loading } = useAsync(() => api.suppliers(sku), [sku]);
  return <div className="page"><Card><label className="field">Product<select value={sku} onChange={e => setSku(e.target.value)}>{products.map(p => <option key={p.sku} value={p.sku}>{p.sku} – {p.name}</option>)}</select></label></Card>
    <Err msg={error} />{loading ? <Loading /> : data && <Card title="Supplier comparison"><SupplierTable rows={data} /></Card>}
    <p className="muted small">Score = 40% price, 30% lead time, 30% reliability, minus a penalty when the MOQ forces over-ordering. Simulated supplier data.</p></div>;
}

export function WhatIfPage({ products, log }: { products: Product[]; log: Log }) {
  const names = useAsync(() => api.suppliers(products[0].sku), []); const [scope, setScope] = useState("PORTFOLIO"); const [dm, setDm] = useState(20); const [la, setLa] = useState(5);
  const [ex, setEx] = useState<string[]>([]); const [bg, setBg] = useState(""); const [res, setRes] = useState<WhatIf | null>(null); const [err, setErr] = useState(""); const [busy, setBusy] = useState(false);
  async function run() { setBusy(true); setErr(""); try { const w = await api.whatif({ sku: scope === "PORTFOLIO" ? null : scope, budget: Number(bg) || null, demand_mult: 1 + dm / 100, lead_add: la, exclude: ex }); setRes(w); log(`What-if ${scope}`, w.audit); } catch (e: any) { setErr(e.message); } setBusy(false); }
  const fmt = (v: unknown) => typeof v === "number" ? (Math.abs(v) >= 1000 ? money(v) : String(v)) : String(v ?? "–");
  return <div className="page"><Card title="Scenario assumptions"><div className="row">
    <label className="field">Scope<select value={scope} onChange={e => setScope(e.target.value)}><option value="PORTFOLIO">Entire portfolio</option>{products.map(p => <option key={p.sku} value={p.sku}>{p.sku} – {p.name}</option>)}</select></label>
    <label className="field" style={{ width: 200 }}>Demand change: {dm > 0 ? "+" : ""}{dm}%<input type="range" min={-50} max={100} step={5} value={dm} onChange={e => setDm(+e.target.value)} /></label>
    <label className="field" style={{ width: 200 }}>Lead time change: {la > 0 ? "+" : ""}{la} days<input type="range" min={-5} max={15} value={la} onChange={e => setLa(+e.target.value)} /></label>
    <label className="field">Budget ($)<input type="number" min={0} value={bg} onChange={e => setBg(e.target.value)} style={{ width: 110 }} /></label>
    <div className="field">Unavailable suppliers<div className="row">{(names.data ?? []).map(s => <label key={s.supplier} className="small"><input type="checkbox" checked={ex.includes(s.supplier)} onChange={e => setEx(e.target.checked ? [...ex, s.supplier] : ex.filter(x => x !== s.supplier))} /> {s.supplier}</label>)}</div></div>
    <button className="btn" onClick={run} disabled={busy}>{busy ? "Running…" : "Run scenario"}</button></div></Card><Err msg={err} />
    {res && <><Card title={`Impact · ${res.scope}`}><table className="tbl"><thead><tr><th>Metric</th><th className="num">Baseline</th><th className="num">Scenario</th><th className="num">Change</th></tr></thead><tbody>
      {Object.keys(res.baseline).map(k => { const d = res.delta[k]; return <tr key={k}><td>{k.replace(/_/g, " ")}</td><td className="num">{fmt(res.baseline[k])}</td><td className="num">{fmt(res.projected[k])}</td>
        <td className={`num ${d > 0 ? "up" : d < 0 ? "down" : ""}`}>{d === undefined ? "" : (d > 0 ? "+" : "") + fmt(d)}</td></tr>; })}</tbody></table></Card>
      <Card title="Trace"><Trace audit={res.audit} /></Card></>}</div>;
}

export function ModelPage({ products }: { products: Product[] }) {
  const { data: m, error, loading } = useAsync(() => api.metrics(), []); const [sku, setSku] = useState(products[0].sku); const h = useAsync(() => api.holdout(sku), [sku]);
  if (loading) return <Loading />; if (!m) return <Err msg={error} />;
  const models: [string, string][] = [["gradient_boosting", "Gradient boosting (production)"], ["naive_lag28", "Naive: demand 28 days ago"], ["moving_avg_28", "28-day moving average"], ["blend_lag28_35", "Blend of lag 28 & 35"]];
  return <div className="page"><div className="kpis"><Kpi label="Model WAPE" value={pct(m.gradient_boosting.WAPE)} sub={`${m.holdout_days}-day holdout, 28-day-ahead`} /><Kpi label="Best baseline WAPE" value={pct(Math.min(...models.slice(1).map(([k]) => m[k].WAPE)))} /><Kpi label="Model MAE" value={m.gradient_boosting.MAE.toFixed(2)} sub="units / day / SKU" /></div>
    <Card title="Model vs baselines"><table className="tbl"><thead><tr><th>Method</th><th className="num">MAE</th><th className="num">RMSE</th><th className="num">WAPE</th></tr></thead><tbody>{models.map(([k, n], i) => <tr key={k} className={i === 0 ? "rec" : ""}><td>{n}</td><td className="num">{m[k].MAE.toFixed(2)}</td><td className="num">{m[k].RMSE.toFixed(2)}</td><td className="num">{pct(m[k].WAPE)}</td></tr>)}</tbody></table></Card>
    <Card title="Per-SKU performance"><table className="tbl"><thead><tr><th>SKU</th><th className="num">MAE</th><th className="num">WAPE</th><th className="num">Naive WAPE</th><th className="num">MA-28 WAPE</th><th className="num">Bias</th><th className="num">80% interval coverage</th></tr></thead>
      <tbody>{Object.entries<any>(m.per_sku).map(([k, v]) => <tr key={k}><td>{k}</td><td className="num">{v.MAE.toFixed(2)}</td><td className="num">{pct(v.WAPE)}</td><td className="num">{pct(v.naive_WAPE)}</td><td className="num">{pct(v.ma28_WAPE)}</td><td className="num">{v.bias.toFixed(2)}</td><td className="num">{pct(v.coverage_80)}</td></tr>)}</tbody></table></Card>
    <p className="muted small">Coverage: share of second-half holdout days inside the 80% interval, with sigma estimated on the first half only. Metrics are on simulated data.</p>
    <Card title="Holdout: actual vs forecast" right={<select value={sku} onChange={e => setSku(e.target.value)}>{products.map(p => <option key={p.sku} value={p.sku}>{p.sku} – {p.name}</option>)}</select>}>
      {h.data && <LineChart labels={h.data.map(x => x.date)} series={[{ name: "Actual", color: "#15202b", vals: h.data.map(x => x.units) }, { name: "Forecast", color: "#1f5f8b", vals: h.data.map(x => x.pred) }]} />}</Card></div>;
}

export function AuditPage({ runs }: { runs: { label: string; audit: Step[] }[] }) {
  const d = useAsync(() => api.decisions(), [runs.length]);
  return <div className="page"><p className="muted">Every request records which agent called which tool, with its reasoning and timing. Numbers come from deterministic tools; an optional LLM only narrates and is verified.</p>
    {!runs.length && <Card><span className="muted">Run a request from Overview, Planner or What-if to see its trace here.</span></Card>}
    {runs.map((r, i) => <Card key={i} title={r.label}><Trace audit={r.audit} /></Card>)}
    <Card title="Persisted decisions (SQLite)">{d.data && <table className="tbl"><thead><tr><th>ID</th><th>Time</th><th>Scope</th><th>Intent</th><th>Steps</th></tr></thead><tbody>{d.data.map(x => { let p: any = {}; try { p = JSON.parse(x.payload); } catch { /* ignore */ }
      return <tr key={x.id}><td>{x.id}</td><td>{x.ts}</td><td>{x.sku}</td><td>{p.intent ?? "product"}</td><td>{(p.audit ?? []).length}</td></tr>; })}</tbody></table>}</Card></div>;
}
