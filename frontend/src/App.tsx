import { useEffect, useState } from "react";
import { api, Step } from "./api";
import { AuditPage, ModelPage, Overview, Planner, PortfolioPage, SuppliersPage, WhatIfPage } from "./pages";
import { Err, Loading, useAsync } from "./ui";
const NAV: [string, [string, string][]][] = [
  ["Operate", [["overview", "Overview"], ["planner", "Product planner"], ["portfolio", "Portfolio risk"], ["suppliers", "Suppliers"]]],
  ["Analyze", [["whatif", "What-if scenarios"], ["model", "Model evaluation"]]], ["Governance", [["audit", "Agent audit trail"]]]];
export default function App() {
  const [page, setPage] = useState(location.hash.slice(1) || "overview"); const [runs, setRuns] = useState<{ label: string; audit: Step[] }[]>([]); const [up, setUp] = useState<boolean | null>(null);
  const prods = useAsync(() => api.products(), []);
  useEffect(() => { const f = () => setPage(location.hash.slice(1) || "overview"); addEventListener("hashchange", f); return () => removeEventListener("hashchange", f); }, []);
  useEffect(() => { api.health().then(() => setUp(true)).catch(() => setUp(false)); }, []);
  const log = (label: string, audit: Step[]) => setRuns(r => [{ label, audit }, ...r].slice(0, 15)); const title = NAV.flatMap(g => g[1]).find(n => n[0] === page)?.[1] ?? "Overview";
  const P = prods.data;
  return <div className="app"><aside className="side"><div className="brand"><div className="logo">S</div>Stockline</div>
    {NAV.map(([g, items]) => <div key={g}><div className="grp">{g}</div>{items.map(([k, n]) => <button key={k} className={`nav ${page === k ? "on" : ""}`} onClick={() => { location.hash = k; }}>{n}</button>)}</div>)}</aside>
    <div className="main"><div className="top"><h1>{title}</h1><div><span className="badge sim">Simulated data</span><span className="badge"><i className={`dot ${up ? "up" : ""}`} />{up === null ? "Connecting" : up ? "API connected" : "API offline"}</span></div></div>
      {up === false && <div className="page"><Err msg="Cannot reach the API. Start it with: uvicorn inventory_ai.api:app --reload (default http://localhost:8000)." /></div>}
      {prods.loading ? <Loading /> : !P ? (up !== false && <div className="page"><Err msg={prods.error} /></div>) :
        page === "planner" ? <Planner products={P} log={log} /> : page === "portfolio" ? <PortfolioPage /> : page === "suppliers" ? <SuppliersPage products={P} /> :
        page === "whatif" ? <WhatIfPage products={P} log={log} /> : page === "model" ? <ModelPage products={P} /> : page === "audit" ? <AuditPage runs={runs} /> : <Overview log={log} />}</div></div>;
}
