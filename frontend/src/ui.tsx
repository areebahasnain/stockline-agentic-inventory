import { ReactNode, useEffect, useState } from "react";
export const money = (n: number) => "$" + Math.round(n).toLocaleString();
export const pct = (n: number) => Math.round(n * 100) + "%";
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]) {
  const [s, set] = useState<{ data?: T; error?: string; loading: boolean }>({ loading: true });
  useEffect(() => { let ok = true; set(p => ({ ...p, loading: true, error: undefined }));
    fn().then(d => ok && set({ data: d, loading: false })).catch(e => ok && set({ loading: false, error: String(e.message ?? e) })); return () => { ok = false; }; }, deps); // eslint-disable-line
  return s;
}
export const Pill = ({ level }: { level: string }) => <span className={`pill ${level.toLowerCase()}`}>{level.replace("_", " ")}</span>;
export const Rich = ({ text }: { text: string }) => <>{text.split("**").map((p, i) => (i % 2 ? <strong key={i}>{p}</strong> : p))}</>;
export const Kpi = ({ label, value, sub, tone }: { label: string; value: ReactNode; sub?: string; tone?: string }) =>
  <div className="kpi"><div className="kpi-l">{label}</div><div className={`kpi-v ${tone ?? ""}`}>{value}</div>{sub && <div className="kpi-s">{sub}</div>}</div>;
export const Card = ({ title, right, children }: { title?: string; right?: ReactNode; children: ReactNode }) =>
  <section className="card">{title && <header><h3>{title}</h3>{right}</header>}<div className="card-b">{children}</div></section>;
export const Err = ({ msg }: { msg?: string }) => msg ? <div className="err">{msg}</div> : null;
export const Loading = () => <div className="muted pad">Loading…</div>;
export const Cover = ({ days, level }: { days: number; level: string }) =>
  <div className="cover"><div className={`bar ${level.toLowerCase()}`} style={{ width: `${Math.min(100, days / 1.5)}%` }} /><span>{days.toFixed(0)}d</span></div>;

type S = { name: string; vals: (number | null)[]; color: string; dash?: boolean };
export function LineChart({ series, band, labels, h = 250 }: { series: S[]; band?: { lo: (number | null)[]; hi: (number | null)[] }; labels: string[]; h?: number }) {
  const W = 760, p = { l: 40, r: 10, t: 10, b: 22 }, n = labels.length;
  const all = series.flatMap(s => s.vals).concat(band?.hi ?? []).filter((v): v is number => v != null); const max = Math.max(1, ...all) * 1.08;
  const x = (i: number) => p.l + (i / Math.max(1, n - 1)) * (W - p.l - p.r), y = (v: number) => h - p.b - (v / max) * (h - p.t - p.b);
  const path = (v: (number | null)[]) => { let d = "", pen = false; v.forEach((a, i) => { if (a == null) { pen = false; return; } d += (pen ? "L" : "M") + x(i).toFixed(1) + " " + y(a).toFixed(1); pen = true; }); return d; };
  const idx = band ? band.hi.map((_, i) => i).filter(i => band.hi[i] != null && band.lo[i] != null) : [];
  const poly = band ? idx.map(i => `${x(i)},${y(band.hi[i]!)}`).concat(idx.slice().reverse().map(i => `${x(i)},${y(band.lo[i]!)}`)).join(" ") : "";
  return <div><svg viewBox={`0 0 ${W} ${h}`} className="chart">
    {[0, .25, .5, .75, 1].map(t => <g key={t}><line x1={p.l} x2={W - p.r} y1={y(max * t / 1.08)} y2={y(max * t / 1.08)} className="grid" /><text x={p.l - 6} y={y(max * t / 1.08) + 3} textAnchor="end" className="ax">{Math.round(max * t / 1.08)}</text></g>)}
    {[0, .25, .5, .75, 1].map(t => { const i = Math.round((n - 1) * t); return <text key={t} x={x(i)} y={h - 6} textAnchor="middle" className="ax">{labels[i]?.slice(5)}</text>; })}
    {band && <polygon points={poly} fill="#2f6f9f" opacity=".14" />}
    {series.map(s => <path key={s.name} d={path(s.vals)} fill="none" stroke={s.color} strokeWidth={2} strokeDasharray={s.dash ? "4 3" : undefined} />)}
  </svg><div className="legend">{series.map(s => <span key={s.name}><i style={{ background: s.color }} />{s.name}</span>)}{band && <span><i style={{ background: "#2f6f9f", opacity: .35 }} />80% interval</span>}</div></div>;
}
