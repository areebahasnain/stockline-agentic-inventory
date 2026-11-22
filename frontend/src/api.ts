const BASE: string = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
export interface Product { sku: string; name: string; category: string; unit_cost: number; on_hand: number; on_order: number }
export interface Risk { level: string; p_stockout_in_lead_time: number; cover_days: number; lead_time_used: number; lead_time_demand: number; excess_units: number }
export interface Order { action: string; order_qty: number; est_cost: number; unit_price: number; moq: number; supplier: string | null; funded_qty?: number; funded_cost?: number;
  order_in_days?: number | null; order_date?: string; safety_stock?: number; reorder_point?: number; confidence?: number; confidence_label?: string; lead_days?: number }
export interface Item { sku: string; name: string; on_hand: number; on_order: number; mean_daily: number; risk: Risk; order: Order; priority: number | null }
export interface Summary { n_products: number; at_risk: string[]; critical: string[]; overstock: string[]; required_spend: number; funded_spend: number; budget: number | null; unfunded: string[]; no_supplier: string[] }
export interface Portfolio { summary: Summary; items: Item[] }
export interface Step { agent: string; tool: string; reasoning: string; ms: number; input?: unknown; output?: unknown }
export interface Forecast { dates: string[]; daily: number[]; sigma: number; wape: number; model: string; mean_daily: number }
export interface ProductResult { kind: "product"; product: Product & { name: string }; forecast: Forecast; risk: Risk; order: Order | null; suppliers: any[] | null; report: string }
export interface Ask { intent: string; plan: string[]; result: any; report: string; audit: Step[] }
export interface Hold { date: string; sku: string; units: number; pred: number }
export interface WhatIf { scope: string; scenario: any; baseline: Record<string, any>; projected: Record<string, any>; delta: Record<string, number>; report: string; audit: Step[] }
export interface Metrics { holdout_days: number; [k: string]: any }

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(BASE + path, { headers: { "Content-Type": "application/json" }, ...init });
  if (!r.ok) { let d: unknown = r.statusText; try { d = (await r.json()).detail ?? d; } catch { /* ignore */ } throw new Error(typeof d === "string" ? d : JSON.stringify(d)); }
  return r.json() as Promise<T>;
}
const post = (b: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(b) });
export const api = {
  health: () => req<{ ok: boolean }>("/health"),
  products: () => req<Product[]>("/products"),
  portfolio: (budget?: number) => req<Portfolio>(`/portfolio${budget ? `?budget=${budget}` : ""}`),
  ask: (b: { question?: string; sku?: string | null; budget?: number | null }) => req<Ask>("/ask", post(b)),
  whatif: (b: unknown) => req<WhatIf>("/whatif", post(b)),
  suppliers: (sku: string) => req<any[]>(`/suppliers/${sku}`),
  metrics: () => req<Metrics>("/metrics"),
  holdout: (sku: string) => req<Hold[]>(`/holdout/${sku}`),
  decisions: () => req<{ id: number; ts: string; sku: string; payload: string }[]>("/decisions?n=25"),
};
