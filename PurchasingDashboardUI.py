# app.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================
# Page config
# ============================
st.set_page_config(
    page_title="OMP Purchasing Dashboard",
    page_icon="📦",
    layout="wide",
)


# ============================
# Olive dashboard CSS
# ============================
DASH_CSS = """
<style>
:root{
  --bg1: #6f8c34;
  --bg2: #5e7a2f;
  --bg3: #4f6927;

  --surface: rgba(40, 63, 22, 0.55);
  --surface2: rgba(30, 52, 16, 0.22);
  --border: rgba(255,255,255,0.10);

  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.68);

  --shadow: 0 12px 30px rgba(0,0,0,0.18);
  --shadow2: 0 6px 16px rgba(0,0,0,0.12);

  --rOuter: 28px;
  --r: 18px;

  --accent: #ff8a2a;
}

html, body, [data-testid="stAppViewContainer"]{
  height: 100%;
}

[data-testid="stAppViewContainer"]{
  background: linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 60%, var(--bg3) 100%) !important;
  color: var(--text) !important;
}

[data-testid="stHeader"]{
  background: transparent !important;
}

.block-container{
  padding-top: 28px !important;
  padding-bottom: 28px !important;
  max-width: 1300px !important;
}

.dashboard-shell{
  background: var(--surface2);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: var(--rOuter);
  padding: 18px;
  box-shadow: var(--shadow);
}

.card{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 14px 14px 12px 14px;
  box-shadow: var(--shadow2);
}

.card h3, .card h4, .card p, .card div{
  color: var(--text);
  margin: 0;
}

.kpi-label{
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 6px;
}

.kpi-value{
  font-size: 30px;
  font-weight: 750;
  line-height: 1.05;
}

.kpi-sub{
  margin-top: 6px;
  font-size: 13px;
  color: var(--muted);
}

.badge{
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.06);
  font-size: 12px;
  color: var(--muted);
}

.cta{
  display: inline-block;
  padding: 10px 14px;
  border-radius: 14px;
  border: 1px solid rgba(0,0,0,0.15);
  background: var(--accent);
  color: rgba(20,20,20,0.95);
  font-weight: 750;
  text-decoration: none;
}

hr{
  border-color: rgba(255,255,255,0.14) !important;
}

[data-testid="stSidebar"]{
  background: rgba(30, 52, 16, 0.25) !important;
  border-right: 1px solid rgba(255,255,255,0.08) !important;
}

[data-testid="stMarkdownContainer"]{
  color: var(--text);
}

.js-plotly-plot .plotly .modebar{
  background: transparent !important;
}
</style>
"""
st.markdown(DASH_CSS, unsafe_allow_html=True)


# ============================
# Helpers
# ============================
def fmt_num(x: Any, digits: int = 0) -> str:
    try:
        v = float(x)
        return f"{v:,.{digits}f}"
    except Exception:
        return "NA"


def card_kpi(label: str, value: str, sub: Optional[str] = None) -> None:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_default_json() -> Optional[Path]:
    """
    Preference:
      1) forecast_package_daily.json in ./output if present
      2) forecast_package_daily.json in current dir
      3) forecast_package.json in current dir
      4) newest *.json in ./output
    """
    cwd = Path(".")
    out = cwd / "output"

    candidates: List[Path] = []
    candidates.append(out / "forecast_package_daily.json")
    candidates.append(cwd / "forecast_package_daily.json")
    candidates.append(cwd / "forecast_package.json")

    for p in candidates:
        if p.exists():
            return p

    if out.exists():
        jsons = sorted(out.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if jsons:
            return jsons[0]
    return None


def sku_map(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    for s in data.get("skus", []) or []:
        item = str(s.get("item_number", "")).strip()
        if item:
            m[item] = s
    return m


def weights_map_monthly(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Older monthly JSON format: data["ensemble_weights"] = [{item_number, weights, rmse}, ...]
    """
    m: Dict[str, Dict[str, Any]] = {}
    for w in data.get("ensemble_weights", []) or []:
        item = str(w.get("item_number", "")).strip()
        if item:
            m[item] = w
    return m


def diagnostics_map_daily(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Daily format from OMPforecasting3.py: data["model_diagnostics"] = [{item_number, model_meta}, ...]
    """
    m: Dict[str, Dict[str, Any]] = {}
    for w in data.get("model_diagnostics", []) or []:
        item = str(w.get("item_number", "")).strip()
        if item:
            m[item] = w
    return m


def parse_forecast_series(sku: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    """
    Returns:
      df columns: date, forecast
      granularity: "daily" or "monthly"
    """
    if "ensemble_forecast_daily" in sku:
        fc = sku.get("ensemble_forecast_daily", {}) or {}
        df = pd.DataFrame(
            [{"date": pd.to_datetime(k), "forecast": float(v)} for k, v in fc.items()]
        ).sort_values("date").reset_index(drop=True)
        return df, "daily"

    fc = sku.get("ensemble_forecast", {}) or {}
    df = pd.DataFrame(
        [{"date": pd.to_datetime(k), "forecast": float(v)} for k, v in fc.items()]
    ).sort_values("date").reset_index(drop=True)
    return df, "monthly"


def rollup_daily(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    freq:
      "D" daily (no rollup)
      "W" weekly (sum)
      "MS" monthly (sum)
    """
    if df.empty:
        return df
    x = df.copy()
    x = x.set_index("date")
    if freq == "D":
        out = x.copy()
    else:
        out = x.resample(freq).sum()
    out = out.reset_index()
    return out


def plot_forecast(df: pd.DataFrame, baseline: Optional[float], label: str) -> go.Figure:
    fig = px.line(df, x="date", y="forecast", markers=True, title=None)
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.88)"),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            title=None,
            tickfont=dict(size=11, color="rgba(255,255,255,0.65)"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            title=None,
            tickfont=dict(size=11, color="rgba(255,255,255,0.65)"),
        ),
        showlegend=False,
    )

    if baseline is not None and np.isfinite(baseline):
        fig.add_hline(
            y=float(baseline),
            line_width=2,
            line_dash="dot",
            line_color="rgba(255,255,255,0.55)",
            annotation_text=label,
            annotation_position="top left",
            annotation_font_color="rgba(255,255,255,0.70)",
        )

    return fig


def plot_weights(weights: Dict[str, float]) -> go.Figure:
    df = pd.DataFrame(
        [{"model": k, "weight": float(v)} for k, v in (weights or {}).items()]
    ).sort_values("weight", ascending=False)
    fig = px.bar(df, x="model", y="weight", title=None)
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.88)"),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            title=None,
            tickfont=dict(size=11, color="rgba(255,255,255,0.65)"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            title=None,
            tickfont=dict(size=11, color="rgba(255,255,255,0.65)"),
            range=[0, 1],
        ),
        showlegend=False,
    )
    return fig


def cumulative_demand_chart(df_fc: pd.DataFrame, inventory_position: float, lead_time_days: Optional[float]) -> go.Figure:
    if df_fc.empty:
        return go.Figure()

    df = df_fc.copy()
    df = df.sort_values("date")
    df["cum"] = df["forecast"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["cum"], mode="lines+markers", name="Cumulative forecast"))
    fig.add_hline(y=float(inventory_position), line_dash="dot", line_width=2, opacity=0.8)

    if lead_time_days is not None and np.isfinite(lead_time_days) and lead_time_days > 0:
        # mark lead time point if forecast is daily and has enough points
        pass

    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.88)"),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            title=None,
            tickfont=dict(size=11, color="rgba(255,255,255,0.65)"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            title=None,
            tickfont=dict(size=11, color="rgba(255,255,255,0.65)"),
        ),
        showlegend=False,
    )

    return fig


def safe_read_text(path: Path) -> Optional[str]:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    return None


# ============================
# Sidebar: load JSON
# ============================
st.sidebar.markdown("### Data source")
uploaded = st.sidebar.file_uploader("Upload forecast JSON", type=["json"])

data: Dict[str, Any]
source_label: str

if uploaded is not None:
    data = json.load(uploaded)
    source_label = "Uploaded JSON"
else:
    default = find_default_json()
    if default is None:
        st.error("No JSON found. Upload one in the sidebar or place it in the folder (or ./output).")
        st.stop()
    data = load_json(default)
    source_label = f"Local file: {default.as_posix()}"

st.sidebar.markdown(f'<span class="badge">{source_label}</span>', unsafe_allow_html=True)

run_meta = data.get("run_meta", {}) or {}
counts = data.get("counts", {}) or {}
skus_by_item = sku_map(data)

if not skus_by_item:
    st.error("No SKUs found in JSON under key: skus")
    st.stop()

# detect schema flavor
is_daily = any("ensemble_forecast_daily" in s for s in skus_by_item.values())
is_monthly = any("ensemble_forecast" in s for s in skus_by_item.values())

weights_by_item = weights_map_monthly(data) if "ensemble_weights" in data else {}
diag_by_item = diagnostics_map_daily(data) if "model_diagnostics" in data else {}

# ============================
# Sidebar: Controls
# ============================
st.sidebar.markdown("---")
st.sidebar.markdown("### Controls")

# quick search
search = st.sidebar.text_input("Search (item, description, vendor)")

# build list for selector based on search
items_all = sorted(list(skus_by_item.keys()))
if search.strip():
    q = search.strip().lower()
    def match_item(k: str) -> bool:
        s = skus_by_item[k]
        blob = " ".join([
            str(k),
            str(s.get("description","")),
            str(s.get("vendor_number","")),
            str(s.get("vendor_name","")),
        ]).lower()
        return q in blob
    items = [k for k in items_all if match_item(k)]
else:
    items = items_all

if not items:
    st.warning("No matches for search.")
    items = items_all

selected_item = st.sidebar.selectbox("Item number", items, index=0)

# rollup controls for daily forecast readability
rollup = "D"
if is_daily:
    rollup = st.sidebar.selectbox("Forecast rollup", ["D", "W", "MS"], index=1, help="D daily, W weekly sum, MS month start sum")
st.sidebar.markdown("---")

# show SQL scripts if present
st.sidebar.markdown("### Data pull SQL")
sql1 = safe_read_text(Path("gartman_sku_master.sql"))
sql2 = safe_read_text(Path("gartman_sales_history_daily.sql"))
if sql1:
    with st.sidebar.expander("gartman_sku_master.sql"):
        st.code(sql1, language="sql")
if sql2:
    with st.sidebar.expander("gartman_sales_history_daily.sql"):
        st.code(sql2, language="sql")

st.sidebar.markdown("---")
st.sidebar.markdown("### Run meta")
st.sidebar.write(run_meta)

st.sidebar.markdown("### Counts")
st.sidebar.write(counts)


# ============================
# Build Purchasing Queue table
# ============================
def build_queue_df() -> pd.DataFrame:
    rows = []
    for item, sku in skus_by_item.items():
        available = float(sku.get("available_sf", 0.0) or 0.0)
        on_po = float(sku.get("on_po_sf", 0.0) or 0.0)
        backorder = float(sku.get("backorder_sf", 0.0) or 0.0)
        inv_pos = available + on_po - backorder

        lead_time_days = float(sku.get("lead_time_days", 0.0) or 0.0)

        df_fc, gran = parse_forecast_series(sku)
        next_30 = float(df_fc["forecast"].sum()) if (not df_fc.empty and gran == "daily") else np.nan

        # For daily: lead time demand = sum of first lead_time_days in forecast
        lt_demand = np.nan
        avg_daily = np.nan
        days_cover = np.nan
        risk = "NA"

        if not df_fc.empty and gran == "daily":
            avg_daily = float(df_fc["forecast"].mean()) if len(df_fc) else np.nan
            if lead_time_days > 0:
                n = int(max(1, round(lead_time_days)))
                lt_demand = float(df_fc.head(n)["forecast"].sum())
            if avg_daily and avg_daily > 0:
                days_cover = float(inv_pos / avg_daily)

            if np.isfinite(lt_demand):
                risk = "RISK" if inv_pos < lt_demand else "OK"

        # For monthly JSON, use planning if present
        order_qty = np.nan
        rop = np.nan
        target_position = np.nan
        if "planning" in sku and isinstance(sku.get("planning"), dict):
            pl = sku.get("planning") or {}
            order_qty = float(pl.get("order_qty", np.nan))
            rop = float(pl.get("rop", np.nan))
            target_position = float(pl.get("target_position", np.nan))
            if np.isfinite(rop):
                risk = "RISK" if inv_pos < rop else "OK"

        seg = sku.get("segmentation", {}) if isinstance(sku.get("segmentation"), dict) else {}
        rows.append({
            "Item": item,
            "Description": str(sku.get("description","")),
            "Vendor": str(sku.get("vendor_name","")),
            "Vend #": str(sku.get("vendor_number","")),
            "Available": available,
            "On PO": on_po,
            "Backorder": backorder,
            "Inv Position": inv_pos,
            "Lead Time (days)": lead_time_days,
            "LT Demand (forecast)": lt_demand,
            "Next 30d (forecast)": next_30,
            "Days Cover (forecast)": days_cover,
            "ROP": rop,
            "Target Pos": target_position,
            "Order Qty": order_qty,
            "Maturity": str(seg.get("maturity","")),
            "Pattern": str(seg.get("demand_pattern","")),
            "Risk": risk,
        })

    df = pd.DataFrame(rows)

    # Purchaser-friendly default sort: Risk then Order Qty then LT Demand
    def risk_rank(x: str) -> int:
        if x == "RISK":
            return 0
        if x == "OK":
            return 1
        return 2

    df["RiskRank"] = df["Risk"].map(risk_rank)
    df = df.sort_values(["RiskRank", "Order Qty", "LT Demand (forecast)"], ascending=[True, False, False])
    df = df.drop(columns=["RiskRank"])
    return df


queue_df = build_queue_df()

# ============================
# Header
# ============================
sku = skus_by_item[selected_item]
title = f'{sku.get("item_number", "")}  |  {sku.get("description", "")}'
subtitle = f'{sku.get("vendor_name","")} (Vendor {sku.get("vendor_number","")})'

run_line_parts = []
if "forecast_horizon_days" in run_meta:
    run_line_parts.append(f'Horizon: {run_meta.get("forecast_horizon_days")} days')
if "forecast_horizon_months" in run_meta:
    run_line_parts.append(f'Horizon: {run_meta.get("forecast_horizon_months")} months')
if run_meta.get("forecast_start_date"):
    run_line_parts.append(f'Forecast start: {run_meta.get("forecast_start_date")}')
if run_meta.get("run_timestamp_local"):
    run_line_parts.append(f'Run: {run_meta.get("run_timestamp_local")}')
run_line = " | ".join(run_line_parts)

st.markdown(
    f"""
<div class="dashboard-shell">
  <div style="display:flex; justify-content:space-between; gap:12px; align-items:flex-start; flex-wrap:wrap;">
    <div>
      <div class="badge">OMP Purchasing Dashboard</div>
      <h1 style="margin:10px 0 0 0; font-size:44px; line-height:1.0;">{title}</h1>
      <div style="margin-top:8px; color: rgba(255,255,255,0.72); font-size:14px;">{subtitle}</div>
    </div>
    <div style="display:flex; gap:10px; align-items:center;">
      <a class="cta" href="#" onclick="return false;">Export tools below</a>
    </div>
  </div>
  <div style="margin-top:14px; color: rgba(255,255,255,0.70); font-size:13px;">
    {run_line}
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

# ============================
# Top: Purchasing Queue
# ============================
st.markdown('<div class="card"><div class="kpi-label">Purchasing queue</div></div>', unsafe_allow_html=True)

cqa, cqb = st.columns([3, 1], gap="large")
with cqa:
    st.dataframe(queue_df, use_container_width=True, hide_index=True)

with cqb:
    # Export buttons
    csv_bytes = queue_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download queue CSV", data=csv_bytes, file_name="purchasing_queue.csv", mime="text/csv")

    # quick filters
    show_only_risk = st.checkbox("Show only Risk items", value=False)
    if show_only_risk:
        st.dataframe(queue_df[queue_df["Risk"] == "RISK"], use_container_width=True, hide_index=True)

st.write("")


# ============================
# SKU drill-down
# ============================
planning = sku.get("planning", {}) if isinstance(sku.get("planning"), dict) else {}
seg = sku.get("segmentation", {}) if isinstance(sku.get("segmentation"), dict) else {}

available = float(sku.get("available_sf", 0.0) or 0.0)
on_po = float(sku.get("on_po_sf", 0.0) or 0.0)
backorder = float(sku.get("backorder_sf", 0.0) or 0.0)
inventory_position = available + on_po - backorder
lead_time_days = float(sku.get("lead_time_days", 0.0) or 0.0)

df_fc_raw, gran = parse_forecast_series(sku)
df_fc = df_fc_raw.copy()

# optional rollup for daily
if gran == "daily":
    df_fc = rollup_daily(df_fc, rollup)

# Baseline handling
baseline_val = None
baseline_label = ""
if "mean_hist_monthly" in sku:
    baseline_val = sku.get("mean_hist_monthly")
    baseline_label = "Mean historical monthly"
elif gran == "daily" and not df_fc_raw.empty and rollup == "D":
    # simple baseline for daily: mean forecast
    baseline_val = float(df_fc_raw["forecast"].mean())
    baseline_label = "Mean forecast (daily)"

# Demand in lead time for daily (use raw daily forecast, not rolled up)
lt_demand = None
days_cover = None
avg_daily = None
if gran == "daily" and not df_fc_raw.empty:
    avg_daily = float(df_fc_raw["forecast"].mean())
    if lead_time_days > 0:
        n = int(max(1, round(lead_time_days)))
        lt_demand = float(df_fc_raw.head(n)["forecast"].sum())
    if avg_daily > 0:
        days_cover = float(inventory_position / avg_daily)

# Monthly planning fields if present
rop = planning.get("rop", None)
target_position = planning.get("target_position", None)
order_qty = planning.get("order_qty", None)

colA, colB, colC = st.columns([2.0, 1.2, 1.2], gap="large")

with colA:
    st.markdown('<div class="card"><div class="kpi-label">Forecast</div></div>', unsafe_allow_html=True)
    fig_fc = plot_forecast(df_fc, baseline_val if baseline_val is not None else None, baseline_label if baseline_label else "Baseline")
    st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown('<div class="card"><div class="kpi-label">Cumulative demand vs inventory position</div></div>', unsafe_allow_html=True)
    # For cumulative, only meaningful for daily series, but still show for rolled up
    fig_cum = cumulative_demand_chart(df_fc, inventory_position=inventory_position, lead_time_days=lead_time_days)
    st.plotly_chart(fig_cum, use_container_width=True)

with colB:
    card_kpi("Inventory position", fmt_num(inventory_position, 2),
             f"Available {fmt_num(available,2)} | On PO {fmt_num(on_po,2)} | BO {fmt_num(backorder,2)}")
    st.write("")
    card_kpi("Lead time", f"{fmt_num(lead_time_days,0)} days", f"Vendor {sku.get('vendor_number','')}")
    st.write("")

    if lt_demand is not None:
        risk_text = "RISK" if inventory_position < lt_demand else "OK"
        card_kpi("Demand during lead time", fmt_num(lt_demand, 2), f"Status {risk_text}")
    elif rop is not None and np.isfinite(float(rop)):
        risk_text = "RISK" if inventory_position < float(rop) else "OK"
        card_kpi("Reorder point (ROP)", fmt_num(float(rop), 2), f"Status {risk_text}")
    else:
        card_kpi("Demand during lead time", "NA", "Daily forecast required for LT demand")

    st.write("")
    if days_cover is not None:
        card_kpi("Days of cover", fmt_num(days_cover, 1), f"Avg daily demand {fmt_num(avg_daily,2)}")
    else:
        card_kpi("Days of cover", "NA", "Daily forecast required for cover")

with colC:
    # If monthly planning exists, show it. Otherwise show segmentation.
    if order_qty is not None and np.isfinite(float(order_qty)):
        card_kpi("Suggested order qty", fmt_num(float(order_qty), 2), "From planning output in JSON")
        st.write("")
        if target_position is not None and np.isfinite(float(target_position)):
            card_kpi("Target position", fmt_num(float(target_position), 2), "From planning output in JSON")
        st.write("")
    else:
        card_kpi("Suggested order qty", "NA", "Daily JSON does not include planning yet")
        st.write("")

    st.markdown('<div class="card"><div class="kpi-label">Segmentation</div></div>', unsafe_allow_html=True)
    if seg:
        seg_df = pd.DataFrame([
            {"Field": k, "Value": v} for k, v in seg.items()
        ])
        st.dataframe(seg_df, use_container_width=True, hide_index=True)
    else:
        st.info("No segmentation block found for this SKU.")

    st.write("")
    st.markdown('<div class="card"><div class="kpi-label">Model diagnostics</div></div>', unsafe_allow_html=True)

    # Monthly: show ensemble weights + RMSE
    if selected_item in weights_by_item:
        wb = weights_by_item[selected_item] or {}
        weights = wb.get("weights", {}) or {}
        rmse = wb.get("rmse", {}) or {}

        if weights:
            st.plotly_chart(plot_weights(weights), use_container_width=True)
        if rmse:
            rmse_df = (pd.DataFrame([{"model": k, "rmse": float(v)} for k, v in rmse.items()])
                       .sort_values("rmse")
                       .reset_index(drop=True))
            st.dataframe(rmse_df, use_container_width=True, hide_index=True)

    # Daily: show model_meta as a table if possible, otherwise JSON
    elif selected_item in diag_by_item:
        mb = diag_by_item[selected_item] or {}
        meta = mb.get("model_meta", {})
        if isinstance(meta, dict):
            # try to flatten one level
            flat_rows = []
            for k, v in meta.items():
                if isinstance(v, (dict, list)):
                    flat_rows.append({"Key": k, "Value": json.dumps(v)[:1000]})
                else:
                    flat_rows.append({"Key": k, "Value": v})
            st.dataframe(pd.DataFrame(flat_rows), use_container_width=True, hide_index=True)
            with st.expander("Raw model_meta JSON"):
                st.json(meta)
        else:
            st.write(meta)

    else:
        st.info("No diagnostics found for this SKU.")


# ============================
# Bottom: Forecast table exports
# ============================
st.write("")
b1, b2 = st.columns([1.6, 1.4], gap="large")

with b1:
    st.markdown('<div class="card"><div class="kpi-label">Forecast table</div></div>', unsafe_allow_html=True)
    df_show = df_fc.copy()
    df_show["period"] = df_show["date"].dt.strftime("%Y-%m-%d" if rollup == "D" else "%Y-%m-%d")
    df_show = df_show[["period", "forecast"]]
    st.dataframe(df_show, use_container_width=True, hide_index=True)

    csv_fc = df_show.to_csv(index=False).encode("utf-8")
    st.download_button("Download forecast CSV", data=csv_fc, file_name=f"{selected_item}_forecast.csv", mime="text/csv")

with b2:
    st.markdown('<div class="card"><div class="kpi-label">SKU snapshot</div></div>', unsafe_allow_html=True)
    snapshot = {
        "item_number": sku.get("item_number", ""),
        "description": sku.get("description", ""),
        "vendor_number": sku.get("vendor_number", ""),
        "vendor_name": sku.get("vendor_name", ""),
        "available_sf": available,
        "on_po_sf": on_po,
        "backorder_sf": backorder,
        "inventory_position": inventory_position,
        "lead_time_days": lead_time_days,
        "lt_demand_forecast": lt_demand,
        "days_cover_forecast": days_cover,
        "json_schema": "daily" if is_daily else "monthly",
    }
    st.dataframe(pd.DataFrame([{"Field": k, "Value": v} for k, v in snapshot.items()]), use_container_width=True, hide_index=True)

st.write("")
st.markdown(
    """
<div class="dashboard-shell">
  <div class="kpi-label">Notes</div>
  <div style="color: rgba(255,255,255,0.74); font-size: 13px;">
    This dashboard supports both JSON formats:
    monthly JSON (ensemble_forecast + planning + ensemble_weights) and daily JSON
    (ensemble_forecast_daily + segmentation + model_diagnostics).
    If you want true purchasing recommendations (ROP, safety stock, MOQ, order qty) for daily JSON,
    add those fields to the JSON payload and this UI will display them automatically.
  </div>
</div>
""",
    unsafe_allow_html=True,
)
