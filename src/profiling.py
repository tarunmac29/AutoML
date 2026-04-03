"""
profiling.py
Auto EDA / Data Profiling Report — generates a structured HTML report
without requiring pandas-profiling (which has heavy dependencies).
Pure pandas + matplotlib + jinja2 approach for portability.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
from datetime import datetime


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _profile_column(series: pd.Series) -> dict:
    """Compute per-column statistics."""
    stats = {
        "dtype": str(series.dtype),
        "count": int(series.count()),
        "missing": int(series.isnull().sum()),
        "missing_pct": round(series.isnull().mean() * 100, 2),
        "unique": int(series.nunique()),
        "unique_pct": round(series.nunique() / max(len(series), 1) * 100, 2),
    }
    if pd.api.types.is_numeric_dtype(series):
        stats.update({
            "mean": round(series.mean(), 4),
            "std": round(series.std(), 4),
            "min": round(series.min(), 4),
            "25%": round(series.quantile(0.25), 4),
            "50%": round(series.median(), 4),
            "75%": round(series.quantile(0.75), 4),
            "max": round(series.max(), 4),
            "skewness": round(series.skew(), 4),
            "kurtosis": round(series.kurtosis(), 4),
            "zeros": int((series == 0).sum()),
            "negatives": int((series < 0).sum()),
        })
    else:
        top = series.value_counts().head(5)
        stats["top_values"] = top.to_dict()
    return stats


def _mini_histogram(series: pd.Series) -> str:
    """Return base64 thumbnail histogram/bar chart for a column."""
    fig, ax = plt.subplots(figsize=(2.8, 1.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")

    if pd.api.types.is_numeric_dtype(series):
        ax.hist(series.dropna(), bins=20, color="#5DA5E8", edgecolor="white", linewidth=0.3)
    else:
        vc = series.value_counts().head(8)
        ax.bar(range(len(vc)), vc.values, color="#5DA5E8", edgecolor="white")
        ax.set_xticks(range(len(vc)))
        ax.set_xticklabels(vc.index.astype(str), rotation=30, ha="right", fontsize=6)

    ax.set_yticks([])
    ax.set_xticks(ax.get_xticks())
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(labelsize=6)
    plt.tight_layout(pad=0.2)
    return _fig_to_base64(fig)


def _correlation_heatmap_b64(df: pd.DataFrame) -> str:
    """Return base64 correlation heatmap image."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return ""
    corr = num_df.corr()
    size = max(5, min(12, corr.shape[0]))
    fig, ax = plt.subplots(figsize=(size, size * 0.8))
    fig.patch.set_facecolor("white")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=corr.shape[0] <= 12,
                fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.4, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation Matrix", fontsize=11)
    plt.tight_layout()
    return _fig_to_base64(fig)


# ── Main report generator ─────────────────────────────────────────────────────

def generate_profile_report(df: pd.DataFrame, title: str = "Dataset Profile Report") -> str:
    """
    Generate a self-contained HTML profiling report.

    Args:
        df: Input DataFrame
        title: Report title string

    Returns:
        Full HTML string (self-contained, embeddable)
    """
    n_rows, n_cols = df.shape
    n_numeric = df.select_dtypes(include=[np.number]).shape[1]
    n_categorical = df.select_dtypes(include=["object", "category"]).shape[1]
    total_missing = int(df.isnull().sum().sum())
    total_cells = n_rows * n_cols
    missing_pct = round(total_missing / max(total_cells, 1) * 100, 2)
    n_duplicates = int(df.duplicated().sum())
    memory_mb = round(df.memory_usage(deep=True).sum() / 1024 / 1024, 3)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Per-column profiles and thumbnails
    col_profiles = {}
    col_thumbs = {}
    for col in df.columns:
        col_profiles[col] = _profile_column(df[col])
        col_thumbs[col] = _mini_histogram(df[col])

    # Correlation heatmap
    corr_b64 = _correlation_heatmap_b64(df)

    # Missing values bar chart
    missing_series = df.isnull().sum()
    missing_series = missing_series[missing_series > 0].sort_values(ascending=False)
    missing_b64 = ""
    if not missing_series.empty:
        fig, ax = plt.subplots(figsize=(max(5, len(missing_series) * 0.6), 3))
        fig.patch.set_facecolor("white")
        ax.bar(missing_series.index.astype(str), missing_series.values, color="#E24B4A", edgecolor="white")
        ax.set_title("Missing Values per Column", fontsize=10)
        ax.set_ylabel("Count")
        plt.xticks(rotation=30, ha="right", fontsize=8)
        plt.tight_layout()
        missing_b64 = _fig_to_base64(fig)

    # ── Build HTML ────────────────────────────────────────────────────────────

    def stat_row(label, val, highlight=False):
        bg = "#fff8e1" if highlight else "transparent"
        return f'<tr style="background:{bg}"><td style="color:#555;padding:4px 8px">{label}</td><td style="font-weight:600;padding:4px 8px">{val}</td></tr>'

    overview_rows = (
        stat_row("Rows", f"{n_rows:,}") +
        stat_row("Columns", n_cols) +
        stat_row("Numeric features", n_numeric) +
        stat_row("Categorical features", n_categorical) +
        stat_row("Missing cells", f"{total_missing:,} ({missing_pct}%)", highlight=total_missing > 0) +
        stat_row("Duplicate rows", f"{n_duplicates:,}", highlight=n_duplicates > 0) +
        stat_row("Memory usage", f"{memory_mb} MB") +
        stat_row("Generated at", generated_at)
    )

    # Column detail cards
    col_cards_html = ""
    for col in df.columns:
        stats = col_profiles[col]
        thumb_src = col_thumbs[col]
        dtype_color = "#3182ce" if pd.api.types.is_numeric_dtype(df[col]) else "#805ad5"
        dtype_label = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"

        stat_items = ""
        skip_keys = {"dtype", "top_values"}
        for k, v in stats.items():
            if k in skip_keys:
                continue
            stat_items += f'<div style="display:flex;justify-content:space-between;padding:2px 0;border-bottom:0.5px solid #eee"><span style="color:#888;font-size:12px">{k}</span><span style="font-size:12px;font-weight:600">{v}</span></div>'

        if "top_values" in stats:
            tv = stats["top_values"]
            tv_html = "".join([f'<span style="background:#eef2ff;color:#4338ca;border-radius:4px;padding:1px 6px;font-size:11px;margin:1px">{k}: {v}</span>' for k, v in list(tv.items())[:5]])
            stat_items += f'<div style="margin-top:6px;font-size:11px;color:#888">Top values:</div><div style="margin-top:3px">{tv_html}</div>'

        missing_warn = ""
        if stats["missing"] > 0:
            missing_warn = f'<span style="background:#fff5f5;color:#e53e3e;border-radius:4px;padding:1px 7px;font-size:11px;margin-left:6px">⚠ {stats["missing_pct"]}% missing</span>'

        col_cards_html += f"""
        <div style="background:#fff;border:0.5px solid #e2e8f0;border-radius:10px;padding:14px;margin-bottom:12px">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
            <span style="font-weight:600;font-size:14px">{col}</span>
            <span style="background:{dtype_color}18;color:{dtype_color};border-radius:10px;padding:1px 9px;font-size:11px;font-weight:600">{dtype_label}</span>
            <span style="color:#aaa;font-size:11px">{stats['dtype']}</span>
            {missing_warn}
          </div>
          <div style="display:grid;grid-template-columns:1fr 200px;gap:14px;align-items:start">
            <div>{stat_items}</div>
            <img src="data:image/png;base64,{thumb_src}" style="width:100%;border-radius:6px" />
          </div>
        </div>
        """

    corr_section = ""
    if corr_b64:
        corr_section = f'<h2 style="font-size:16px;font-weight:600;color:#2d3748;border-bottom:2px solid #667eea;padding-bottom:6px;margin:24px 0 12px">Correlation Matrix</h2><img src="data:image/png;base64,{corr_b64}" style="max-width:100%;border-radius:8px" />'

    missing_section = ""
    if missing_b64:
        missing_section = f'<h2 style="font-size:16px;font-weight:600;color:#2d3748;border-bottom:2px solid #667eea;padding-bottom:6px;margin:24px 0 12px">Missing Values</h2><img src="data:image/png;base64,{missing_b64}" style="max-width:600px;border-radius:8px" />'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{title}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f7f8fc; color: #2d3748; padding: 24px; }}
  .container {{ max-width: 960px; margin: 0 auto; background: #fff; border-radius: 14px; padding: 28px; box-shadow: 0 2px 16px rgba(0,0,0,0.07); }}
  h1 {{ font-size: 22px; font-weight: 700; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 4px; }}
  .subtitle {{ color: #888; font-size: 13px; margin-bottom: 20px; }}
  .overview-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 24px; }}
  .ov-card {{ background: #f7f8fc; border-radius: 9px; padding: 12px; text-align: center; }}
  .ov-val {{ font-size: 22px; font-weight: 700; color: #2d3748; }}
  .ov-lbl {{ font-size: 11px; color: #888; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.04em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td {{ vertical-align: top; }}
</style>
</head>
<body>
<div class="container">
  <h1>🤖 {title}</h1>
  <div class="subtitle">Generated by Smart AutoML · {generated_at}</div>

  <div class="overview-grid">
    <div class="ov-card"><div class="ov-val">{n_rows:,}</div><div class="ov-lbl">Rows</div></div>
    <div class="ov-card"><div class="ov-val">{n_cols}</div><div class="ov-lbl">Columns</div></div>
    <div class="ov-card"><div class="ov-val">{n_numeric}</div><div class="ov-lbl">Numeric</div></div>
    <div class="ov-card"><div class="ov-val">{n_categorical}</div><div class="ov-lbl">Categorical</div></div>
    <div class="ov-card"><div class="ov-val" style="color:{'#e53e3e' if total_missing>0 else '#38a169'}">{missing_pct}%</div><div class="ov-lbl">Missing</div></div>
    <div class="ov-card"><div class="ov-val" style="color:{'#e53e3e' if n_duplicates>0 else '#38a169'}">{n_duplicates}</div><div class="ov-lbl">Duplicates</div></div>
    <div class="ov-card"><div class="ov-val">{memory_mb}</div><div class="ov-lbl">MB Memory</div></div>
  </div>

  <table style="margin-bottom:20px;font-size:13px;background:#f7f8fc;border-radius:8px;overflow:hidden">
    {overview_rows}
  </table>

  {missing_section}
  {corr_section}

  <h2 style="font-size:16px;font-weight:600;color:#2d3748;border-bottom:2px solid #667eea;padding-bottom:6px;margin:24px 0 12px">Column Details ({n_cols} columns)</h2>
  {col_cards_html}

  <div style="text-align:center;color:#aaa;font-size:12px;margin-top:20px">Smart AutoML Data Profiler</div>
</div>
</body>
</html>"""

    return html
