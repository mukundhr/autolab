import os
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from database import DB_NAME

COLORS = {
    "primary": "#2563eb",
    "success": "#16a34a",
    "warning": "#ea580c",
    "accent": "#7c3aed",
    "bg": "#f8fafc",
    "card": "#ffffff",
    "text": "#1e293b",
    "muted": "#64748b",
}

CHART_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, system-ui, sans-serif", size=13, color=COLORS["text"]),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#f1f5f9",
    margin=dict(l=50, r=30, t=50, b=50),
    height=400,
)


def load_data():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM experiments ORDER BY id", conn)
    conn.close()
    if df.empty:
        print("No experiments in the database yet. Run main_loop.py first.")
        exit(0)
    return df


# ── Chart builders ──────────────────────────────────────────────────────────


def chart_accuracy_timeline(df):
    """Each dot is one experiment. The dashed red line tracks the best so far."""
    df = df.copy()
    df["exp_num"] = range(1, len(df) + 1)
    df["running_best"] = df["accuracy"].cummax()
    df["label"] = (
        df["filters"].astype(str) + "f / "
        + df["num_layers"].astype(str) + "L / "
        + df["optimizer"] + " / lr="
        + df["learning_rate"].astype(str)
    )

    best_idx = df["accuracy"].idxmax()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["exp_num"], y=df["accuracy"],
        mode="markers+lines",
        marker=dict(size=10, color=COLORS["primary"], line=dict(width=1, color="white")),
        line=dict(width=1, color=COLORS["primary"], dash="dot"),
        text=df["label"], hovertemplate="%{text}<br>Accuracy: %{y:.2f}%<extra></extra>",
        name="Experiment",
    ))
    fig.add_trace(go.Scatter(
        x=df["exp_num"], y=df["running_best"],
        mode="lines", line=dict(width=2, color="#dc2626", dash="dash"),
        name="Best so far",
    ))
    # annotate best point
    fig.add_annotation(
        x=df.loc[best_idx, "exp_num"], y=df.loc[best_idx, "accuracy"],
        text=f"Best: {df.loc[best_idx, 'accuracy']:.2f}%",
        showarrow=True, arrowhead=2, arrowcolor="#dc2626",
        font=dict(size=12, color="#dc2626", weight="bold"),
        bgcolor="white", bordercolor="#dc2626", borderpad=4,
    )
    fig.update_layout(
        **CHART_LAYOUT,
        xaxis_title="Experiment #", yaxis_title="Accuracy (%)",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    return fig


def chart_cycle_progress(df):
    """Shows how accuracy improves (or not) with each agent cycle."""
    cycle_stats = df.groupby("cycle")["accuracy"].agg(["mean", "max", "min"]).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cycle_stats["cycle"], y=cycle_stats["mean"],
        name="Average", marker_color=COLORS["primary"],
        text=cycle_stats["mean"].round(1), textposition="outside",
    ))
    fig.add_trace(go.Scatter(
        x=cycle_stats["cycle"], y=cycle_stats["max"],
        mode="markers+lines", name="Best in cycle",
        marker=dict(size=10, color=COLORS["success"], symbol="diamond"),
        line=dict(width=2, color=COLORS["success"]),
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        xaxis_title="Agent Cycle", yaxis_title="Accuracy (%)",
        xaxis=dict(dtick=1),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        barmode="overlay",
    )
    return fig


def chart_heatmap(df):
    """Darker = higher accuracy. Numbers inside cells show the value."""
    pivot = df.pivot_table(values="accuracy", index="num_layers", columns="filters", aggfunc="mean")
    pivot = pivot.sort_index(ascending=False)

    fig = px.imshow(
        pivot,
        text_auto=".1f",
        color_continuous_scale=["#fee2e2", "#fca5a5", "#f87171", "#dc2626", "#16a34a", "#15803d"],
        labels=dict(x="Number of Filters", y="Number of Conv Layers", color="Avg Accuracy %"),
        aspect="auto",
    )
    layout_opts = {k: v for k, v in CHART_LAYOUT.items() if k != "height"}
    fig.update_layout(**layout_opts, height=350)
    fig.update_traces(textfont_size=14)
    return fig


def chart_optimizer_lr(df):
    """Side-by-side bars for each optimizer + learning rate combo."""
    grouped = df.groupby(["optimizer", "learning_rate"])["accuracy"].agg(
        ["mean", "count"]
    ).reset_index()
    grouped.columns = ["optimizer", "learning_rate", "avg_accuracy", "num_runs"]
    grouped["combo"] = grouped["optimizer"].str.upper() + "  (lr=" + grouped["learning_rate"].astype(str) + ")"
    grouped = grouped.sort_values("avg_accuracy", ascending=True)

    fig = go.Figure(go.Bar(
        y=grouped["combo"], x=grouped["avg_accuracy"],
        orientation="h",
        marker_color=[COLORS["primary"], COLORS["accent"], COLORS["success"], COLORS["warning"]][:len(grouped)],
        text=grouped.apply(lambda r: f"{r['avg_accuracy']:.1f}%  ({int(r['num_runs'])} runs)", axis=1),
        textposition="outside",
    ))
    layout_opts = {k: v for k, v in CHART_LAYOUT.items() if k != "height"}
    fig.update_layout(
        **layout_opts,
        xaxis_title="Average Accuracy (%)", yaxis_title="",
        height=max(250, 70 * len(grouped)),
    )
    return fig


def chart_top_configs(df):
    """Horizontal bar chart ranking every tested configuration."""
    df_plot = df.copy()
    df_plot["config"] = (
        df_plot["filters"].astype(str) + " filters, "
        + df_plot["num_layers"].astype(str) + " layers, "
        + df_plot["optimizer"] + ", lr=" + df_plot["learning_rate"].astype(str)
    )
    ranked = df_plot.groupby("config")["accuracy"].agg(["mean", "count"]).reset_index()
    ranked.columns = ["config", "avg_accuracy", "runs"]
    ranked = ranked.sort_values("avg_accuracy", ascending=True)

    bar_colors = [COLORS["success"] if i == len(ranked) - 1 else COLORS["primary"]
                  for i in range(len(ranked))]

    fig = go.Figure(go.Bar(
        y=ranked["config"],
        x=ranked["avg_accuracy"],
        orientation="h",
        marker_color=bar_colors,
        text=ranked.apply(lambda r: f"{r['avg_accuracy']:.2f}%  ({int(r['runs'])} run{'s' if r['runs']>1 else ''})", axis=1),
        textposition="outside",
    ))
    layout_opts = {k: v for k, v in CHART_LAYOUT.items() if k != "height"}
    fig.update_layout(
        **layout_opts,
        xaxis_title="Average Accuracy (%)",
        yaxis_title="",
        height=max(300, 35 * len(ranked)),
    )
    return fig


def chart_hyperparameter_impact(df):
    """Shows how each hyperparameter individually affects accuracy."""
    from plotly.subplots import make_subplots

    params = [
        ("filters", "Filters", [16, 32, 64, 128]),
        ("num_layers", "Conv Layers", [1, 2, 3]),
        ("optimizer", "Optimizer", None),
        ("learning_rate", "Learning Rate", None),
    ]

    fig = make_subplots(rows=1, cols=4, subplot_titles=[p[1] for p in params])

    for i, (col, label, _) in enumerate(params, 1):
        grouped = df.groupby(col)["accuracy"].agg(["mean", "std", "count"]).reset_index()
        grouped = grouped.sort_values("mean")
        grouped["std"] = grouped["std"].fillna(0)
        x_labels = grouped[col].astype(str)

        fig.add_trace(go.Bar(
            x=x_labels, y=grouped["mean"],
            error_y=dict(type="data", array=grouped["std"], visible=True, color="#94a3b8"),
            marker_color=COLORS["primary"],
            text=grouped["mean"].round(1), textposition="outside",
            showlegend=False,
        ), row=1, col=i)

    layout_opts = {k: v for k, v in CHART_LAYOUT.items() if k != "height"}
    fig.update_layout(**layout_opts, height=370)
    fig.update_yaxes(title_text="Avg Accuracy (%)", row=1, col=1)
    return fig


# ── Summary stats ───────────────────────────────────────────────────────────


def build_summary_html(df):
    """Generate a stats banner at the top of the dashboard."""
    best = df.loc[df["accuracy"].idxmax()]
    worst = df.loc[df["accuracy"].idxmin()]
    avg = df["accuracy"].mean()
    n_configs = df.groupby(["filters", "num_layers", "learning_rate", "optimizer"]).ngroups

    cards = [
        ("Total Experiments", str(len(df)), COLORS["primary"]),
        ("Agent Cycles", str(df["cycle"].nunique()), COLORS["accent"]),
        ("Unique Configs Tested", str(n_configs), COLORS["warning"]),
        ("Best Accuracy", f"{best['accuracy']:.2f}%", COLORS["success"]),
        ("Average Accuracy", f"{avg:.2f}%", COLORS["primary"]),
        ("Worst Accuracy", f"{worst['accuracy']:.2f}%", COLORS["warning"]),
    ]

    best_config = (
        f"{int(best['filters'])} filters &middot; "
        f"{int(best['num_layers'])} layers &middot; "
        f"{best['optimizer']} &middot; lr={best['learning_rate']}"
    )

    cards_html = "".join(
        f"<div class='stat-card'>"
        f"  <div class='stat-value' style='color:{color}'>{value}</div>"
        f"  <div class='stat-label'>{label}</div>"
        f"</div>"
        for label, value, color in cards
    )

    return f"""
    <div class="summary">
        <div class="stats-row">{cards_html}</div>
        <div class="best-banner">
            <span class="best-icon">&#9733;</span>
            Best configuration: <strong>{best_config}</strong>
            &mdash; found in cycle {int(best['cycle'])}
        </div>
    </div>
    """


# ── Dashboard HTML builder ──────────────────────────────────────────────────

CHART_SECTIONS = [
    (
        "Accuracy Over Time",
        "Each dot is one experiment in the order it was run. "
        "The <span style='color:#dc2626;font-weight:600'>red dashed line</span> "
        "tracks the best accuracy found so far. "
        "An upward trend means the LLM agent is learning from past results.",
        chart_accuracy_timeline,
    ),
    (
        "Progress by Agent Cycle",
        "Each cycle, the LLM proposes a batch of experiments. "
        "The <span style='color:#2563eb;font-weight:600'>blue bars</span> show the average accuracy per cycle, "
        "and the <span style='color:#16a34a;font-weight:600'>green diamonds</span> show the best result in that cycle. "
        "Ideally both trend upward over time.",
        chart_cycle_progress,
    ),
    (
        "How Each Hyperparameter Affects Accuracy",
        "Each subplot isolates one hyperparameter. "
        "Taller bars = higher average accuracy for that value. "
        "Error bars show the spread (standard deviation). "
        "Use this to quickly see which settings tend to work best.",
        chart_hyperparameter_impact,
    ),
    (
        "Filters vs. Layers Heatmap",
        "This grid shows the average accuracy for each combination of filter count and layer depth. "
        "Darker green = higher accuracy. Numbers inside cells are the actual values. "
        "Look for the darkest cell to find the best structural combination.",
        chart_heatmap,
    ),
    (
        "Optimizer & Learning Rate Comparison",
        "Compares every optimizer + learning rate pairing tested. "
        "The number of runs is shown next to each bar so you can gauge confidence.",
        chart_optimizer_lr,
    ),
    (
        "All Configurations Ranked",
        "Every unique configuration ranked from worst (top) to best (bottom). "
        "The <span style='color:#16a34a;font-weight:600'>green bar</span> highlights the #1 config.",
        chart_top_configs,
    ),
]


def build_dashboard(df):
    sections_html = []
    plotly_js_included = False

    for title, description, chart_fn in CHART_SECTIONS:
        fig = chart_fn(df)
        chart_html = fig.to_html(
            full_html=False,
            include_plotlyjs=(not plotly_js_included),
        )
        plotly_js_included = True

        sections_html.append(
            f"<div class='section'>"
            f"  <h2>{title}</h2>"
            f"  <p class='desc'>{description}</p>"
            f"  <div class='chart'>{chart_html}</div>"
            f"</div>"
        )

    page = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AutoLab Dashboard</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: {COLORS['bg']};
    color: {COLORS['text']};
    font-family: Inter, system-ui, -apple-system, sans-serif;
    line-height: 1.6;
    padding: 24px;
    max-width: 1100px;
    margin: 0 auto;
  }}
  h1 {{
    text-align: center;
    font-size: 1.8rem;
    margin-bottom: 8px;
  }}
  .subtitle {{
    text-align: center;
    color: {COLORS['muted']};
    margin-bottom: 28px;
    font-size: 0.95rem;
  }}

  /* Stats banner */
  .summary {{ margin-bottom: 36px; }}
  .stats-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 14px;
    margin-bottom: 16px;
  }}
  .stat-card {{
    background: {COLORS['card']};
    border-radius: 12px;
    padding: 18px 14px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }}
  .stat-value {{ font-size: 1.6rem; font-weight: 700; }}
  .stat-label {{ font-size: 0.8rem; color: {COLORS['muted']}; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .best-banner {{
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 1px solid #a7f3d0;
    border-radius: 10px;
    padding: 12px 18px;
    font-size: 0.95rem;
    text-align: center;
  }}
  .best-icon {{ color: #f59e0b; font-size: 1.1rem; }}

  /* Chart sections */
  .section {{
    background: {COLORS['card']};
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 28px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  .section h2 {{
    font-size: 1.15rem;
    margin-bottom: 6px;
  }}
  .section .desc {{
    color: {COLORS['muted']};
    font-size: 0.88rem;
    margin-bottom: 16px;
    max-width: 800px;
  }}
  .chart {{ width: 100%; }}
</style>
</head>
<body>
  <h1>AutoLab &mdash; Experiment Dashboard</h1>
  <p class="subtitle">LLM-driven hyperparameter search on MNIST &middot; CNN experiments</p>

  {build_summary_html(df)}

  {"".join(sections_html)}

</body></html>"""

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public", "index.html")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Dashboard saved to {out_path}")
    return out_path


if __name__ == "__main__":
    df = load_data()

    print(f"Loaded {len(df)} experiments across {df['cycle'].nunique()} cycles.\n")

    best = df.loc[df["accuracy"].idxmax()]
    print(f"Best accuracy: {best['accuracy']:.2f}%")
    print(f"  Config: filters={best['filters']}, num_layers={best['num_layers']}, "
          f"lr={best['learning_rate']}, optimizer={best['optimizer']}")
    print(f"  Cycle: {best['cycle']}\n")

    path = build_dashboard(df)

    import webbrowser, os
    webbrowser.open("file://" + os.path.realpath(path))
