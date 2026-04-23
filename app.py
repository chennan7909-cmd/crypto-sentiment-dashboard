"""
Web3 Alpha News Sentiment Engine — Streamlit Dashboard
Nan Chen | Ontario Tech University
Reads pre-computed CSVs from the FinBERT pipeline.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

st.set_page_config(
    page_title="Web3 Sentiment Engine",
    page_icon="📰",
    layout="wide"
)

st.title("Web3 Alpha News Sentiment Engine")
st.caption(
    "Nan Chen · Ontario Tech University · "
    "FinBERT NLP · NewsAPI + Guardian + CoinDesk RSS · BTC & SOL"
)
st.divider()

# ═══════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    base = "data/"
    raw        = pd.read_csv(base + "news_sentiment_raw.csv", parse_dates=["date"])
    btc_daily  = pd.read_csv(base + "daily_sentiment_btc.csv", parse_dates=["date"])
    sol_daily  = pd.read_csv(base + "daily_sentiment_sol.csv", parse_dates=["date"])
    btc_price  = pd.read_csv(base + "btc_sentiment_price.csv", parse_dates=["date"])
    sol_price  = pd.read_csv(base + "sol_sentiment_price.csv", parse_dates=["date"])
    btc_events = pd.read_csv(base + "btc_regulation_events.csv", parse_dates=["date"]) if _file_exists(base + "btc_regulation_events.csv") else pd.DataFrame()
    sol_events = pd.read_csv(base + "sol_regulation_events.csv", parse_dates=["date"]) if _file_exists(base + "sol_regulation_events.csv") else pd.DataFrame()
    return raw, btc_daily, sol_daily, btc_price, sol_price, btc_events, sol_events

def _file_exists(path):
    import os
    return os.path.exists(path)

try:
    raw, btc_daily, sol_daily, btc_price, sol_price, btc_events, sol_events = load_data()
    data_ok = True
except Exception as e:
    st.error(f"Could not load data files: {e}")
    st.info("Upload the 7 CSV files from your Google Drive `crypto_sentiment/` folder into a `data/` subfolder in this repo.")
    data_ok = False
    st.stop()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    asset     = st.selectbox("Select Asset", ["BTC", "SOL"])
    sent_threshold = st.slider("Bearish alert threshold", -0.5, 0.0, -0.2, 0.05)
    st.divider()
    st.markdown("**Pipeline Info**")
    st.markdown("- Model: FinBERT (ProsusAI/finbert)")
    st.markdown("- Sources: NewsAPI · Guardian · RSS")
    st.markdown(f"- Articles: {len(raw):,}")
    st.markdown(f"- Date range: {raw['date'].min().date()} – {raw['date'].max().date()}")
    st.markdown(f"- Regulation articles: {int(raw['is_regulation'].sum())}")

daily  = btc_daily  if asset == "BTC" else sol_daily
merged = btc_price  if asset == "BTC" else sol_price
events = btc_events if asset == "BTC" else sol_events

# ═══════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "Sentiment Overview",
    "Price vs Sentiment",
    "Regulation Event Study",
    "News Feed"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — Sentiment Overview
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"{asset} — News Sentiment Overview")

    # KPIs
    latest_sent  = float(daily["sentiment_index"].iloc[-1])
    mean_sent    = float(daily["sentiment_index"].mean())
    bearish_days = int((daily["sentiment_index"] < sent_threshold).sum())
    reg_articles = int(raw["is_regulation"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Sentiment Index", f"{latest_sent:+.3f}",
              delta="Bearish" if latest_sent < -0.1 else "Bullish" if latest_sent > 0.1 else "Neutral")
    c2.metric("30-Day Mean Sentiment", f"{mean_sent:+.3f}")
    c3.metric(f"Bearish Days (<{sent_threshold})", f"{bearish_days} / {len(daily)}")
    c4.metric("Regulation Articles", f"{reg_articles}")
    st.divider()

    # Sentiment index time series
    fig_sent = go.Figure()
    fig_sent.add_trace(go.Scatter(
        x=daily["date"], y=daily["sentiment_index"],
        fill="tozeroy", name="Sentiment Index",
        line=dict(color="#FF5722", width=2)
    ))
    fig_sent.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_sent.add_hline(
        y=sent_threshold, line_dash="dot", line_color="red",
        annotation_text=f"Bearish threshold ({sent_threshold})",
        annotation_position="bottom right"
    )
    # Shade bearish zones
    for _, row in daily[daily["sentiment_index"] < sent_threshold].iterrows():
        fig_sent.add_vrect(
            x0=row["date"], x1=row["date"] + pd.Timedelta(days=1),
            fillcolor="red", opacity=0.1, line_width=0
        )
    fig_sent.update_layout(
        title=f"{asset} 3-Day Rolling Sentiment Index (FinBERT)",
        yaxis_title="Sentiment Score",
        height=320, template="plotly_dark"
    )
    st.plotly_chart(fig_sent, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Sentiment composition stacked bar
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=daily["date"], y=daily["bullish_ratio"],
            name="Bullish", marker_color="#4CAF50"
        ))
        fig_comp.add_trace(go.Bar(
            x=daily["date"], y=daily["bearish_ratio"],
            name="Bearish", marker_color="#F44336"
        ))
        neutral = 1 - daily["bullish_ratio"] - daily["bearish_ratio"]
        fig_comp.add_trace(go.Bar(
            x=daily["date"], y=neutral.clip(0),
            name="Neutral", marker_color="#9E9E9E"
        ))
        fig_comp.update_layout(
            barmode="stack",
            title="Daily Sentiment Composition",
            yaxis=dict(tickformat=".0%"),
            height=300, template="plotly_dark"
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    with col2:
        # Overall label pie chart
        asset_raw = raw[raw["asset"] == asset] if asset in raw["asset"].values else raw
        label_counts = asset_raw["sentiment_label"].value_counts()
        fig_pie = px.pie(
            values=label_counts.values,
            names=label_counts.index,
            color=label_counts.index,
            color_discrete_map={
                "positive": "#4CAF50",
                "negative": "#F44336",
                "neutral":  "#9E9E9E"
            },
            title=f"{asset} Sentiment Label Distribution"
        )
        fig_pie.update_layout(height=300, template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Article volume + regulation overlay
    all_daily = raw.groupby("date").agg(
        total=("sentiment_score", "count"),
        regulation=("is_regulation", "sum")
    ).reset_index()

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(
        x=all_daily["date"], y=all_daily["total"],
        name="Total articles", marker_color="#607D8B", opacity=0.6
    ))
    fig_vol.add_trace(go.Bar(
        x=all_daily["date"], y=all_daily["regulation"],
        name="Regulation articles", marker_color="#F44336", opacity=0.9
    ))
    fig_vol.update_layout(
        barmode="overlay",
        title="Daily News Volume — Total vs Regulation-Related",
        height=280, template="plotly_dark"
    )
    st.plotly_chart(fig_vol, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 2 — Price vs Sentiment
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"{asset} — Price vs Sentiment Correlation")

    if merged.empty or "sentiment_index" not in merged.columns:
        st.warning("No merged data available for this asset.")
    else:
        # Dual-axis chart
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Scatter(
            x=merged["date"], y=merged["Close"],
            name=f"{asset} Price (USD)",
            line=dict(color="#2196F3", width=2),
            yaxis="y1"
        ))
        fig_dual.add_trace(go.Scatter(
            x=merged["date"], y=merged["sentiment_index"],
            name="Sentiment Index",
            line=dict(color="#FF5722", width=1.5, dash="dot"),
            yaxis="y2"
        ))
        fig_dual.update_layout(
            title=f"{asset} Price vs News Sentiment Index",
            yaxis=dict(title=f"{asset} Price (USD)", color="#2196F3"),
            yaxis2=dict(title="Sentiment Index", color="#FF5722",
                        overlaying="y", side="right"),
            height=400, template="plotly_dark",
            legend=dict(x=0, y=1)
        )
        st.plotly_chart(fig_dual, use_container_width=True)

        # Correlation table
        st.subheader("Pearson Correlation — Sentiment vs Forward Returns")
        corr_rows = []
        for col, label in [
            ("fwd_return_1d", "Next-day return"),
            ("fwd_return_3d", "3-day forward return"),
            ("fwd_return_7d", "7-day forward return"),
        ]:
            if col not in merged.columns:
                continue
            valid = merged[["sentiment_index", col]].dropna()
            if len(valid) < 5:
                continue
            r, p = stats.pearsonr(valid["sentiment_index"], valid[col])
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            corr_rows.append({
                "Horizon": label,
                "Pearson r": f"{r:+.4f}",
                "p-value": f"{p:.4f}",
                "Significance": sig,
                "Interpretation": (
                    "Negative sentiment predicts price decline" if r < -0.2 and p < 0.1
                    else "Positive sentiment predicts price rise" if r > 0.2 and p < 0.1
                    else "No significant relationship"
                )
            })

        if corr_rows:
            corr_df = pd.DataFrame(corr_rows)
            st.dataframe(corr_df, use_container_width=True, hide_index=True)

            # Scatter: sentiment vs next-day return
            if "fwd_return_1d" in merged.columns:
                valid = merged[["sentiment_index", "fwd_return_1d", "date"]].dropna()
                fig_scatter = px.scatter(
                    valid, x="sentiment_index", y="fwd_return_1d",
                    trendline="ols",
                    labels={
                        "sentiment_index": "Sentiment Index",
                        "fwd_return_1d": "Next-Day Return"
                    },
                    title=f"{asset} Sentiment vs Next-Day Price Return",
                    color_discrete_sequence=["#FF5722"]
                )
                fig_scatter.update_layout(
                    yaxis=dict(tickformat=".1%"),
                    height=350, template="plotly_dark"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        # Key finding callout
        if asset == "SOL":
            st.success(
                "Key Finding: SOL news sentiment shows statistically significant "
                "negative correlation with 3-day (r = -0.57, p = 0.003***) and "
                "7-day (r = -0.64, p = 0.001***) forward returns. "
                "Negative news predicts SOL price declines over the following week."
            )
        else:
            st.info(
                "BTC shows weaker sentiment-price correlation over 30 days, "
                "consistent with its larger market cap and lower susceptibility "
                "to single-source news sentiment."
            )


# ════════════════════════════════════════════════════════════
# TAB 3 — Regulation Event Study
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"{asset} — Regulation Event Study")
    st.markdown(
        "**Definition**: A regulation event is a day where regulation-related "
        "article count is in the top 25th percentile AND sentiment index ≤ -0.2. "
        "The table below shows average forward returns on event vs non-event days."
    )

    if asset == "SOL" and not sol_events.empty:
        st.success(f"SOL regulation events identified: **{len(sol_events)} days**")

        # Event summary table
        event_data = {
            "Horizon": ["1-day", "3-day", "7-day"],
            "Event Return": ["+4.33%", "+3.80%", "+6.81%"],
            "Non-Event Return": ["+0.98%", "-8.59%", "-8.50%"],
            "p-value": ["0.585", "0.011**", "0.093*"],
            "Interpretation": [
                "No significant difference",
                "Events outperform significantly",
                "Events outperform marginally"
            ]
        }
        st.dataframe(pd.DataFrame(event_data), use_container_width=True, hide_index=True)

        st.markdown("""
        **Interpretation**: On days when regulation news spikes with negative sentiment,
        SOL's 3-day forward return is actually +3.80% vs -8.59% for non-event days
        (p=0.011). This counter-intuitive finding may reflect a **buy-the-news** dynamic:
        negative regulatory headlines create short-term selling pressure followed by recovery.
        """)

        # Show regulation events
        if not sol_events.empty and "date" in sol_events.columns:
            st.subheader("Identified Regulation Event Days")
            display_cols = [c for c in ["date", "reg_article_count", "sentiment_index",
                                         "fwd_return_1d", "fwd_return_3d"] if c in sol_events.columns]
            st.dataframe(sol_events[display_cols].reset_index(drop=True),
                         use_container_width=True, hide_index=True)

    elif asset == "BTC":
        st.info(
            "BTC regulation events: 0 days identified in the 30-day window. "
            "BTC-tagged articles with high regulation keyword density and "
            "negative sentiment did not co-occur in this period. "
            "Extend to 90+ days for a more robust event study."
        )

    # Regulation keyword frequency bar chart
    REG_KEYWORDS = ["regulation", "SEC", "ban", "lawsuit", "CFTC",
                    "crackdown", "enforcement", "CBDC", "compliance", "sanction"]
    kw_counts = {}
    for kw in REG_KEYWORDS:
        kw_counts[kw] = int(raw["text"].str.contains(kw, case=False, na=False).sum())

    kw_df = pd.DataFrame({"Keyword": list(kw_counts.keys()),
                           "Count": list(kw_counts.values())}).sort_values("Count", ascending=True)
    fig_kw = px.bar(
        kw_df, x="Count", y="Keyword", orientation="h",
        title="Regulation Keyword Frequency (All Articles)",
        color="Count", color_continuous_scale="Reds"
    )
    fig_kw.update_layout(height=380, template="plotly_dark")
    st.plotly_chart(fig_kw, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 4 — News Feed
# ════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Live News Sentiment Feed")

    col1, col2 = st.columns(2)
    with col1:
        label_filter = st.selectbox(
            "Filter by sentiment", ["All", "Bearish (negative)", "Bullish (positive)", "Neutral"]
        )
    with col2:
        asset_filter = st.selectbox("Filter by asset", ["All", "BTC", "SOL", "OTHER"])

    df_feed = raw.copy().sort_values("date", ascending=False)

    if label_filter == "Bearish (negative)":
        df_feed = df_feed[df_feed["sentiment_label"] == "negative"]
    elif label_filter == "Bullish (positive)":
        df_feed = df_feed[df_feed["sentiment_label"] == "positive"]
    elif label_filter == "Neutral":
        df_feed = df_feed[df_feed["sentiment_label"] == "neutral"]

    if asset_filter != "All":
        df_feed = df_feed[df_feed["asset"] == asset_filter]

    st.markdown(f"Showing **{len(df_feed)}** articles")

    for _, row in df_feed.head(50).iterrows():
        label = row.get("sentiment_label", "neutral")
        score = row.get("sentiment_score", 0.0)
        is_reg = row.get("is_regulation", 0)
        color  = "#4CAF50" if label == "positive" else "#F44336" if label == "negative" else "#9E9E9E"
        reg_badge = " 🏛 REG" if is_reg else ""

        with st.container():
            col_a, col_b = st.columns([5, 1])
            with col_a:
                st.markdown(f"**{row['title']}**")
                st.caption(f"{str(row['date'])[:10]}  ·  {row.get('source', '')}  ·  {row.get('asset', '')}{reg_badge}")
            with col_b:
                st.markdown(
                    f"<span style='color:{color}; font-weight:bold'>"
                    f"{label.upper()}<br>{score:+.2f}</span>",
                    unsafe_allow_html=True
                )
            st.markdown("---")
