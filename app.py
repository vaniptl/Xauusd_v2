"""
XAUUSD Trading Bot — Streamlit Cloud Edition
Bloomberg-style dark terminal UI.
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone, timedelta
from typing import Optional
import os, sys, time

sys.path.insert(0, os.path.dirname(__file__))

# ── page config (MUST be first Streamlit call) ─────────────────────
st.set_page_config(
    page_title="XAUUSD Trading Bot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core.data_engine  import DataEngine
from core.indicators   import Indicators
from core.sr_engine    import combine_sr
from core.smc_engine   import SMCEngine
from core.regime       import detect_regime, detect_session, REGIME_LABEL, SESSION_LABEL
from core.strategies   import evaluate_all
from core.risk         import RiskManager, RiskState
from core.macro        import get_dxy_bias, dxy_alignment
from core.optimizer    import should_run, run_optimizer, DEFAULT_WEIGHTS
from database.db       import (init_db, save_signal, save_trade, update_trade,
                                get_open_trades, get_trades, get_recent_signals,
                                get_metrics_cached, get_daily_history)

# ── init ───────────────────────────────────────────────────────────
init_db()

# ── session state bootstrap ────────────────────────────────────────
def _init_state():
    defaults = {
        "bot_running":    False,
        "dry_run":        True,
        "equity":         10000.0,
        "pip_target":     200.0,
        "risk_pct":       1.5,
        "strategy_weights": dict(DEFAULT_WEIGHTS),
        "optimizer_last": None,
        "open_trades":    [],
        "last_signals":   [],
        "last_price":     0.0,
        "cycle_count":    0,
        "errors":         [],
        "risk_state":     None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state.risk_state is None:
        st.session_state.risk_state = RiskState(
            equity=st.session_state.equity,
            initial_equity=st.session_state.equity,
            pip_target=st.session_state.pip_target,
            max_risk_pct=st.session_state.risk_pct,
        )

_init_state()

DATA   = DataEngine()
SMC_ENG = SMCEngine()

# ── CSS — Bloomberg dark terminal ──────────────────────────────────
st.markdown("""
<style>
/* Root colours */
:root {
  --gold:   #B8860B;
  --gold2:  #D4A017;
  --green:  #00C851;
  --red:    #FF4444;
  --blue:   #4FC3F7;
  --bg:     #0A0A0F;
  --bg2:    #13131C;
  --bg3:    #1A1A28;
  --border: #2A2A3D;
  --text:   #E0E0E0;
  --text2:  #9090A0;
}
.stApp { background: var(--bg); }
.stApp > header { background: transparent !important; }

/* Hide default Streamlit elements */
#MainMenu, footer, .stDeployButton { display:none !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--bg2) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .stMarkdown { color: var(--text); }

/* Metric cards */
[data-testid="stMetric"] {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 1rem !important;
}
[data-testid="stMetricLabel"] { color: var(--text2) !important; font-size: 11px !important; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-size: 22px !important; font-weight: 700 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: var(--bg2); border-bottom: 1px solid var(--border); gap: 0; }
.stTabs [data-baseweb="tab"] { color: var(--text2) !important; padding: .6rem 1.4rem; font-size: 13px; border-radius: 0 !important; }
.stTabs [aria-selected="true"] { color: var(--gold) !important; border-bottom: 2px solid var(--gold) !important; background: transparent !important; }

/* Buttons */
.stButton > button {
  background: var(--bg3) !important; border: 1px solid var(--border) !important;
  color: var(--text) !important; border-radius: 6px !important; font-size: 13px !important;
}
.stButton > button:hover { border-color: var(--gold) !important; color: var(--gold) !important; }

/* Inputs */
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb] {
  background: var(--bg3) !important; border-color: var(--border) !important; color: var(--text) !important;
}

/* DataFrames */
[data-testid="stDataFrame"] { background: var(--bg2) !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Signal cards */
.signal-card {
  background: var(--bg2); border: 1px solid var(--border); border-radius: 10px;
  padding: 1rem; margin-bottom: .75rem; position: relative; overflow: hidden;
}
.signal-buy  { border-left: 4px solid var(--green) !important; }
.signal-sell { border-left: 4px solid var(--red)   !important; }
.badge { display:inline-block; padding:.2rem .6rem; border-radius:4px; font-size:11px; font-weight:700; }
.badge-buy  { background:#0a2e1a; color:var(--green); }
.badge-sell { background:#2e0a0a; color:var(--red);   }
.badge-gold { background:#1a1400; color:var(--gold);  }

/* Price ticker */
.price-ticker { font-size:28px; font-weight:700; color:var(--gold); letter-spacing:2px; }
.price-change-pos { color:var(--green); font-size:14px; }
.price-change-neg { color:var(--red);   font-size:14px; }

/* Progress bar */
.pip-progress { background:var(--bg3); border-radius:4px; height:8px; margin:.3rem 0; overflow:hidden; }
.pip-fill { background: linear-gradient(90deg,var(--gold),var(--gold2)); border-radius:4px; height:100%; }

/* Status indicator */
.status-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }
.dot-green { background:var(--green); box-shadow:0 0 6px var(--green); }
.dot-red   { background:var(--red);   box-shadow:0 0 6px var(--red);   }
.dot-amber { background:#f59e0b;      box-shadow:0 0 6px #f59e0b;      }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0 .5rem">
          <div style="font-size:28px">⚡</div>
          <div style="color:#B8860B;font-size:16px;font-weight:700;letter-spacing:2px">XAUUSD BOT</div>
          <div style="color:#606070;font-size:11px">24×7 Autonomous Trading</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        # Bot toggle
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ START" if not st.session_state.bot_running else "⏸ STOP",
                         use_container_width=True,
                         type="primary" if not st.session_state.bot_running else "secondary"):
                st.session_state.bot_running = not st.session_state.bot_running
                if st.session_state.bot_running:
                    st.toast("Bot started!", icon="✅")
                else:
                    st.toast("Bot stopped", icon="⏸")
        with col2:
            mode = "PAPER" if st.session_state.dry_run else "LIVE"
            color = "#60a5fa" if st.session_state.dry_run else "#ef4444"
            st.markdown(f"<div style='text-align:center;padding:.4rem;background:#13131C;border:1px solid #2A2A3D;border-radius:6px;font-size:12px;font-weight:700;color:{color}'>{mode}</div>", unsafe_allow_html=True)

        st.markdown("**Bot Status**")
        rs = st.session_state.risk_state
        if rs:
            rm = RiskManager(rs)
            can, reason = rm.can_trade()
            dot = "dot-green" if (st.session_state.bot_running and can) else ("dot-amber" if not can else "dot-red")
            label = "Running" if st.session_state.bot_running and can else reason[:28]
            st.markdown(f'<div><span class="status-dot {dot}"></span><span style="font-size:12px;color:#9090A0">{label}</span></div>', unsafe_allow_html=True)

        st.divider()
        st.markdown("**⚙️ Configuration**")

        new_dry = st.toggle("Paper Trading Mode", value=st.session_state.dry_run)
        if new_dry != st.session_state.dry_run:
            st.session_state.dry_run = new_dry

        eq = st.number_input("Account Equity ($)", value=float(st.session_state.equity),
                              min_value=100.0, step=100.0, format="%.2f")
        if eq != st.session_state.equity:
            st.session_state.equity = eq
            if rs: rs.equity = eq; rs.initial_equity = eq

        target = st.number_input("Daily Pip Target", value=float(st.session_state.pip_target),
                                  min_value=20.0, step=10.0)
        if target != st.session_state.pip_target:
            st.session_state.pip_target = target
            if rs: rs.pip_target = target

        risk = st.slider("Risk per Trade (%)", 0.5, 3.0, float(st.session_state.risk_pct), 0.1)
        if risk != st.session_state.risk_pct:
            st.session_state.risk_pct = risk
            if rs: rs.max_risk_pct = risk

        st.divider()
        st.markdown("**📊 Strategy Weights** (optimizer)")
        for s_name, s_w in st.session_state.strategy_weights.items():
            label = s_name.replace("_"," ").title()[:16]
            st.markdown(f'<div style="display:flex;justify-content:space-between;font-size:11px;color:#9090A0;margin:.2rem 0"><span>{label}</span><span style="color:#B8860B;font-weight:700">{s_w:.2f}×</span></div>', unsafe_allow_html=True)

        if st.button("▶ Run Optimizer Now", use_container_width=True):
            trades = get_trades(limit=500)
            new_w = run_optimizer(trades, st.session_state.strategy_weights)
            st.session_state.strategy_weights = new_w
            st.session_state.optimizer_last   = datetime.now(timezone.utc)
            st.toast("Optimizer complete!", icon="⚙️")

        if rs and rs.paused:
            st.divider()
            st.warning(f"⏸ {rs.pause_reason[:40]}")
            if st.button("▶ Resume Trading", use_container_width=True):
                RiskManager(rs).resume()
                st.toast("Trading resumed!", icon="✅")

        st.divider()
        st.markdown(f'<div style="font-size:10px;color:#404055;text-align:center">v1.0 · Cycle #{st.session_state.cycle_count}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  HEADER — price ticker + regime
# ══════════════════════════════════════════════════════════════════════
def render_header():
    price = DATA.current_price()
    if price > 0:
        st.session_state.last_price = price

    regime  = Regime_curr = None
    session = None

    # Fetch H1 for regime
    df_h1 = DATA.get("1h", 250)
    if not df_h1.empty:
        df_h1e = Indicators.enrich(df_h1)
        if not df_h1e.empty:
            from core.regime import detect_regime as _dr, detect_session as _ds
            Regime_curr = _dr(df_h1e)
            session     = _ds()

    r_label, r_color = (REGIME_LABEL.get(Regime_curr, ("Unknown","#6b7280")) if Regime_curr else ("Unknown","#6b7280"))
    s_label, s_color = (SESSION_LABEL.get(session, ("—","#6b7280")) if session else ("—","#6b7280"))

    col_p, col_r, col_s, col_dxy, col_mode = st.columns([3, 2, 2, 2, 2])

    with col_p:
        p_str = f"${price:,.2f}" if price > 0 else "Loading..."
        st.markdown(f'<div class="price-ticker">{p_str}</div><div style="color:#606070;font-size:11px">XAU/USD · Gold Spot</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown(f'<div style="font-size:11px;color:#606070;text-transform:uppercase;margin-bottom:.3rem">Regime</div><span style="background:{r_color}22;color:{r_color};border:1px solid {r_color}44;padding:.3rem .7rem;border-radius:20px;font-size:12px;font-weight:700">{r_label}</span>', unsafe_allow_html=True)

    with col_s:
        st.markdown(f'<div style="font-size:11px;color:#606070;text-transform:uppercase;margin-bottom:.3rem">Session</div><span style="background:{s_color}22;color:{s_color};border:1px solid {s_color}44;padding:.3rem .7rem;border-radius:20px;font-size:12px;font-weight:700">{s_label}</span>', unsafe_allow_html=True)

    with col_dxy:
        dxy_df  = DATA.get_dxy(30)
        dxy_b   = get_dxy_bias(dxy_df)
        dxy_col = "#ef4444" if dxy_b == "bullish" else "#22c55e" if dxy_b == "bearish" else "#6b7280"
        dxy_lbl = f"{'↑' if dxy_b=='bullish' else '↓' if dxy_b=='bearish' else '—'} DXY {dxy_b.title()}"
        st.markdown(f'<div style="font-size:11px;color:#606070;text-transform:uppercase;margin-bottom:.3rem">DXY Macro</div><span style="color:{dxy_col};font-size:13px;font-weight:700">{dxy_lbl}</span>', unsafe_allow_html=True)

    with col_mode:
        rs = st.session_state.risk_state
        if rs:
            pct = min(100, (rs.daily_pips / max(rs.pip_target, 1)) * 100)
            st.markdown(f'<div style="font-size:11px;color:#606070;text-transform:uppercase;margin-bottom:.3rem">Daily Target</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="pip-progress"><div class="pip-fill" style="width:{pct:.0f}%"></div></div>', unsafe_allow_html=True)
            col_text = "#22c55e" if rs.daily_pips >= 0 else "#ef4444"
            st.markdown(f'<span style="color:{col_text};font-size:12px;font-weight:700">{rs.daily_pips:+.1f} / {rs.pip_target:.0f} pips</span>', unsafe_allow_html=True)

    st.divider()


# ══════════════════════════════════════════════════════════════════════
#  TAB 1 — Dashboard
# ══════════════════════════════════════════════════════════════════════
def tab_dashboard():
    rs = st.session_state.risk_state
    if rs is None:
        st.info("Configure settings in sidebar.")
        return

    # KPI row
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    eq_chg = ((rs.equity - rs.initial_equity) / max(rs.initial_equity, 1)) * 100
    c1.metric("Equity", f"${rs.equity:,.2f}",    f"{eq_chg:+.2f}%")
    c2.metric("Daily PnL", f"${rs.daily_pnl_usd:+.2f}", f"{rs.daily_pips:+.1f} pips")
    c3.metric("Open Trades", str(rs.open_trades), f"Max {rs.max_concurrent}")

    metrics = get_metrics_cached()
    c4.metric("Win Rate",      f"{metrics.get('win_rate',0):.1f}%",  f"{metrics.get('total',0)} trades")
    c5.metric("Profit Factor", f"{metrics.get('profit_factor',0):.2f}", "Target ≥1.3")
    c6.metric("Total Pips",    f"{(metrics.get('total_pips') or 0):+.1f}", f"Avg {(metrics.get('avg_pips') or 0):.1f}/trade")

    st.markdown("###")

    # Charts row
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("**📈 Daily Pip Performance**")
        daily = get_daily_history(30)
        if daily:
            df_d = pd.DataFrame(daily)
            df_d["pips"] = pd.to_numeric(df_d["pips"], errors="coerce").fillna(0)
            df_d["color"] = df_d["pips"].apply(lambda x: "#22c55e" if x >= 0 else "#ef4444")
            fig = go.Figure(go.Bar(
                x=df_d["dt"], y=df_d["pips"],
                marker_color=df_d["color"],
                hovertemplate="%{x}<br>%{y:+.1f} pips<extra></extra>"
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=220, margin=dict(l=0,r=0,t=10,b=0),
                font=dict(color="#9090A0", size=11),
                xaxis=dict(gridcolor="#1A1A28", color="#9090A0"),
                yaxis=dict(gridcolor="#1A1A28", color="#9090A0", zeroline=True, zerolinecolor="#2A2A3D"),
                showlegend=False,
            )
            st.plotly_chart(fig)
        else:
            st.info("No closed trades yet. Run the bot to generate data.")

    with col_right:
        st.markdown("**🎯 Strategy Performance**")
        by_s = metrics.get("by_strategy", [])
        if by_s:
            df_s = pd.DataFrame(by_s)
            df_s["win_rate"] = (df_s["w"] / df_s["t"].clip(1) * 100).round(1)
            df_s["label"] = df_s["strategy"].str.replace("_"," ").str.title().str[:14]
            fig2 = go.Figure(go.Bar(
                x=df_s["label"], y=df_s["pp"],
                marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in df_s["pp"]],
                hovertemplate="%{x}<br>%{y:+.1f} pips<extra></extra>",
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=220, margin=dict(l=0,r=0,t=10,b=0),
                font=dict(color="#9090A0", size=11),
                xaxis=dict(gridcolor="#1A1A28", color="#9090A0", tickangle=-20),
                yaxis=dict(gridcolor="#1A1A28", color="#9090A0"),
                showlegend=False,
            )
            st.plotly_chart(fig2)
        else:
            st.info("No trade data yet.")

    # Price chart
    st.markdown("**🕯️ XAU/USD — H1 Chart**")
    df_h1 = DATA.get("1h", 100)
    if not df_h1.empty:
        df_h1e = Indicators.enrich(df_h1)
        _render_candlestick(df_h1e.tail(80), show_emas=True)


def _render_candlestick(df: pd.DataFrame, show_emas=True):
    if df.empty:
        return
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        name="Price",
    ))
    if show_emas and "ema20" in df.columns:
        for col, color, name in [("ema20","#60a5fa","EMA 20"),("ema50","#f59e0b","EMA 50"),("ema200","#a78bfa","EMA 200")]:
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], line=dict(color=color, width=1), name=name, opacity=0.8))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=380, margin=dict(l=0,r=0,t=10,b=0),
        font=dict(color="#9090A0", size=11),
        xaxis=dict(gridcolor="#1A1A28", color="#9090A0", rangeslider_visible=False),
        yaxis=dict(gridcolor="#1A1A28", color="#9090A0", side="right"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9090A0", size=10)),
        showlegend=True,
    )
    st.plotly_chart(fig)


# ══════════════════════════════════════════════════════════════════════
#  TAB 2 — Signals
# ══════════════════════════════════════════════════════════════════════
def tab_signals():
    st.markdown("""
    <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:1rem">
      <div style="width:8px;height:8px;border-radius:50%;background:#22c55e;box-shadow:0 0 8px #22c55e"></div>
      <span style="color:#9090A0;font-size:12px">Auto-refreshes every 60s when bot is running</span>
    </div>
    """, unsafe_allow_html=True)

    # Show live signals from session state
    live_signals = st.session_state.last_signals
    if live_signals:
        st.markdown(f"**🔴 Live Signals ({len(live_signals)})**")
        _render_signal_cards(live_signals, live=True)
        st.divider()

    # Recent signals from DB
    st.markdown("**📡 Signal History**")
    recent = get_recent_signals(30)
    if not recent:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#404055">
          <div style="font-size:40px;margin-bottom:.5rem">📡</div>
          <div>No signals yet. Start the bot to scan markets.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    _render_signal_cards(recent, live=False)


def _render_signal_cards(signals, live=False):
    cols = st.columns(2)
    for i, sig in enumerate(signals):
        with cols[i % 2]:
            direction  = sig.get("direction","")
            strategy   = (sig.get("strategy","") or "").replace("_"," ").upper()
            conf       = float(sig.get("confidence") or 0)
            entry      = float(sig.get("entry_price") or sig.get("entry") or 0)
            sl         = float(sig.get("sl") or 0)
            tp1        = float(sig.get("tp1") or sig.get("take_profit_1") or 0)
            tp2        = float(sig.get("tp2") or sig.get("take_profit_2") or 0)
            rr2        = float(sig.get("rr2") or sig.get("risk_reward_2") or 0)
            session_s  = sig.get("session","")
            regime_s   = (sig.get("regime","") or "").replace("_"," ")
            notes      = sig.get("notes","")
            ts         = sig.get("timestamp","")
            acted      = sig.get("acted_on", 0)

            is_buy     = direction == "BUY"
            dir_color  = "#22c55e" if is_buy else "#ef4444"
            dir_emoji  = "▲" if is_buy else "▼"
            conf_color = "#22c55e" if conf >= 80 else "#B8860B" if conf >= 65 else "#ef4444"
            border_col = "#22c55e44" if is_buy else "#ef4444aa"

            st.markdown(f"""
            <div class="signal-card signal-{'buy' if is_buy else 'sell'}" style="border:1px solid {border_col}">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.6rem">
                <div>
                  <div style="font-size:12px;font-weight:700;color:#E0E0E0;letter-spacing:.04em">{strategy}</div>
                  <span class="badge badge-{'buy' if is_buy else 'sell'}" style="margin-top:3px">{dir_emoji} {direction}</span>
                  {'<span class="badge badge-gold" style="margin-left:4px">✓ TRADED</span>' if acted else ''}
                </div>
                <div style="text-align:right">
                  <div style="font-size:22px;font-weight:700;color:{conf_color}">{conf:.0f}%</div>
                  <div style="font-size:10px;color:#606070">confidence</div>
                </div>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem .8rem;margin:.5rem 0">
                <div><div style="font-size:10px;color:#606070;text-transform:uppercase">Entry</div>
                     <div style="font-size:13px;font-weight:600;font-family:monospace">${entry:,.2f}</div></div>
                <div><div style="font-size:10px;color:#606070;text-transform:uppercase">Stop Loss</div>
                     <div style="font-size:13px;font-weight:600;font-family:monospace;color:#ef4444">${sl:,.2f}</div></div>
                <div><div style="font-size:10px;color:#606070;text-transform:uppercase">TP 1</div>
                     <div style="font-size:13px;font-weight:600;font-family:monospace;color:#22c55e">${tp1:,.2f}</div></div>
                <div><div style="font-size:10px;color:#606070;text-transform:uppercase">TP 2</div>
                     <div style="font-size:13px;font-weight:600;font-family:monospace;color:#22c55e">${tp2:,.2f}</div></div>
              </div>
              <div style="display:flex;justify-content:space-between;align-items:center;
                          border-top:1px solid #1A1A28;padding-top:.5rem;font-size:11px;color:#606070">
                <span>{session_s} · {regime_s}</span>
                <span style="background:#1a1400;color:#B8860B;padding:.15rem .5rem;border-radius:4px;font-weight:700">1:{rr2:.1f} RR</span>
              </div>
              {f'<div style="font-size:10px;color:#505060;margin-top:.3rem;border-top:1px solid #1A1A28;padding-top:.3rem">{notes[:80]}</div>' if notes else ''}
              <div style="font-size:10px;color:#404050;margin-top:.2rem">{ts[:19].replace("T"," ") if ts else ""}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  TAB 3 — Trade History
# ══════════════════════════════════════════════════════════════════════
def tab_trades():
    # Open trades
    open_trades = get_open_trades()
    if open_trades:
        st.markdown(f"**🟢 Open Trades ({len(open_trades)})**")
        df_open = pd.DataFrame(open_trades)
        display_cols = [c for c in ["strategy","direction","entry_price","sl","tp1","tp2","lot_size","confidence","open_time"] if c in df_open.columns]
        st.dataframe(
            df_open[display_cols].rename(columns={
                "entry_price":"Entry","sl":"SL","tp1":"TP1","tp2":"TP2",
                "lot_size":"Lot","confidence":"Conf","open_time":"Opened"
            }),
            use_container_width=True, hide_index=True,
        )
        st.divider()

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        strat_filter = st.selectbox("Strategy", ["All","ema_momentum","trend_continuation",
                                                   "liquidity_sweep","breakout_expansion","smc_concepts"])
    with col_f2:
        date_from = st.date_input("From", value=None)
    with col_f3:
        date_to = st.date_input("To", value=None)

    trades = get_trades(
        limit=300,
        strategy=None if strat_filter == "All" else strat_filter,
        date_from=str(date_from) if date_from else None,
        date_to=str(date_to) if date_to else None,
    )

    if not trades:
        st.info("No closed trades yet.")
        return

    df = pd.DataFrame(trades)

    # Colour coding
    def pip_color(v):
        try:
            v = float(v)
            return "color: #22c55e" if v > 0 else "color: #ef4444" if v < 0 else ""
        except: return ""

    cols = [c for c in ["open_time","strategy","direction","entry_price","exit_price",
                         "pnl_pips","pnl_usd","rr_actual","session","regime","status","duration_min"]
             if c in df.columns]

    df_show = df[cols].copy()
    df_show.columns = [c.replace("_"," ").title() for c in cols]

    st.markdown(f"**📋 Trade History ({len(df)} trades)**")
    st.dataframe(
        df_show.style.map(pip_color, subset=[c for c in ["Pnl Pips","Pnl Usd"] if c in df_show.columns]),
        use_container_width=True, hide_index=True, height=500,
    )

    # Summary stats
    st.divider()
    c1,c2,c3,c4 = st.columns(4)
    wins   = df[df["pnl_usd"] > 0]
    losses = df[df["pnl_usd"] < 0]
    c1.metric("Win Rate",      f"{len(wins)/max(len(df),1)*100:.1f}%")
    c2.metric("Total Pips",    f"{df['pnl_pips'].sum():+.1f}")
    c3.metric("Total PnL",     f"${df['pnl_usd'].sum():+.2f}")
    c4.metric("Avg Duration",  f"{df['duration_min'].mean():.0f} min")


# ══════════════════════════════════════════════════════════════════════
#  TAB 4 — Analytics
# ══════════════════════════════════════════════════════════════════════
def tab_analytics():
    metrics = get_metrics_cached()

    # Heatmap: strategy × session
    st.markdown("**🗺️ Strategy × Session Heatmap (Win Rate %)**")
    trades = get_trades(limit=500)
    if trades:
        df = pd.DataFrame(trades)
        if "strategy" in df.columns and "session" in df.columns:
            df["win"] = (df["pnl_usd"].astype(float) > 0).astype(int)
            pivot = df.pivot_table(values="win", index="strategy", columns="session",
                                   aggfunc=lambda x: round(x.mean() * 100, 1) if len(x) > 0 else 0)
            if not pivot.empty:
                fig = go.Figure(go.Heatmap(
                    z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                    colorscale=[[0,"#2d0000"],[0.5,"#B8860B"],[1,"#004d00"]],
                    text=pivot.values, texttemplate="%{text:.0f}%",
                    hovertemplate="%{y} × %{x}<br>WR: %{z:.1f}%<extra></extra>",
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=280, margin=dict(l=0,r=0,t=10,b=0),
                    font=dict(color="#9090A0", size=11),
                )
                st.plotly_chart(fig)

    # Equity curve
    st.markdown("**📊 Equity Curve**")
    if trades:
        df = pd.DataFrame(trades).sort_values("open_time")
        df["cumulative_pnl"] = df["pnl_usd"].astype(float).cumsum()
        initial = st.session_state.equity
        df["equity"] = initial + df["cumulative_pnl"]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df["open_time"], y=df["equity"],
            mode="lines", name="Equity",
            line=dict(color="#B8860B", width=2),
            fill="tozeroy", fillcolor="rgba(184,134,11,0.1)",
        ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(l=0,r=0,t=10,b=0),
            font=dict(color="#9090A0", size=11),
            xaxis=dict(gridcolor="#1A1A28", color="#9090A0"),
            yaxis=dict(gridcolor="#1A1A28", color="#9090A0", side="right"),
            showlegend=False,
        )
        st.plotly_chart(fig3)

    # Optimizer log
    st.markdown("**⚙️ Optimizer Weights**")
    w = st.session_state.strategy_weights
    df_w = pd.DataFrame([{"Strategy": k.replace("_"," ").title(), "Weight": f"{v:.3f}×",
                           "Status": "↑ Boosted" if v > 1.1 else "↓ Reduced" if v < 0.9 else "— Normal"}
                         for k, v in w.items()])
    st.dataframe(df_w, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════
#  BOT CYCLE — runs when bot is active
# ══════════════════════════════════════════════════════════════════════
def run_bot_cycle():
    """Execute one full analysis + signal cycle."""
    if not st.session_state.bot_running:
        return

    try:
        st.session_state.cycle_count += 1

        # Fetch data
        df_m1  = DATA.get("1m",  100)
        df_m15 = DATA.get("15m", 200)
        df_h1  = DATA.get("1h",  300)
        df_h4  = DATA.get("4h",  200)

        if df_h1.empty or len(df_h1) < 50:
            return

        df_h1e  = Indicators.enrich(df_h1)
        df_m15e = Indicators.enrich(df_m15) if not df_m15.empty and len(df_m15) > 30 else None

        # Context
        regime  = detect_regime(df_h1e)
        session = detect_session()
        smc_h1  = SMC_ENG.analyze(df_h1e)
        smc_m15 = SMC_ENG.analyze(df_m15e) if df_m15e is not None else None
        from core.smc_engine import SMCData
        if smc_m15 is None: smc_m15 = SMCData()

        sr_levels = combine_sr(df_h1e, df_m15e if df_m15e is not None else df_h1e)

        # Signals
        signals = evaluate_all(
            df_m1, df_m15e, df_h1e, df_h4,
            smc_h1, smc_m15, sr_levels, regime, session,
            st.session_state.strategy_weights,
        )
        # Store as dicts so _render_signal_cards can call .get()
        st.session_state.last_signals = [s.to_dict() for s in signals]

        # Save signals to DB
        for sig in signals:
            save_signal(sig.to_dict(), acted_on=False)

        # Risk check + execute top signal
        rs = st.session_state.risk_state
        if rs and signals:
            rm = RiskManager(rs)
            can, reason = rm.can_trade()
            if can:
                top = signals[0]
                # DXY filter
                dxy_df = DATA.get_dxy(30)
                dxy_b  = get_dxy_bias(dxy_df)
                aligned, conf_mod = dxy_alignment(dxy_b, top.direction)
                top.confidence = min(100, top.confidence + conf_mod)

                if top.confidence >= 65:
                    lot = rm.lot_size(top)
                    trade_record = {
                        "strategy": top.strategy, "direction": top.direction,
                        "entry_price": top.entry, "sl": top.sl,
                        "tp1": top.tp1, "tp2": top.tp2,
                        "lot_size": lot, "atr": top.atr,
                        "confidence": top.confidence,
                        "session": top.session, "regime": top.regime,
                        "timeframe": top.timeframe,
                        "status": "open",
                        "open_time": datetime.now(timezone.utc).isoformat(),
                        "notes": top.notes,
                    }
                    trade_id = save_trade(trade_record)
                    trade_record["id"] = trade_id
                    rs.open_trades += 1
                    st.session_state.open_trades.append(trade_record)
                    save_signal(top.to_dict(), acted_on=True)

        # Walk-forward optimizer check
        if should_run(st.session_state.optimizer_last, interval_hours=4):
            trades = get_trades(limit=500)
            new_w = run_optimizer(trades, st.session_state.strategy_weights)
            st.session_state.strategy_weights = new_w
            st.session_state.optimizer_last   = datetime.now(timezone.utc)

        # Monitor open trades (paper)
        if rs and st.session_state.dry_run:
            price = DATA.current_price()
            open_t = get_open_trades()
            for t in open_t:
                entry  = float(t.get("entry_price") or 0)
                sl_p   = float(t.get("sl") or 0)
                tp2_p  = float(t.get("tp2") or 0)
                direct = t.get("direction","")
                if price <= 0 or entry <= 0:
                    continue
                if direct == "BUY":
                    if price <= sl_p:
                        pips = (price - entry) / 0.1
                        _close_trade(t, price, "sl_hit", pips, rs)
                    elif price >= tp2_p:
                        pips = (price - entry) / 0.1
                        _close_trade(t, price, "tp2_hit", pips, rs)
                elif direct == "SELL":
                    if price >= sl_p:
                        pips = (entry - price) / 0.1
                        _close_trade(t, price, "sl_hit", pips, rs)
                    elif price <= tp2_p:
                        pips = (entry - price) / 0.1
                        _close_trade(t, price, "tp2_hit", pips, rs)

    except Exception as e:
        st.session_state.errors.append(f"{datetime.now().strftime('%H:%M')} {e}")


def _close_trade(t: dict, price: float, status: str, pips: float, rs: RiskState):
    lot    = float(t.get("lot_size") or 0.01)
    pnl    = pips * lot
    entry  = float(t.get("entry_price") or price)
    opened = t.get("open_time", "")
    dur    = 0.0
    if opened:
        try:
            dur = (datetime.now(timezone.utc) - datetime.fromisoformat(opened)).total_seconds() / 60
        except Exception:
            pass
    rr = abs(price - entry) / max(abs(entry - float(t.get("sl") or entry)), 1e-6)
    update_trade(t["id"], {
        "exit_price": price, "status": status,
        "pnl_pips": round(pips, 1), "pnl_usd": round(pnl, 2),
        "close_time": datetime.now(timezone.utc).isoformat(),
        "duration_min": round(dur, 1), "rr_actual": round(rr, 2),
    })
    RiskManager(rs).on_close(pips, pnl)


# ══════════════════════════════════════════════════════════════════════
#  MAIN RENDER
# ══════════════════════════════════════════════════════════════════════
def main():
    render_sidebar()
    render_header()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "📡 Signals",
        "📋 Trade History",
        "🔬 Analytics",
    ])

    with tab1: tab_dashboard()
    with tab2: tab_signals()
    with tab3: tab_trades()
    with tab4: tab_analytics()

    # Run bot cycle
    run_bot_cycle()

    # Auto-rerun every 60s when bot is running
    if st.session_state.bot_running:
        time.sleep(0.5)
        st.rerun()


if __name__ == "__main__":
    main()
