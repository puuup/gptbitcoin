import time
import datetime as dt
import sqlite3
import pandas as pd
import streamlit as st
import altair as alt

DB_PATH = "trading_history.db"

def load_data(table, limit=1000):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # 현재 테이블 컬럼 목록 조회
    cur.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cur.fetchall()]

    # 정렬 기준 컬럼 자동 선택: timestamp > created_at > id
    if "timestamp" in cols:
        order_col = "timestamp"
    elif "created_at" in cols:
        order_col = "created_at"
    else:
        order_col = "id"

    df = pd.read_sql_query(
        f"SELECT * FROM {table} ORDER BY {order_col} DESC LIMIT ?",
        conn,
        params=(limit,)
    )
    conn.close()
    return df

def load_trading_history_filtered(start_ts: str, end_ts: str, decisions: list, limit: int = 1000) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # 컬럼 존재 확인
    cur.execute("PRAGMA table_info(trading_history)")
    cols = [r[1] for r in cur.fetchall()]

    base_sql = "SELECT * FROM trading_history WHERE 1=1"
    params = []

    # 날짜 필터 (timestamp 존재)
    if "timestamp" in cols:
        base_sql += " AND timestamp BETWEEN ? AND ?"
        params += [start_ts, end_ts]

    # 거래유형(의사결정) 필터
    if decisions:
        placeholders = ",".join(["?"] * len(decisions))
        base_sql += f" AND decision IN ({placeholders})"
        params += decisions

    # 정렬 컬럼
    order_col = "timestamp" if "timestamp" in cols else ("id" if "id" in cols else "rowid")
    base_sql += f" ORDER BY {order_col} DESC LIMIT ?"
    params.append(limit)

    df = pd.read_sql_query(base_sql, conn, params=params)
    conn.close()
    return df

def load_table_with_date(table: str, start_ts: str, end_ts: str, limit: int = 1000) -> pd.DataFrame:
    """portfolio_snapshots / ai_reflections 등에 날짜 필터 적용"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]

    # 시간 컬럼 감지
    if "timestamp" in cols:
        time_col = "timestamp"
    elif "created_at" in cols:
        time_col = "created_at"
    else:
        time_col = None

    if time_col:
        sql = f"SELECT * FROM {table} WHERE {time_col} BETWEEN ? AND ? ORDER BY {time_col} DESC LIMIT ?"
        params = (start_ts, end_ts, limit)
    else:
        sql = f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?"
        params = (limit,)

    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df

# --- 페이지 설정 ---
st.set_page_config(page_title="AutoTrade Dashboard", layout="wide")
alt.data_transformers.disable_max_rows()

# --- 공통: 날짜/필터 UI (사이드바) ---
today = dt.date.today()
default_start = today - dt.timedelta(days=7)

st.sidebar.header("필터")
date_range = st.sidebar.date_input(
    "날짜 범위",
    value=(default_start, today),
    max_value=today,
)

# date_range는 단일/범위 모두 가능하니 안전하게 처리
if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date, end_date = date_range, date_range

# sqlite 필터용 문자열로 변환
start_ts = f"{start_date} 00:00:00"
end_ts   = f"{end_date} 23:59:59"

# 거래유형(의사결정) 필터
decision_options = ["buy", "sell", "hold"]
selected_decisions = st.sidebar.multiselect(
    "거래 유형(의사결정)",
    options=decision_options,
    default=decision_options
)

# 자동 새로고침
auto_refresh = st.sidebar.checkbox("자동 새로고침", value=False)
refresh_sec = st.sidebar.slider("새로고침 간격(초)", min_value=5, max_value=300, value=30, step=5)

st.title("📊 AI 자동매매 실시간 대시보드")

st.title("자동 매매 모니터링 대시보드")

# 1) trading_history: 날짜 + 거래유형 필터 반영
st.subheader("매매 기록 (trading_history)")
df_th = load_trading_history_filtered(start_ts, end_ts, selected_decisions, limit=1000)
st.caption(f"필터: {start_date} ~ {end_date}, 유형: {', '.join(selected_decisions) if selected_decisions else '없음'}")
st.dataframe(df_th, use_container_width=True)

# 2) portfolio_snapshots: 날짜만
st.subheader("포트폴리오 스냅샷 (portfolio_snapshots)")
df_ps = load_table_with_date("portfolio_snapshots", start_ts, end_ts, limit=1000)
st.dataframe(df_ps, use_container_width=True)

# 3) ai_reflections: created_at 기준 날짜 필터
st.subheader("회고(Reflections) (ai_reflections)")
df_ref = load_table_with_date("ai_reflections", start_ts, end_ts, limit=100)
st.dataframe(df_ref, use_container_width=True)

# (선택) 간단 KPI 예시
col1, col2, col3 = st.columns(3)
try:
    col1.metric("의사결정 수", len(df_th))
    col2.metric("매수/매도/보유", f"{(df_th['decision']=='buy').sum()} / {(df_th['decision']=='sell').sum()} / {(df_th['decision']=='hold').sum()}")
    if 'profit_rate' in df_th.columns and not df_th['profit_rate'].isna().all():
        col3.metric("최근 수익률(가장 최신)", f"{df_th['profit_rate'].dropna().iloc[0]:+.2f}%")
    else:
        col3.metric("최근 수익률(가장 최신)", "N/A")
except Exception:
    pass

# --- 자동 새로고침 ---
if auto_refresh:
    # 페이지 하단에 간단 안내
    st.info(f"자동 새로고침: {refresh_sec}초 간격")
    time.sleep(int(refresh_sec))
    st.rerun()

# --- 탭 구성 ---
tab1, tab2, tab3 = st.tabs(["매매 기록", "포트폴리오 추이", "AI 회고(Reflection)"])

# =========================
# 1. 매매 기록 탭
# =========================
with tab1:
    st.subheader("최근 매매 기록")
    df_trades = load_data("trading_history", limit=200)

    if "timestamp" in df_trades.columns:
        df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"], errors="coerce")

    if not df_trades.empty:
        st.dataframe(df_trades, use_container_width=True)

        # --- 시각화: 결정별 분포 ---
        st.markdown("#### 매매 결정 비율")
        decision_counts = df_trades["decision"].value_counts().reset_index()
        decision_counts.columns = ["decision", "count"]
        chart = alt.Chart(decision_counts).mark_arc(innerRadius=50).encode(
            theta="count:Q",
            color="decision:N",
            tooltip=["decision", "count"]
        )
        st.altair_chart(chart, use_container_width=True)

        # --- 시각화: 수익률 히스토리 ---
        st.markdown("#### 수익률 추이")
        if "profit_rate" in df_trades.columns:
            line_chart = alt.Chart(df_trades).mark_line(point=True).encode(
                x="timestamp:T",
                y="profit_rate:Q",
                color="decision:N",
                tooltip=["timestamp", "decision", "profit_rate"]
            )
            st.altair_chart(line_chart, use_container_width=True)
    else:
        st.info("매매 기록이 없습니다.")

# =========================
# 2. 포트폴리오 탭
# =========================
with tab2:
    st.subheader("포트폴리오 추이")
    df_port = load_data("portfolio_snapshots", limit=500)

    if not df_port.empty:
        st.dataframe(df_port, use_container_width=True)

        # --- 자산 추이 ---
        st.markdown("#### 총 자산 변화")
        asset_chart = alt.Chart(df_port).mark_line(point=True).encode(
            x="timestamp:T",
            y="total_asset:Q",
            tooltip=["timestamp", "krw_balance", "btc_balance", "total_asset", "profit_rate"]
        )
        st.altair_chart(asset_chart, use_container_width=True)

        # --- 수익률 추이 ---
        st.markdown("#### 수익률 추이")
        profit_chart = alt.Chart(df_port).mark_line(point=True, color="green").encode(
            x="timestamp:T",
            y="profit_rate:Q",
            tooltip=["timestamp", "profit_rate"]
        )
        st.altair_chart(profit_chart, use_container_width=True)
    else:
        st.info("포트폴리오 데이터가 없습니다.")

# =========================
# 3. AI 회고 탭
# =========================
with tab3:
    st.subheader("AI Reflection 기록")
    df_ref = load_data("ai_reflections", limit=100)

    if not df_ref.empty:
        st.dataframe(df_ref, use_container_width=True)

        # 교훈 요약
        st.markdown("#### 최근 교훈 요약")
        for idx, row in df_ref.head(5).iterrows():
            st.write(f"- {row['created_at']} | Lessons: {row['lessons']} | WinRate: {row['win_rate']:.2f}, AvgReturn: {row['avg_return']:.2f}")
    else:
        st.info("회고 데이터가 없습니다.")