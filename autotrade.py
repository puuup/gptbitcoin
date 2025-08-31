import os
import json
import time
import base64
import pyupbit
import pandas as pd
import requests
import sqlite3
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
from io import BytesIO
from PIL import Image
from typing import Optional

# TA
import ta
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volume import ChaikinMoneyFlowIndicator

# Selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# YouTube transcript
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript,
)

# --- Constants ---
load_dotenv()
UPBIT_CHART_URL = "https://upbit.com/full_chart?code=CRIX.UPBIT.KRW-BTC"
XPATH_TIME_BUTTON = "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]/span/cq-clickable"
XPATH_TIME_OPTION_1H = "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]/cq-menu-dropdown/cq-item[8]"
XPATH_INDICATOR_BUTTON = "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/span"
XPATH_BB_OPTION = "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/cq-menu-dropdown/cq-scroll/cq-studies/cq-studies-content/cq-item[14]"

# --- Small shared helpers ---
def _safe_click(driver, xpath, timeout=20, sleep_after=0.6):
    el = WebDriverWait(driver, timeout).until(
        EC.element_to_be_clickable((By.XPATH, xpath))
    )
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", el)
    try:
        el.click()
    except Exception:
        driver.execute_script("arguments[0].click();", el)
    if sleep_after:
        time.sleep(sleep_after)
    return el

def parse_orderbook_safely(orderbook, current_price, include_totals=False):
    default_values = {"best_ask_price": current_price, "best_bid_price": current_price}
    if include_totals:
        default_values.update({"total_ask_size": 0, "total_bid_size": 0})
    if not orderbook or len(orderbook) == 0:
        return default_values
    try:
        ob = orderbook[0] if isinstance(orderbook, list) else orderbook
        result = {
            "best_ask_price": ob["orderbook_units"][0]["ask_price"],
            "best_bid_price": ob["orderbook_units"][0]["bid_price"],
        }
        if include_totals:
            result.update(
                {
                    "total_ask_size": ob.get("total_ask_size", 0),
                    "total_bid_size": ob.get("total_bid_size", 0),
                }
            )
        return result
    except (KeyError, IndexError, TypeError):
        return default_values

def get_news_trading_signal(sentiment, score):
    if sentiment == "positive" and score > 0.2:
        return "strong_buy"
    elif sentiment == "positive" and score > 0.1:
        return "buy"
    elif sentiment == "negative" and score < -0.2:
        return "strong_sell"
    elif sentiment == "negative" and score < -0.1:
        return "sell"
    else:
        return "neutral"

def get_fng_signal(value, classification):
    if value <= 20:
        return "strong_buy"
    elif value <= 35:
        return "buy"
    elif value <= 45:
        return "neutral_buy"
    elif value <= 55:
        return "neutral"
    elif value <= 65:
        return "neutral_sell"
    elif value <= 80:
        return "sell"
    else:
        return "strong_sell"

# --- DB Layer ---
class DB:
    def __init__(self, path="trading_history.db"):
        self.path = path
        self.init()

    def _conn(self):
        return sqlite3.connect(self.path)

    def init(self):
        try:
            conn = self._conn()
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS trading_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    decision TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    percentage REAL NOT NULL,
                    reason TEXT,

                    current_price REAL NOT NULL,
                    krw_balance REAL NOT NULL,
                    btc_balance REAL NOT NULL,
                    btc_avg_buy_price REAL,
                    btc_krw_price REAL NOT NULL,
                    total_asset REAL NOT NULL,
                    profit_rate REAL,

                    daily_trend TEXT,
                    hourly_momentum TEXT,
                    fear_greed_value INTEGER,
                    fear_greed_signal TEXT,
                    news_sentiment TEXT,
                    confluence_analysis TEXT,

                    order_executed BOOLEAN DEFAULT FALSE,
                    order_type TEXT,
                    order_amount REAL,
                    order_result TEXT,
                    order_error TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    krw_balance REAL NOT NULL,
                    btc_balance REAL NOT NULL,
                    btc_avg_price REAL,
                    current_btc_price REAL NOT NULL,
                    total_asset REAL NOT NULL,
                    profit_rate REAL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_reflections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    window_hours INTEGER NOT NULL,
                    lookback INTEGER NOT NULL,
                    sample_size INTEGER NOT NULL,
                    win_rate REAL,
                    avg_return REAL,
                    lessons TEXT NOT NULL,          -- JSON array string
                    weight REAL DEFAULT 1.0,
                    applied_count INTEGER DEFAULT 0
                )
                """
            )

            conn.commit()
            conn.close()
            print("데이터베이스 초기화 완료")
        except Exception as e:
            print(f"데이터베이스 초기화 오류: {e}")

    def save_decision(self, decision_data, market_data):
        try:
            conn = self._conn()
            cur = conn.cursor()
            ts = decision_data.get("technical_summary", {}) or {}
            cur.execute(
                """
                INSERT INTO trading_history
                (decision, confidence, percentage, reason, current_price, krw_balance, btc_avg_buy_price, btc_krw_price,
                 btc_balance, total_asset, profit_rate, daily_trend, hourly_momentum,
                 fear_greed_value, fear_greed_signal, news_sentiment, confluence_analysis)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_data.get("decision", ""),
                    decision_data.get("confidence", ""),
                    decision_data.get("percentage", 0),
                    decision_data.get("reason", ""),
                    market_data.get("current_price", 0),
                    market_data.get("krw_balance", 0),
                    market_data.get("btc_balance", 0),
                    market_data.get("current_price", 0),
                    market_data.get("btc_balance", 0),
                    market_data.get("total_asset", 0),
                    market_data.get("profit_rate", 0),
                    ts.get("daily_trend", ""),
                    ts.get("hourly_momentum", ""),
                    (market_data.get("fear_greed_data") or {}).get("current_value", 0),
                    ts.get("fear_greed_signal", ""),
                    ts.get("news_sentiment", ""),
                    ts.get("confluence_analysis", ""),
                ),
            )
            rid = cur.lastrowid
            conn.commit()
            conn.close()
            print(f"매매 결정 저장 완료 (ID: {rid})")
            return rid
        except Exception as e:
            print(f"매매 결정 저장 오류: {e}")
            return None

    def update_order(self, record_id, executed, order_type, amount, result=None, error=None):
        try:
            conn = self._conn()
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE trading_history
                SET order_executed=?, order_type=?, order_amount=?, order_result=?, order_error=?
                WHERE id=?
                """,
                (
                    executed,
                    order_type,
                    amount,
                    json.dumps(result) if result else None,
                    error,
                    record_id,
                ),
            )
            conn.commit()
            conn.close()
            print(f"주문 결과 업데이트 완료 (ID: {record_id})")
        except Exception as e:
            print(f"주문 결과 업데이트 오류: {e}")

    def save_portfolio(self, krw_balance, btc_balance, btc_avg_price, btc_price):
        try:
            total_asset = krw_balance + btc_balance * btc_price
            profit_rate = 0
            if btc_balance > 0 and btc_avg_price > 0:
                profit_rate = ((btc_price - btc_avg_price) / btc_avg_price) * 100
            conn = self._conn()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO portfolio_snapshots
                (krw_balance, btc_balance, btc_avg_price, current_btc_price, total_asset, profit_rate)
                VALUES (?,?,?,?,?,?)
                """,
                (krw_balance, btc_balance, btc_avg_price, btc_price, total_asset, profit_rate),
            )
            conn.commit()
            conn.close()
            print("포트폴리오 스냅샷 저장 완료")
        except Exception as e:
            print(f"포트폴리오 스냅샷 저장 오류: {e}")

    def print_stats(self, days=30):
        try:
            conn = self._conn()
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT decision, confidence, order_executed, profit_rate, timestamp
                FROM trading_history
                WHERE timestamp >= datetime('now','-{days} days')
                ORDER BY timestamp DESC
                """
            )
            rows = cur.fetchall()
            if rows:
                total = len(rows)
                exe = sum(1 for r in rows if r[2])
                buys = sum(1 for r in rows if r[0] == "buy")
                sells = sum(1 for r in rows if r[0] == "sell")
                holds = sum(1 for r in rows if r[0] == "hold")
                print(f"\n=== 최근 {days}일 매매 통계 ===")
                print(f"총 결정 횟수: {total}")
                print(f"실제 실행: {exe}")
                print(f"매수 결정: {buys}")
                print(f"매도 결정: {sells}")
                print(f"보유 결정: {holds}")
                latest_profit = rows[0][3] if rows[0][3] else 0
                print(f"최근 수익률: {latest_profit:+.2f}%")
            conn.close()
        except Exception as e:
            print(f"통계 조회 오류: {e}")

    def get_recent_decisions(self, limit=50):
        """최근 매매 의사결정 일부를 가져온다(회고 평가용 간략 필드 세트)."""
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, timestamp, decision, current_price,
                daily_trend, hourly_momentum, fear_greed_value, news_sentiment
            FROM trading_history
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        conn.close()
        # dict 변환
        out = []
        for r in rows:
            out.append({
                "id": r[0],
                "timestamp": r[1],
                "decision": r[2],
                "entry_price": float(r[3]) if r[3] is not None else None,
                "daily_trend": r[4] or "",
                "hourly_momentum": r[5] or "",
                "fear_greed_value": int(r[6]) if r[6] not in (None, "") else None,
                "news_sentiment": (r[7] or "").lower(),
            })
        return out

    def save_reflection(self, window_hours, lookback, sample_size, metrics: dict, lessons: list, weight: float = 1.0):
        """회고 결과(교훈 + 메트릭)를 저장."""
        try:
            conn = self._conn()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO ai_reflections
                (window_hours, lookback, sample_size, win_rate, avg_return, lessons, weight)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(window_hours), int(lookback), int(sample_size),
                    float(metrics.get("win_rate", 0.0)),
                    float(metrics.get("avg_return", 0.0)),
                    json.dumps(lessons, ensure_ascii=False),
                    float(weight),
                ),
            )
            conn.commit()
            conn.close()
            print("회고(Reflection) 저장 완료")
        except Exception as e:
            print(f"회고 저장 오류: {e}")

    def get_recent_lessons_text(self, max_items=5, max_chars=800, min_weight=0.5):
        """최근 회고의 교훈을 텍스트로 합성하여 시스템 메시지에 주입할 수 있게 반환."""
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT lessons, weight
            FROM ai_reflections
            ORDER BY created_at DESC
            LIMIT 10
            """
        )
        rows = cur.fetchall()
        conn.close()

        merged = []
        for ls_json, w in rows:
            try:
                if w is not None and float(w) < min_weight:
                    continue
            except:
                pass
            try:
                arr = json.loads(ls_json) if ls_json else []
                for s in arr:
                    if isinstance(s, str):
                        merged.append(s.strip())
            except:
                continue

        # 중복 제거 & 잘라내기
        uniq = []
        seen = set()
        for s in merged:
            if s and s not in seen:
                uniq.append(s); seen.add(s)
            if len(uniq) >= max_items:
                break

        text = ""
        for s in uniq:
            line = f"- {s}\n"
            if len(text) + len(line) > max_chars:
                break
            text += line
        return text.strip()

    def mark_lessons_applied(self, count=1):
        """가장 최근 회고 레코드의 applied_count 증가."""
        try:
            conn = self._conn()
            cur = conn.cursor()
            cur.execute(
                "UPDATE ai_reflections SET applied_count = applied_count + ? WHERE id = (SELECT id FROM ai_reflections ORDER BY created_at DESC LIMIT 1)",
                (int(count),),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"회고 적용 카운트 업데이트 오류: {e}")

# --- Market / Indicators ---
class MarketData:
    def __init__(self):
        pass

    def get_ohlcv(self):
        print("차트 데이터 조회 중...")
        df_daily = pyupbit.get_ohlcv("KRW-BTC", count=50, interval="day")
        df_hourly = pyupbit.get_ohlcv("KRW-BTC", count=48, interval="minute60")
        return df_daily, df_hourly

    def add_indicators(self, df):
        try:
            if df is None or len(df) < 20:
                print("데이터가 부족하여 기술적 지표를 계산할 수 없습니다.")
                return df
            df.columns = [c.lower() for c in df.columns]
            df = df.dropna()
            if len(df) < 14:
                print("기술적 지표 계산을 위한 데이터가 부족합니다.")
                return df

            df["sma_5"] = SMAIndicator(df["close"], 5).sma_indicator()
            df["sma_10"] = SMAIndicator(df["close"], 10).sma_indicator()
            df["sma_20"] = SMAIndicator(df["close"], 20).sma_indicator()
            df["ema_12"] = EMAIndicator(df["close"], 12).ema_indicator()
            df["ema_26"] = EMAIndicator(df["close"], 26).ema_indicator()

            df["rsi"] = RSIIndicator(df["close"], 14).rsi()

            macd = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
            df["macd_diff"] = macd.macd_diff()

            bb = BollingerBands(df["close"], 20, 2)
            df["bb_upper"] = bb.bollinger_hband()
            df["bb_middle"] = bb.bollinger_mavg()
            df["bb_lower"] = bb.bollinger_lband()
            df["bb_width"] = bb.bollinger_wband()
            df["bb_percent"] = bb.bollinger_pband()

            st = StochasticOscillator(df["high"], df["low"], df["close"], 14, 3)
            df["stoch_k"] = st.stoch()
            df["stoch_d"] = st.stoch_signal()

            if "volume" in df.columns:
                cmf = ChaikinMoneyFlowIndicator(
                    df["high"], df["low"], df["close"], df["volume"], 20
                )
                df["cmf"] = cmf.chaikin_money_flow()
            return df
        except Exception as e:
            print(f"기술적 지표 계산 오류: {e}")
            return df

    def analyze_indicators(self, df, timeframe):
        try:
            if df is None or len(df) == 0:
                return {}
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            a = {
                "timeframe": timeframe,
                "price_analysis": {
                    "current_price": float(latest["close"]),
                    "price_change": float(latest["close"] - prev["close"]),
                    "price_change_pct": float(
                        (latest["close"] - prev["close"]) / prev["close"] * 100
                    )
                    if prev["close"] != 0
                    else 0,
                },
            }
            if "rsi" in latest and pd.notna(latest["rsi"]):
                rsi = float(latest["rsi"])
                a["rsi"] = {
                    "value": rsi,
                    "signal": "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral",
                    "strength": "strong" if rsi < 20 or rsi > 80 else "weak",
                }
            if all(
                c in latest for c in ["macd", "macd_signal", "macd_diff"]
            ) and all(pd.notna(latest[c]) for c in ["macd", "macd_signal", "macd_diff"]):
                md = float(latest["macd_diff"])
                pmd = float(prev["macd_diff"]) if pd.notna(prev.get("macd_diff")) else md
                a["macd"] = {
                    "histogram": md,
                    "signal": "bullish" if (md > 0 and md > pmd) else "bearish" if (md < 0 and md < pmd) else "neutral",
                    "macd_value": float(latest["macd"]),
                    "signal_value": float(latest["macd_signal"]),
                }
            if all(c in latest for c in ["bb_upper", "bb_lower", "bb_percent"]) and all(
                pd.notna(latest[c]) for c in ["bb_upper", "bb_lower", "bb_percent"]
            ):
                bbp = float(latest["bb_percent"])
                a["bollinger"] = {
                    "position": bbp,
                    "signal": "oversold" if bbp < 0.2 else "overbought" if bbp > 0.8 else "neutral",
                    "upper": float(latest["bb_upper"]),
                    "lower": float(latest["bb_lower"]),
                    "width": float(latest["bb_width"]) if pd.notna(latest.get("bb_width")) else 0,
                }
            if all(c in latest for c in ["sma_5", "sma_20"]) and all(pd.notna(latest[c]) for c in ["sma_5", "sma_20"]):
                sma5 = float(latest["sma_5"])
                sma20 = float(latest["sma_20"])
                cp = float(latest["close"])
                a["moving_averages"] = {
                    "sma5": sma5,
                    "sma20": sma20,
                    "price_vs_sma5": "above" if cp > sma5 else "below",
                    "price_vs_sma20": "above" if cp > sma20 else "below",
                    "trend": "bullish" if sma5 > sma20 else "bearish",
                }
            if all(c in latest for c in ["stoch_k", "stoch_d"]) and all(pd.notna(latest[c]) for c in ["stoch_k", "stoch_d"]):
                k = float(latest["stoch_k"])
                d = float(latest["stoch_d"])
                a["stochastic"] = {
                    "k_value": k,
                    "d_value": d,
                    "signal": "oversold" if (k < 20 and d < 20) else "overbought" if (k > 80 and d > 80) else "neutral",
                    "crossover": "bullish" if k > d else "bearish",
                }
            return a
        except Exception as e:
            print(f"기술적 지표 분석 오류: {e}")
            return {}

    def balances_and_price(self, upbit):
        my_krw = upbit.get_balance("KRW")
        my_btc = upbit.get_balance("KRW-BTC")
        balances = upbit.get_balances()
        btc_avg_price = 0
        for b in balances or []:
            if b.get("currency") == "BTC":
                btc_avg_price = float(b.get("avg_buy_price", 0))
                break
        current_price = pyupbit.get_current_price("KRW-BTC")
        orderbook = pyupbit.get_orderbook(ticker="KRW-BTC")
        return my_krw, my_btc, balances, btc_avg_price, current_price, orderbook

# --- External signals (FNG, News, YouTube) ---
class ExternalSignals:
    def __init__(self):
        pass

    def fear_greed(self):
        try:
            print("공포탐욕지수 조회 중...")
            r = requests.get("https://api.alternative.me/fng/?limit=7", timeout=10)
            if r.status_code != 200:
                print(f"공포탐욕지수 API 요청 실패: {r.status_code}")
                return None
            data = r.json()
            if data["metadata"]["error"] is not None or not data["data"]:
                print(f"공포탐욕지수 API 오류: {data['metadata']['error']}")
                return None
            cur = data["data"][0]
            v = int(cur["value"])
            cls = cur["value_classification"]
            vals = [int(x["value"]) for x in data["data"]]
            avg7 = sum(vals) / len(vals)
            change = v - vals[-1] if len(vals) >= 7 else 0
            analysis = {
                "current_value": v,
                "classification": cls,
                "7day_average": round(avg7, 1),
                "change_from_week": change,
                "trend": "improving" if change > 0 else "declining" if change < 0 else "stable",
                "market_signal": get_fng_signal(v, cls),
                "historical_data": vals,
            }
            print(f"공포탐욕지수: {v} ({cls}), 7일평균 {avg7:.1f}, Δ주간 {change:+d}")
            return analysis
        except Exception as e:
            print(f"공포탐욕지수 조회 오류: {e}")
            return None

    def news(self):
        try:
            print("암호화폐 뉴스 조회 중...")
            key = os.getenv("SERPAPI_KEY")
            if not key:
                print("SERPAPI_KEY 미설정 → 뉴스 분석 건너뜀")
                return None
            queries = ["bitcoin cryptocurrency", "bitcoin market analysis"]
            all_news = []
            for q in queries:
                try:
                    url = "https://serpapi.com/search.json"
                    params = {"engine": "google_news", "q": q, "gl": "us", "hl": "en", "api_key": key}
                    r = requests.get(url, params=params, timeout=10)
                    if r.status_code == 200:
                        j = r.json()
                        for n in (j.get("news_results") or [])[:3]:
                            all_news.append(
                                {
                                    "title": n.get("title", ""),
                                    "snippet": n.get("snippet", ""),
                                    "source": (n.get("source") or {}).get("name", ""),
                                    "date": n.get("date", ""),
                                    "query": q,
                                }
                            )
                        time.sleep(1)
                except Exception as e:
                    print(f"뉴스 조회 오류({q}): {e}")
            if not all_news:
                print("수집된 뉴스가 없습니다.")
                return None
            analysis = self._analyze_news(all_news)
            print(f"✅ 뉴스 {len(all_news)}개 수집, 감정={analysis['overall_sentiment']}")
            return {"news_items": all_news, "analysis": analysis, "total_count": len(all_news)}
        except Exception as e:
            print(f"뉴스 조회 전체 오류: {e}")
            return None

    def _analyze_news(self, news_items):
        pos_k = [
            "bull", "bullish", "surge", "rally", "rise", "gain", "up", "increase",
            "adoption", "breakthrough", "positive", "growth", "soar", "climb",
            "institutional", "etf", "approval", "investment", "buying",
        ]
        neg_k = [
            "bear", "bearish", "crash", "fall", "drop", "decline", "down", "decrease",
            "regulation", "ban", "crackdown", "negative", "concern", "fear", "sell",
            "plunge", "collapse", "warning", "risk", "volatility", "uncertainty",
        ]
        neu_k = [
            "analysis", "market", "trading", "price", "trend", "forecast",
            "prediction", "outlook", "technical", "chart", "support", "resistance",
        ]
        pos = neg = neu = 0
        all_words = []
        for n in news_items:
            t = f"{n.get('title','')} {n.get('snippet','')}".lower()
            for k in pos_k:
                if k in t:
                    c = t.count(k)
                    pos += c; all_words += [k]*c
            for k in neg_k:
                if k in t:
                    c = t.count(k)
                    neg += c; all_words += [k]*c
            for k in neu_k:
                if k in t:
                    neu += t.count(k)
        total = pos + neg + neu
        if total == 0:
            sentiment, score = "neutral", 0
        else:
            score = (pos - neg) / total
            sentiment = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
        from collections import Counter
        key_kw = [w for w, _ in Counter(all_words).most_common(10)]
        return {
            "overall_sentiment": sentiment,
            "sentiment_score": score,
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": neu,
            "key_keywords": key_kw,
            "news_signal": get_news_trading_signal(sentiment, score),
        }

    def youtube_bundle(self, video_ids, languages=("en", "en-US", "ko"), max_items=2000):
        results, all_texts = [], []
        for vid in video_ids:
            try:
                transcript = YouTubeTranscriptApi.list_transcripts(vid).find_transcript(languages)
                fetched = transcript.fetch()
                text_list, count = [], 0
                for snip in fetched:
                    t = (snip.get("text") or "").strip()
                    if not t:
                        continue
                    if t.lower() in {"[music]", "(music)", "[applause]"}:
                        continue
                    text_list.append(t)
                    count += 1
                    if count >= max_items:
                        break
                if text_list:
                    results.append({"video_id": vid, "text_list": text_list})
                    all_texts.extend(text_list)
            except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript) as e:
                print(f"[YouTube] 자막 불가({vid}): {e}")
            except Exception as e:
                print(f"[YouTube] 예외({vid}): {e}")
        return {"items": results, "all_text": " ".join(all_texts) if all_texts else ""}

    def youtube_sentiment(self, all_text):
        if not all_text:
            return None
        text = all_text.lower()
        pos_k = [
            "bull", "bullish", "surge", "rally", "breakout", "accumulate", "uptrend",
            "support holds", "buy", "strong buy", "golden cross", "institutional buying",
            "adoption", "approval", "etf inflow",
        ]
        neg_k = [
            "bear", "bearish", "crash", "sell-off", "breakdown", "distribution", "downtrend",
            "resistance rejection", "sell", "strong sell", "death cross", "ban", "crackdown",
            "outflow", "fud",
        ]
        neu_k = ["range", "sideways", "consolidation", "volatility", "uncertainty", "wait", "neutral", "no trade", "flat"]
        pos = sum(text.count(k) for k in pos_k)
        neg = sum(text.count(k) for k in neg_k)
        neu = sum(text.count(k) for k in neu_k)
        total = pos + neg + neu
        if total == 0:
            overall, score = "neutral", 0.0
        else:
            score = (pos - neg) / total
            overall = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
        if overall == "positive" and score > 0.2:
            yt = "strong_buy"
        elif overall == "positive" and score > 0.1:
            yt = "buy"
        elif overall == "negative" and score < -0.2:
            yt = "strong_sell"
        elif overall == "negative" and score < -0.1:
            yt = "sell"
        else:
            yt = "neutral"
        from collections import Counter
        words = []
        for k in pos_k + neg_k:
            if k in text:
                words += [k]*text.count(k)
        top = [w for w, _ in Counter(words).most_common(10)]
        return {"overall_sentiment": overall, "sentiment_score": score, "key_keywords": top, "yt_signal": yt}

# --- Chart Capture ---
class ChartCapture:
    def capture_chart_base64(self, url=UPBIT_CHART_URL, save_dir="./screenshots"):
        os.makedirs(save_dir, exist_ok=True)
        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--hide-scrollbars")
        opts.add_experimental_option("excludeSwitches", ["enable-logging"])
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
        try:
            driver.get(url)
            WebDriverWait(driver, 25).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(2)
            _safe_click(driver, XPATH_TIME_BUTTON, 25)
            _safe_click(driver, XPATH_TIME_OPTION_1H, 25, 1.2)
            _safe_click(driver, XPATH_INDICATOR_BUTTON, 25)
            _safe_click(driver, XPATH_BB_OPTION, 25, 1.5)
            time.sleep(1.5)
            layout = driver.execute_cdp_cmd("Page.getLayoutMetrics", {})
            size = layout["contentSize"]
            width, height = int(size["width"]), int(size["height"])
            result = driver.execute_cdp_cmd(
                "Page.captureScreenshot",
                {
                    "format": "png",
                    "fromSurface": True,
                    "clip": {"x": 0, "y": 0, "width": width, "height": height, "scale": 1},
                    "captureBeyondViewport": True,
                },
            )
            raw_png_b64 = result["data"]
            raw_bytes = base64.b64decode(raw_png_b64)
            img = Image.open(BytesIO(raw_bytes))
            img.thumbnail((800, 800), Image.LANCZOS)
            buf = BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=70, optimize=True)
            optimized = buf.getvalue()
            optimized_b64 = base64.b64encode(optimized).decode("utf-8")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(save_dir, f"upbit_btc_1h_bb_{ts}.jpg")
            with open(out_path, "wb") as f:
                f.write(optimized)
            return out_path, optimized_b64
        finally:
            driver.quit()

# --- OpenAI Decision ---
class DecisionAI:
    def __init__(self):
        self.client = OpenAI()

    def build_system_msg(self, reflection_text: Optional[str] = None):
        base = {
            "role": "system",
            "content": (
                "You are an expert crypto technical analyst.\n"
                "Rules summary: RSI <30=buy, >70=sell; MACD uptrend=buy; "
                "Bollinger near lower=buy/upper=sell; Stoch K>D oversold=buy, K<D overbought=sell; "
                "MAs price>MA & rising=buy; F&G: 0-25 strong_buy, 25-45 buy, 45-55 neutral, "
                "55-75 sell, 75-100 strong_sell; News: positive=buy, negative=sell; "
                "Confluence 3=high, 2=medium, else=low; Risk sizing high:60-80, mid:30-50, low:20-30.\n"
                "Return ONLY JSON with keys: decision, reason, confidence, percentage, technical_summary "
                "(daily_trend, hourly_momentum, fear_greed_signal, market_sentiment, news_sentiment, "
                "news_signal, confluence_analysis, key_indicators, risk_factors)."
            ),
        }
        if reflection_text:
            base["content"] += (
                "\n\nRecent reflections (must be applied unless overwhelming contrary evidence):\n"
                f"{reflection_text}\n"
                "When reflections conflict with raw indicators, lower confidence and position sizing accordingly."
            )
        return base

    def build_user_content(self, analysis_data_json, yt_excerpt, chart_b64):
        user_content = [
            {
                "type": "text",
                "text": (
                    "Below is the structured analysis_data JSON to consider.\n"
                    "If an image is provided, analyze it as a recent BTC/KRW chart (1H, Bollinger Bands).\n\n"
                    + analysis_data_json
                ),
            }
        ]
        if yt_excerpt:
            user_content.append(
                {
                    "type": "text",
                    "text": "Below is an excerpt from recent YouTube transcripts:\n" + yt_excerpt,
                }
            )
        if chart_b64:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{chart_b64}", "detail": "low"},
                }
            )
        return user_content

    def request_decision(self, user_content, reflection_text: Optional[str] = None):
        resp = self.client.chat.completions.create(
            model="gpt-5",
            messages=[
                self.build_system_msg(reflection_text),
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)

# --- Trader ---
class Trader:
    def __init__(self, upbit: pyupbit.Upbit, db: DB):
        self.upbit = upbit
        self.db = db

    @staticmethod
    def calc_amount(decision, percentage, my_krw, my_btc):
        pct = float(percentage) / 100.0
        if decision == "buy":
            return my_krw * 0.9995 * pct
        elif decision == "sell":
            return my_btc * pct
        return 0

    @staticmethod
    def validate_min(order_type, amount, btc_price, min_amount=5000):
        if order_type == "buy":
            order_value = amount
        else:
            order_value = amount * btc_price
        if order_value < min_amount:
            action = "매수" if order_type == "buy" else "매도"
            print(f"❌ {action} 실패: 최소 주문 금액 부족 ({order_value:,.0f}원 < {min_amount:,.0f}원)")
            return False
        return True

    def execute(self, order_type, amount, record_id):
        action = {
            "buy": {"fn": self.upbit.buy_market_order, "label": "매수"},
            "sell": {"fn": self.upbit.sell_market_order, "label": "매도"},
        }.get(order_type)
        if not action:
            print(f"잘못된 주문 타입: {order_type}")
            return False
        print(f"\n=== {action['label']} 주문 실행 ===")
        try:
            res = action["fn"]("KRW-BTC", amount)
            if res:
                print(f"✅ {action['label']} 주문 성공!")
                print(f"주문 상세: {res}")
                self.db.update_order(record_id, True, order_type, amount, res, None)
                return True
            else:
                print(f"❌ {action['label']} 주문 실패")
                self.db.update_order(record_id, False, order_type, amount, None, f"{action['label']} 실패")
                return False
        except Exception as e:
            msg = f"{action['label']} 주문 중 예외: {e}"
            print(f"❌ {msg}")
            self.db.update_order(record_id, False, order_type, amount, None, msg)
            return False
        
class ReflectionEngine:
    @staticmethod
    def _forward_return(decision_ts: str, entry_price: float, horizon_hours: int = 6) -> float:
        """
        의사결정 시점 대비 horizon_hours 이후의 수익률(%)을 대략 추정.
        pyupbit.get_ohlcv의 'to' 인자를 활용해 해당 시점까지의 시세를 끊어 가져온 뒤 마지막 종가 사용.
        """
        try:
            # timestamp 문자열 → datetime
            base_dt = datetime.strptime(decision_ts.split('.')[0], "%Y-%m-%d %H:%M:%S")
        except:
            try:
                base_dt = datetime.fromisoformat(decision_ts)
            except:
                return 0.0

        end_dt = base_dt + timedelta(hours=horizon_hours)
        # end_dt 까지의 최근 캔들들 가져와 마지막 클로즈 사용
        to_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
        try:
            df = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", to=to_str, count=horizon_hours + 2)
            if df is None or len(df) == 0 or entry_price in (None, 0):
                return 0.0
            last_close = float(df["close"].iloc[-1])
            return (last_close - entry_price) / entry_price * 100.0
        except Exception:
            return 0.0

    @staticmethod
    def _is_correct(decision: str, fwd_ret_pct: float, thr: float = 0.5) -> bool:
        """
        간단 정확도 판정:
        - buy: 미래수익률 > +thr%  → 성공
        - sell: 미래수익률 < -thr% → 성공
        - hold: |미래수익률| < thr → 성공(변동 미약 시 홀드 OK)
        """
        if decision == "buy":
            return fwd_ret_pct > thr
        elif decision == "sell":
            return fwd_ret_pct < -thr
        else:
            return abs(fwd_ret_pct) <= thr

    @staticmethod
    def _derive_lessons(samples: list[dict]) -> list[str]:
        """
        틀린 의사결정들에서 반복되는 컨텍스트(공포/탐욕, 트렌드 등)로부터 간단한 교훈 생성.
        """
        lessons = []
        if not samples:
            return lessons

        # 집계
        total = len(samples)
        wrongs = [s for s in samples if not s["correct"]]
        if not wrongs:
            lessons.append("최근 의사결정은 대체로 올바랐음. 기존 규칙 유지.")
            return lessons

        # 영역별 카운트
        cnt = {
            "buy_in_greed": 0,
            "sell_in_fear": 0,
            "sell_against_bull_daily": 0,
            "buy_against_bear_daily": 0,
            "hold_while_strong_move": 0,
        }
        for s in wrongs:
            fg = s.get("fear_greed_value")
            daily = (s.get("daily_trend") or "").lower()
            hourly = (s.get("hourly_momentum") or "").lower()
            dr = s.get("decision")
            ret = s.get("fwd_ret_pct", 0.0)

            if dr == "buy" and fg is not None and fg >= 70:
                cnt["buy_in_greed"] += 1
            if dr == "sell" and fg is not None and fg <= 30:
                cnt["sell_in_fear"] += 1
            if dr == "sell" and daily == "bullish":
                cnt["sell_against_bull_daily"] += 1
            if dr == "buy" and daily == "bearish":
                cnt["buy_against_bear_daily"] += 1
            if dr == "hold" and abs(ret) >= 1.0:
                cnt["hold_while_strong_move"] += 1

        # 빈도 높은 항목을 교훈으로
        def add_if(label, text):
            if cnt[label] > 0:
                lessons.append(text)

        add_if("buy_in_greed", "탐욕 구간(≥70)에서는 매수 보수적: MACD 상승 + RSI ≤ 60 동시 충족 시에만 매수.")
        add_if("sell_in_fear", "공포 구간(≤30)에서는 성급한 매도 자제: 일봉 추세가 하락 확정일 때만 매도.")
        add_if("sell_against_bull_daily", "일봉이 상승(bullish)일 때는 단순한 1시간봉 약세만으로 매도하지 말 것. 이격 확대 시점까지 대기.")
        add_if("buy_against_bear_daily", "일봉이 하락(bearish)일 때는 1시간봉 단기 반등만으로 매수 제한. 볼밴 하단 이탈+RSI 과매도 회복 확인.")
        add_if("hold_while_strong_move", "강한 변동(±1% 이상) 중 HOLD 남발 자제: MACD 히스토그램 급변 시 비중 조절.")

        # 보편 규칙
        lessons.append("뉴스가 부정적이고 탐욕 높을 때는 매수 확신을 한 단계 낮춤.")
        lessons.append("공포+긍정 뉴스+일봉 상승이 동시에 나오면 매수 확신을 한 단계 높임.")

        # 중복 제거
        seen = set(); uniq = []
        for s in lessons:
            if s not in seen:
                uniq.append(s); seen.add(s)
        return uniq

    @staticmethod
    def run_and_get_lessons(db: "DB", horizon_hours: int = 6, lookback: int = 40, save: bool = True) -> str:
        """
        최근 의사결정(lookback)을 horizon_hours 시점의 결과로 평가하고 교훈을 DB에 저장.
        반환값은 시스템 메시지에 주입할 텍스트(불릿 리스트).
        """
        rows = db.get_recent_decisions(limit=lookback)
        if not rows:
            return ""

        samples = []
        rets = []
        for r in rows:
            if not r.get("entry_price"):
                continue
            fwd = ReflectionEngine._forward_return(r["timestamp"], r["entry_price"], horizon_hours=horizon_hours)
            correct = ReflectionEngine._is_correct(r["decision"], fwd)
            samples.append({
                **r,
                "fwd_ret_pct": fwd,
                "correct": correct,
            })
            rets.append(fwd)

        if not samples:
            return ""

        win_rate = sum(1 for s in samples if s["correct"]) / len(samples)
        avg_ret = sum(rets) / len(rets) if rets else 0.0

        lessons = ReflectionEngine._derive_lessons(samples)
        metrics = {"win_rate": round(win_rate, 3), "avg_return": round(avg_ret, 3)}

        if save:
            # 성능이 좋을수록 weight↑
            weight = 0.5 + max(0.0, win_rate - 0.4)  # 대략 0.5~1.1
            db.save_reflection(horizon_hours, lookback, len(samples), metrics, lessons, weight=weight)

        text = ""
        for s in lessons:
            line = f"- {s}\n"
            if len(text) + len(line) > 800:
                break
            text += line
        return text.strip()

# --- Reporter / Translation ---
def translate_analysis_to_korean(result):
    decision_map = {"buy": "매수", "sell": "매도", "hold": "보유"}
    confidence_map = {"high": "높음", "medium": "보통", "low": "낮음"}
    trend_map = {
        "bullish": "상승", "bearish": "하락", "neutral": "중립",
        "buy": "매수", "sell": "매도", "strong_buy": "강한 매수",
        "strong_sell": "강한 매도", "neutral_buy": "약한 매수",
        "neutral_sell": "약한 매도"
    }
    sentiment_map = {
        "extreme_fear": "극도의 공포", "fear": "공포", "neutral": "중립",
        "greed": "탐욕", "extreme_greed": "극도의 탐욕",
        "positive": "긍정적", "negative": "부정적"
    }
    confluence_map = {
        "technical and sentiment align": "기술적 지표와 심리 지표가 일치", 
        "technical and sentiment conflict": "기술적 지표와 심리 지표가 상충",
        "triple_align": "3개 지표 모두 일치", 
        "double_align": "2개 지표 일치", 
        "conflicting": "지표들이 상충"
    }

    # confidence 값이 float이나 숫자로 들어오는 경우 대비 → str 변환
    conf_raw = str(result.get('confidence', '')).lower().strip()

    translated = {
        "decision": decision_map.get(result.get('decision', ''), result.get('decision', '')),
        "confidence": confidence_map.get(conf_raw, conf_raw),
        "percentage": result.get('percentage', 100),
        "reason": result.get('reason', '')
    }

    if 'technical_summary' in result:
        ts = result['technical_summary'] or {}
        translated['technical_summary'] = {
            "daily_trend": trend_map.get(ts.get('daily_trend', ''), ts.get('daily_trend', '')),
            "hourly_momentum": trend_map.get(ts.get('hourly_momentum', ''), ts.get('hourly_momentum', '')),
            "fear_greed_signal": trend_map.get(ts.get('fear_greed_signal', ''), ts.get('fear_greed_signal', '')),
            "market_sentiment": sentiment_map.get(ts.get('market_sentiment', ''), ts.get('market_sentiment', '')),
            "news_sentiment": sentiment_map.get(ts.get('news_sentiment', ''), ts.get('news_sentiment', '')),
            "news_signal": trend_map.get(ts.get('news_signal', ''), ts.get('news_signal', '')),
            "confluence_analysis": confluence_map.get(ts.get('confluence_analysis', ''), ts.get('confluence_analysis', '')),
            "key_indicators": ts.get('key_indicators', []),
            "risk_factors": ts.get('risk_factors', [])
        }
    return translated

def translate_fear_greed_to_korean(fng):
    if not fng:
        return None
    classification_map = {
        "Extreme Fear": "극도의 공포",
        "Fear": "공포",
        "Neutral": "중립",
        "Greed": "탐욕",
        "Extreme Greed": "극도의 탐욕",
    }
    trend_map = {"improving": "개선", "declining": "악화", "stable": "안정"}
    signal_map = {
        "strong_buy": "강한 매수",
        "buy": "매수",
        "neutral_buy": "약한 매수",
        "neutral": "중립",
        "neutral_sell": "약한 매도",
        "sell": "매도",
        "strong_sell": "강한 매도",
    }
    return {
        "current_value": fng["current_value"],
        "classification": classification_map.get(fng["classification"], fng["classification"]),
        "7day_average": fng["7day_average"],
        "change_from_week": fng["change_from_week"],
        "trend": trend_map.get(fng["trend"], fng["trend"]),
        "market_signal": signal_map.get(fng["market_signal"], fng["market_signal"]),
    }

class Reporter:
    @staticmethod
    def print_portfolio(my_krw, my_btc, btc_avg_price, current_price):
        total = my_krw + my_btc * current_price
        profit_rate = 0
        if my_btc > 0 and btc_avg_price > 0:
            profit_rate = ((current_price - btc_avg_price) / btc_avg_price) * 100
        print(f"보유 현금: {my_krw:,.0f}원")
        print(f"보유 BTC: {my_btc:.8f} BTC")
        print(f"총 자산: {total:,.0f}원")
        if my_btc > 0:
            print(f"평균 매수가: {btc_avg_price:,.0f}원")
            print(f"수익률: {profit_rate:+.2f}%")
        return total, profit_rate

    @staticmethod
    def print_ai_result(result, fear_greed_data):
        tr = translate_analysis_to_korean(result)
        print(f"\n=== AI 분석 결과 ===")
        print(f"결정: {tr['decision'].upper()}")
        print(f"신뢰도: {str(tr['confidence']).upper()}")
        print(f"거래 비율: {tr.get('percentage', 100)}%")
        print(f"이유: {result.get('reason','')}")
        ts = tr.get("technical_summary") or {}
        if ts:
            print(f"\n=== 종합 분석 요약 ===")
            print(f"일봉 트렌드: {ts.get('daily_trend','N/A')}")
            print(f"시간봉 모멘텀: {ts.get('hourly_momentum','N/A')}")
            if ts.get("fear_greed_signal"):
                print(f"공포탐욕 신호: {ts['fear_greed_signal']}")
            if ts.get("market_sentiment"):
                print(f"시장 심리: {ts['market_sentiment']}")
            if ts.get("confluence_analysis"):
                print(f"신호 일치성: {ts['confluence_analysis']}")
            if ts.get("key_indicators"):
                print("주요 지표:")
                for k in ts["key_indicators"]:
                    print(f"  - {k}")
            if ts.get("risk_factors"):
                print("위험 요소:")
                for r in ts["risk_factors"]:
                    print(f"  - {r}")
        if fear_greed_data:
            fng_tr = translate_fear_greed_to_korean(fear_greed_data)
            print(f"\n=== 공포탐욕지수 상세 ===")
            print(f"현재 지수: {fng_tr['current_value']} ({fng_tr['classification']})")
            print(f"7일 평균: {fng_tr['7day_average']}")
            print(f"주간 변화: {fng_tr['change_from_week']:+d} ({fng_tr['trend']})")
            print(f"매매 신호: {fng_tr['market_signal']}")

# --- Orchestration ---
def ai_trading():
    print(f"\n=== AI 매매 분석 시작 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")
    db = DB()
    md = MarketData()
    ex = ExternalSignals()
    cap = ChartCapture()
    dai = DecisionAI()

    # Data
    df_daily, df_hourly = md.get_ohlcv()
    df_daily_i = md.add_indicators(df_daily.copy()) if df_daily is not None else None
    df_hourly_i = md.add_indicators(df_hourly.copy()) if df_hourly is not None else None
    daily_a = md.analyze_indicators(df_daily_i, "daily")
    hourly_a = md.analyze_indicators(df_hourly_i, "hourly")

    # External signals
    fear_greed = ex.fear_greed()
    news_data = ex.news()

    yt_ids_env = os.getenv("YT_VIDEO_IDS", "").strip()
    youtube_bundle = youtube_sent = None
    yt_excerpt = ""
    if yt_ids_env:
        yt_ids = [v.strip() for v in yt_ids_env.split(",") if v.strip()]
        if yt_ids:
            youtube_bundle = ex.youtube_bundle(yt_ids)
            youtube_sent = ex.youtube_sentiment(youtube_bundle.get("all_text", ""))
            all_text = youtube_bundle.get("all_text", "")
            yt_excerpt = all_text[:3000] if all_text else ""

    # Upbit / balances
    upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS_KEY"), os.getenv("UPBIT_SECRET_KEY"))
    my_krw, my_btc, balances, btc_avg_price, current_price, orderbook = md.balances_and_price(upbit)
    orderbook_summary = parse_orderbook_safely(orderbook, current_price, include_totals=True)

    total_asset, profit_rate = Reporter.print_portfolio(my_krw, my_btc, btc_avg_price, current_price)
    db.save_portfolio(my_krw, my_btc, btc_avg_price, current_price)

    # Chart
    chart_path, chart_b64 = None, None
    try:
        print("\n=== 차트 캡처(1H + 볼린저밴드) 시작 ===")
        chart_path, chart_b64 = cap.capture_chart_base64(UPBIT_CHART_URL)
        print(f"차트 캡처 완료: {chart_path}, base64 length={len(chart_b64)}")
    except Exception as e:
        print(f"차트 캡처 실패(이미지 없이 진행): {e}")

    # Build analysis payload
    analysis_data = {
        "current_status": {
            "krw_balance": my_krw,
            "btc_balance": my_btc,
            "btc_avg_buy_price": btc_avg_price,
            "current_btc_price": current_price,
            "total_asset": total_asset,
            "profit_rate": profit_rate,
        },
        "daily_ohlcv": df_daily.tail(10).to_dict("records") if df_daily is not None else [],
        "hourly_ohlcv": df_hourly.tail(12).to_dict("records") if df_hourly is not None else [],
        "daily_technical_analysis": daily_a,
        "hourly_technical_analysis": hourly_a,
        "fear_greed_index": fear_greed,
        "news_analysis": news_data,
        "youtube_transcripts": youtube_bundle,
        "youtube_sentiment": youtube_sent,
        "orderbook_summary": orderbook_summary,
        "technical_indicators_latest": {
            "daily": df_daily_i.iloc[-1].to_dict() if (df_daily_i is not None and len(df_daily_i) > 0) else {},
            "hourly": df_hourly_i.iloc[-1].to_dict() if (df_hourly_i is not None and len(df_hourly_i) > 0) else {},
        },
    }
    analysis_json = json.dumps(analysis_data, ensure_ascii=False, default=str)

    # --- 회고/반성: 최근 성과 평가 및 교훈 생성/로드 ---
    # 1) 새로 평가해서 저장하면서 텍스트 받기
    reflection_text_new = ReflectionEngine.run_and_get_lessons(db, horizon_hours=6, lookback=40, save=True)

    # 2) 최근 저장된 회고들에서 텍스트 합성(중복 제거/길이 제한)
    reflection_text_recent = db.get_recent_lessons_text(max_items=6, max_chars=800)

    # 우선순위: 방금 생성한 텍스트가 있으면 그걸, 없으면 누적 텍스트
    reflection_text = reflection_text_new or reflection_text_recent

    print("\n=== 회고/반성 규칙 ===")
    print(reflection_text if reflection_text else "(없음)")

    print("AI 분석 중...")
    user_content = dai.build_user_content(analysis_json, yt_excerpt, chart_b64)
    result = dai.request_decision(user_content, reflection_text=reflection_text)

    # 회고를 적용했으므로 카운트 올리기
    if reflection_text:
        db.mark_lessons_applied()

    # Save decision
    record_id = db.save_decision(
        result,
        {
            "current_price": current_price,
            "krw_balance": my_krw,
            "btc_balance": my_btc,
            "total_asset": total_asset,
            "profit_rate": profit_rate,
            "fear_greed_data": fear_greed,
        },
    )

    # Report
    Reporter.print_ai_result(result, fear_greed)

    # Trade
    trader = Trader(upbit, db)
    decision = (result.get("decision") or "hold").lower()
    percentage = float(result.get("percentage", 100))

    if decision == "buy":
        amount = Trader.calc_amount("buy", percentage, my_krw, my_btc)
        if Trader.validate_min("buy", amount, current_price):
            print(f"매수 금액: {amount:,.0f}원")
            trader.execute("buy", amount, record_id)
    elif decision == "sell":
        amount = Trader.calc_amount("sell", percentage, my_krw, my_btc)
        ask = parse_orderbook_safely(orderbook, current_price)["best_ask_price"]
        if Trader.validate_min("sell", amount, ask):
            print(f"매도 수량: {amount:.8f} BTC")
            print(f"예상 매도 금액: {amount * ask:,.0f}원")
            trader.execute("sell", amount, record_id)
    else:
        print("\n=== 보유 결정 ===\n현재 포지션 유지")

    # Stats
    db.print_stats()
    print("=" * 60)


if __name__ == "__main__":
    ai_trading()
    # 지속 실행 원하면 아래 주석 해제
    # while True:
    #     time.sleep(600)
    #     ai_trading()
