import os
import json
import math
import random
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

JST = ZoneInfo("Asia/Tokyo")
SHEET_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SECRET_ENV = "APP_SECRET_JSON"
BENCHMARK = "1306.T"
RISK_FREE_RATE = 0.021
TRADING_DAYS = 252
MIN_PRICE_DAYS = 180
PRICE_PERIOD = "400d"
USER_AGENTS = [
    # Chrome / Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Chrome / macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Firefox / Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    # Firefox / macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:125.0) Gecko/20100101 Firefox/125.0",
    # Safari / macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    # Safari / iPhone (iOS 17)
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1",
    # Edge / Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
]
FETCH_WORKERS = 5       # 並列スレッド数（レート制限に応じて調整）
CHECKPOINT_EVERY = 50    # 何件ごとに途中書き込みするか
PRICE_BATCH_SIZE = 200   # yf.download 一括取得の銘柄数上限

# 出力列定義（D列以降、固定）
# SYMBOL / NAME_YF / PRICE_DATE は処理内部で使うが人間には不要なため出力しない
OUTPUT_HEADERS = [
    "RUN_STATUS",        # D
    "SPEC_CHECK",        # E
    "DD_CHECK",          # F
    "DURATION_CHECK",    # G
    "Q5_CHECK",          # H
    "FINAL_CHECK",       # I
    "SPEC_SCORE",        # J
    "DD_SCORE",          # K
    "DURATION_SCORE",    # L
    "Q5_SCORE",          # M
    "FINAL_SCORE",       # N
    "BETA_252D",         # O
    "IVOL_252D",         # P
    "DD_RAW",            # Q
    "FCF_YIELD",         # R
    "CFO_YIELD",         # S
    "CAPEX_BURDEN",      # T
    "ASSET_GROWTH_1Y",   # U
    "ROE_TTM",           # V
    "DATA_COVERAGE",     # W
    "REASON_CODES",      # X
    "REASON_TEXT",       # Y
    "ERROR_MESSAGE",     # Z
    "UPDATED_AT_JST",    # AA
]

logger = logging.getLogger("screening")
logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass
class AppConfig:
    spreadsheet_url: str
    worksheet_name: str
    credentials_info: Dict[str, Any]


@dataclass
class PriceMetrics:
    beta_252d: Optional[float]
    ivol_252d: Optional[float]
    price_date: Optional[str]
    latest_price: Optional[float]
    price_days: int


@dataclass
class FinancialMetrics:
    market_cap: Optional[float]
    total_assets: Optional[float]
    prev_total_assets: Optional[float]
    total_debt: Optional[float]
    short_term_debt: Optional[float]
    long_term_debt: Optional[float]
    cash: Optional[float]
    operating_cf: Optional[float]
    free_cf: Optional[float]
    capex: Optional[float]
    equity: Optional[float]
    prev_equity: Optional[float]
    net_income: Optional[float]
    prev_net_income: Optional[float]
    quarterly_ocf: List[float]
    short_name: str


# ---------------------------------------------------------------------------
# 設定・認証
# ---------------------------------------------------------------------------

def parse_secret() -> AppConfig:
    raw = os.environ.get(SECRET_ENV, "")
    if not raw:
        raise RuntimeError(f"Missing env: {SECRET_ENV}")
    payload = json.loads(raw)
    return AppConfig(
        spreadsheet_url=payload["spreadsheet_url"],
        worksheet_name=payload["worksheet_name"],
        credentials_info=payload["gcp_service_account"],
    )


def get_sheets_service(credentials_info: Dict[str, Any]):
    creds = Credentials.from_service_account_info(credentials_info, scopes=SHEET_SCOPES)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def spreadsheet_id_from_url(url: str) -> str:
    marker = "/d/"
    if marker not in url:
        raise ValueError("Invalid spreadsheet_url")
    tail = url.split(marker, 1)[1]
    return tail.split("/", 1)[0]


def with_retry(fn, retries: int, retryable=(Exception,), label: str = "operation"):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except retryable as exc:
            last_exc = exc
            if attempt >= retries:
                break
            time.sleep(2 ** attempt)
    raise last_exc


# ---------------------------------------------------------------------------
# シート読み書き
# 仕様 1-1: ensure_headers() は呼ばない。1行目はユーザー管理。
# ---------------------------------------------------------------------------

def fetch_sheet_rows(service, spreadsheet_id: str, worksheet_name: str) -> List[List[str]]:
    """シート全体を読む。失敗時は SHEET_READ_FAILED をログして再 raise する。"""
    range_name = f"{worksheet_name}!A:AD"
    try:
        resp = with_retry(
            lambda: service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id, range=range_name
                valueRenderOption="UNFORMATTED_VALUE"  # ← この1行を追加
            ).execute(),
            retries=2,
            retryable=(HttpError, OSError),
            label="sheet_read",
        )
        return resp.get("values", [])
    except (HttpError, OSError):
        logger.info("FATAL stage=sheet_read error=SHEET_READ_FAILED")
        raise


def batch_write_output(
    service,
    spreadsheet_id: str,
    worksheet_name: str,
    row_outputs: List[Tuple[int, List[Any]]],
) -> None:
    """
    D列以降のみを、元の行番号を保持して書き込む。
    row_outputs: [(sheet_row_number, output_values), ...]
      sheet_row_number : シートの実際の行番号（1始まり）
      output_values    : OUTPUT_HEADERS 分（D〜AD）の値リスト
    A〜C列は一切触らない。失敗時は SHEET_WRITE_FAILED をログして再 raise する。
    """
    if not row_outputs:
        return
    data = []
    for sheet_row, values in row_outputs:
        range_str = f"{worksheet_name}!D{sheet_row}:AD{sheet_row}"
        data.append({"range": range_str, "values": [values]})
    body = {"valueInputOption": "RAW", "data": data}
    try:
        with_retry(
            lambda: service.spreadsheets().values().batchUpdate(
                spreadsheetId=spreadsheet_id, body=body
            ).execute(),
            retries=3,
            retryable=(HttpError, OSError),
            label="sheet_write",
        )
    except (HttpError, OSError):
        logger.info("FATAL stage=sheet_write error=SHEET_WRITE_FAILED")
        raise


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def normalize_code(raw: str) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) != 4:
        return None
    return digits


def yf_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": random.choice(USER_AGENTS)})
    return s


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and not x.strip():
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _squeeze_to_series(obj: Any) -> pd.Series:
    """DataFrame/Series どちらでも安全に 1-d Series へ変換する。
    yfinance 0.2.x+ の MultiIndex DataFrame 対応。"""
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj


def last_nonnull(df: Optional[pd.DataFrame], labels: List[str]) -> Optional[float]:
    if df is None or df.empty:
        return None
    lower = {str(idx).strip().lower(): idx for idx in df.index}
    for label in labels:
        key = label.strip().lower()
        if key in lower:
            series = pd.to_numeric(_squeeze_to_series(df.loc[lower[key]]), errors="coerce").dropna()
            if not series.empty:
                return safe_float(series.iloc[0])
    return None


def first_two_nonnull(df: Optional[pd.DataFrame], labels: List[str]) -> Tuple[Optional[float], Optional[float]]:
    if df is None or df.empty:
        return None, None
    lower = {str(idx).strip().lower(): idx for idx in df.index}
    for label in labels:
        key = label.strip().lower()
        if key in lower:
            series = pd.to_numeric(_squeeze_to_series(df.loc[lower[key]]), errors="coerce").dropna()
            if not series.empty:
                first = safe_float(series.iloc[0])
                second = safe_float(series.iloc[1]) if len(series) > 1 else None
                return first, second
    return None, None


def extract_quarterly_series(df: Optional[pd.DataFrame], labels: List[str], limit: int = 4) -> List[float]:
    if df is None or df.empty:
        return []
    lower = {str(idx).strip().lower(): idx for idx in df.index}
    for label in labels:
        key = label.strip().lower()
        if key in lower:
            series = pd.to_numeric(_squeeze_to_series(df.loc[lower[key]]), errors="coerce").dropna()
            return [float(v) for v in series.iloc[:limit].tolist()]
    return []


def format_num(v: Optional[float], digits: int = 4) -> Any:
    if v is None or math.isnan(v):
        return ""
    return round(v, digits)


# ---------------------------------------------------------------------------
# 価格・財務データ取得
# ---------------------------------------------------------------------------

def fetch_benchmark_returns(session: requests.Session) -> Tuple[pd.Series, str]:
    df = with_retry(
        lambda: yf.download(
            BENCHMARK,
            period=PRICE_PERIOD,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        ),
        retries=2,
        retryable=(Exception,),
        label="benchmark_fetch",
    )
    if df is None or df.empty:
        raise RuntimeError("Benchmark fetch failed")

    close_raw = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    close = pd.to_numeric(_squeeze_to_series(close_raw), errors="coerce").dropna()

    if len(close) < MIN_PRICE_DAYS:
        raise RuntimeError("Benchmark price history too short")
    price_date = close.index[-1].strftime("%Y-%m-%d")
    returns = np.log(close).diff().dropna().tail(TRADING_DAYS)
    return returns, price_date


def fetch_price_metrics(
    symbol: str,
    benchmark_returns: pd.Series,
    price_date: str,
    session: requests.Session,
    close_bulk: Optional[pd.Series] = None,
) -> PriceMetrics:
    """
    close_bulk: D案で事前一括取得した Close Series（symbol→値）。
    渡された場合はHTTPを呼ばずにそのまま使う。
    None の場合は従来通り個別取得（フォールバック）。
    """
    if close_bulk is not None:
        close_raw = close_bulk
    else:
        hist = with_retry(
            lambda: yf.Ticker(symbol).history(
                period=PRICE_PERIOD, interval="1d", auto_adjust=True, repair=True
            ),
            retries=2,
            retryable=(Exception,),
            label="price_fetch",
        )
        if hist is None or hist.empty or "Close" not in hist.columns:
            return PriceMetrics(None, None, None, None, 0)
        close_raw = hist["Close"]

    close = pd.to_numeric(_squeeze_to_series(close_raw), errors="coerce").dropna()
    close = close[close.index.strftime("%Y-%m-%d") <= price_date]
    if close.empty:
        return PriceMetrics(None, None, None, None, 0)

    returns = np.log(close).diff().dropna()
    joined = pd.concat(
        [returns.rename("stock"), benchmark_returns.rename("bench")],
        axis=1, join="inner",
    ).dropna()
    joined = joined.tail(TRADING_DAYS)
    latest_price = safe_float(close.iloc[-1])
    actual_price_date = close.index[-1].strftime("%Y-%m-%d")
    if len(joined) < MIN_PRICE_DAYS or actual_price_date != price_date:
        return PriceMetrics(None, None, actual_price_date, latest_price, len(joined))

    var_b = joined["bench"].var(ddof=1)
    beta = None if not var_b or math.isnan(var_b) else joined["stock"].cov(joined["bench"]) / var_b
    alpha_beta = np.polyfit(joined["bench"].values, joined["stock"].values, 1)
    residuals = joined["stock"].values - (alpha_beta[0] * joined["bench"].values + alpha_beta[1])
    ivol = float(np.std(residuals, ddof=1) * np.sqrt(TRADING_DAYS)) if len(residuals) > 1 else None
    return PriceMetrics(safe_float(beta), safe_float(ivol), actual_price_date, latest_price, len(joined))


def fetch_financial_metrics(
    symbol: str,
    latest_price: Optional[float],
    session: requests.Session,
) -> FinancialMetrics:
    ticker = yf.Ticker(symbol)

    fast: Dict[str, Any] = {}
    info: Dict[str, Any] = {}
    try:
        fast = dict(ticker.fast_info)
    except Exception:
        fast = {}
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    balance_sheet = ticker.balance_sheet
    cashflow = ticker.cashflow
    income_stmt = ticker.income_stmt
    quarterly_cashflow = ticker.quarterly_cashflow

    market_cap = safe_float(fast.get("marketCap")) or safe_float(info.get("marketCap"))
    shares_out = safe_float(info.get("sharesOutstanding")) or safe_float(fast.get("shares"))
    if market_cap is None and shares_out is not None and latest_price is not None:
        market_cap = shares_out * latest_price

    total_assets, prev_total_assets = first_two_nonnull(balance_sheet, ["Total Assets"])

    # 仕様 4-1: Net Debt は total_debt の候補から除外する
    total_debt = last_nonnull(balance_sheet, [
        "Total Debt",
        "Total Debt And Capital Lease Obligation",
    ])
    short_term_debt = last_nonnull(balance_sheet, [
        "Current Debt And Capital Lease Obligation",
        "Current Debt",
        "Short Long Term Debt",
        "Short Term Debt",
    ])
    long_term_debt = last_nonnull(balance_sheet, [
        "Long Term Debt And Capital Lease Obligation",
        "Long Term Debt",
    ])
    cash = last_nonnull(balance_sheet, [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash",
    ])
    equity, prev_equity = first_two_nonnull(balance_sheet, [
        "Stockholders Equity",
        "Total Equity Gross Minority Interest",
        "Common Stock Equity",
    ])

    operating_cf = last_nonnull(cashflow, [
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
        "Total Cash From Operating Activities",
    ])
    free_cf = last_nonnull(cashflow, ["Free Cash Flow"])
    capex = last_nonnull(cashflow, [
        "Capital Expenditure",
        "Capital Expenditures",
    ])
    if free_cf is None and operating_cf is not None and capex is not None:
        free_cf = operating_cf - abs(capex)

    net_income, prev_net_income = first_two_nonnull(income_stmt, [
        "Net Income",
        "Net Income Common Stockholders",
    ])

    quarterly_ocf = extract_quarterly_series(quarterly_cashflow, [
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
    ], limit=4)

    short_name = str(info.get("shortName", "") or "")[:100]

    return FinancialMetrics(
        market_cap=market_cap,
        total_assets=total_assets,
        prev_total_assets=prev_total_assets,
        total_debt=total_debt,
        short_term_debt=short_term_debt,
        long_term_debt=long_term_debt,
        cash=cash,
        operating_cf=operating_cf,
        free_cf=free_cf,
        capex=capex,
        equity=equity,
        prev_equity=prev_equity,
        net_income=net_income,
        prev_net_income=prev_net_income,
        quarterly_ocf=quarterly_ocf,
        short_name=short_name,
    )

def _col(raw: List[str], header: str) -> str:
    """シートの生行から OUTPUT_HEADERS の列を取り出す。D列=index3 に対応。"""
    idx = 3 + OUTPUT_HEADERS.index(header)
    return str(raw[idx]).strip() if len(raw) > idx else ""


def restore_cached_scores_from_sheet(raw: List[str]) -> Optional[Dict[str, Any]]:
    """
    E案: 前回シートのスコア・チェック・理由列をそのまま dict に復元する。
    価格系（BETA, IVOL, PRICE_DATE）は毎日更新するのでここでは復元しない。
    復元失敗（RUN_STATUS != OK、FINAL_CHECK が空）なら None を返す。
    """
    def _sf(header: str) -> Optional[float]:
        return safe_float(_col(raw, header))

    run_status = _col(raw, "RUN_STATUS")
    if run_status != "OK":
        return None

    cached: Dict[str, Any] = {
        "SPEC_CHECK":      _col(raw, "SPEC_CHECK"),
        "DD_CHECK":        _col(raw, "DD_CHECK"),
        "DURATION_CHECK":  _col(raw, "DURATION_CHECK"),
        "Q5_CHECK":        _col(raw, "Q5_CHECK"),
        "FINAL_CHECK":     _col(raw, "FINAL_CHECK"),
        "SPEC_SCORE":      _sf("SPEC_SCORE"),
        "DD_SCORE":        _sf("DD_SCORE"),
        "DURATION_SCORE":  _sf("DURATION_SCORE"),
        "Q5_SCORE":        _sf("Q5_SCORE"),
        "FINAL_SCORE":     _sf("FINAL_SCORE"),
        "DD_RAW":          _sf("DD_RAW"),
        "FCF_YIELD":       _sf("FCF_YIELD"),
        "CFO_YIELD":       _sf("CFO_YIELD"),
        "CAPEX_BURDEN":    _sf("CAPEX_BURDEN"),
        "ASSET_GROWTH_1Y": _sf("ASSET_GROWTH_1Y"),
        "ROE_TTM":         _sf("ROE_TTM"),
        "DATA_COVERAGE":   int(_sf("DATA_COVERAGE") or 0),
        "REASON_CODES":    _col(raw, "REASON_CODES"),
        "REASON_TEXT":     _col(raw, "REASON_TEXT"),
    }

    # FINAL_CHECK が空 → 前回結果なし → 復元失敗
    if not cached["FINAL_CHECK"]:
        return None

    return cached


# ---------------------------------------------------------------------------
# スコアリング
# ---------------------------------------------------------------------------

def pct_score(values: Dict[str, Optional[float]], invert: bool = False) -> Dict[str, Optional[float]]:
    clean = {k: v for k, v in values.items() if v is not None and not math.isnan(v)}
    if not clean:
        return {k: None for k in values}
    arr = np.array(list(clean.values()), dtype=float)
    low, high = np.nanpercentile(arr, 2.5), np.nanpercentile(arr, 97.5)
    clipped = {k: float(min(max(v, low), high)) for k, v in clean.items()}
    sorted_items = sorted(clipped.items(), key=lambda kv: kv[1])
    n = len(sorted_items)
    out: Dict[str, Optional[float]] = {k: None for k in values}
    if n == 1:
        only_key = sorted_items[0][0]
        out[only_key] = 100.0 if not invert else 0.0
        return out
    for rank, (k, _) in enumerate(sorted_items):
        pct = rank / (n - 1) * 100.0
        out[k] = 100.0 - pct if invert else pct
    return out


def weighted_mean(pairs: List[Tuple[Optional[float], float]]) -> Optional[float]:
    valid = [(v, w) for v, w in pairs if v is not None]
    if not valid:
        return None
    total_w = sum(w for _, w in valid)
    if total_w <= 0:
        return None
    return sum(v * w for v, w in valid) / total_w


def coverage_score(price_ok: bool, spec_ok: bool, dd_ok: bool, dur_ok: bool, q5_ok: bool) -> int:
    score = 0
    score += 40 if price_ok else 0
    score += 20 if spec_ok else 0
    score += 20 if dd_ok else 0
    score += 10 if dur_ok else 0
    score += 10 if q5_ok else 0
    return score


def reason_text(final_check: str, reason_codes: List[str], has_error: bool) -> str:
    """仕様 8-2 の優先順位に従う。"""
    if has_error:
        return "取得エラーで未判定"
    if final_check in {"AA", "A", "B", "C"}:
        return "全ゲート通過、総合評価上位"
    if final_check == "D":
        return "安全条件は通過、総合評価は中位"
    if "DD_LOW" in reason_codes or "DD_HIGH_VOL" in reason_codes:
        return "信用リスクが高く見送り"
    if any(c.startswith("SPEC_") for c in reason_codes):
        return "投機性が高く見送り"
    if any(c.startswith("DUR_") for c in reason_codes):
        return "現金回収力が弱く見送り"
    if any(c.startswith("Q5_") for c in reason_codes):
        return "成長品質が弱く見送り"
    return "データ不足で判定精度低下"


# ---------------------------------------------------------------------------
# 出力行ビルダー
# ---------------------------------------------------------------------------

def _make_empty_output() -> Dict[str, Any]:
    """OUTPUT_HEADERS キーをすべて空で持つ dict を返す。"""
    return {h: "" for h in OUTPUT_HEADERS}


def _make_error_output(error_message: str, reason_code: str) -> Dict[str, Any]:
    """エラー行の出力 dict を生成する。REASON_TEXT も必ずセットする。"""
    out = _make_empty_output()
    out["RUN_STATUS"] = "ERROR"
    out["UPDATED_AT_JST"] = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
    out["FINAL_CHECK"] = "ERROR"
    out["ERROR_MESSAGE"] = error_message
    out["REASON_CODES"] = reason_code
    out["REASON_TEXT"] = reason_text("ERROR", [reason_code], has_error=True)
    return out


def _render_output_row(result: Dict[str, Any]) -> List[Any]:
    """result dict → OUTPUT_HEADERS 順の値リスト（D〜AD列に対応）。"""
    row = []
    for h in OUTPUT_HEADERS:
        val = result.get(h, "")
        if h == "REASON_CODES" and isinstance(val, list):
            val = ",".join(val)
        elif isinstance(val, float):
            val = format_num(
                val,
                6 if h in {
                    "BETA_252D", "IVOL_252D", "DD_RAW",
                    "FCF_YIELD", "CFO_YIELD", "CAPEX_BURDEN",
                    "ASSET_GROWTH_1Y", "ROE_TTM",
                } else 2,
            )
        elif val is None:
            val = ""
        row.append(val)
    return row


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def main() -> None:
    started = datetime.now(JST).isoformat(timespec="seconds")
    cfg = parse_secret()
    service = get_sheets_service(cfg.credentials_info)
    spreadsheet_id = spreadsheet_id_from_url(cfg.spreadsheet_url)

    # E案: JST 日曜日のみ財務データを更新する。それ以外の日は前回シート値を流用。
    now_jst = datetime.now(JST)
    is_fin_update_day = (now_jst.weekday() == 6)  # 6 = 日曜
    logger.info(f"MODE fin_update={is_fin_update_day}")

    # 仕様 1-1: ensure_headers() は呼ばない
    # 仕様 7-1: fetch_sheet_rows 内で read 失敗を SHEET_READ_FAILED として出す
    rows_raw = fetch_sheet_rows(service, spreadsheet_id, cfg.worksheet_name)

    if not rows_raw:
        raise RuntimeError("Sheet is empty")

    # 仕様 1-2: 2行目以降のみ処理対象
    data_rows = rows_raw[1:] if len(rows_raw) > 1 else []

    # ------------------------------------------------------------------
    # 仕様 1-4: 行を詰めない。シートの行番号（1始まり）を保持する。
    # row_meta[sheet_row] = {"code": str|None, "skip": bool, "raw": list}
    # sheet_row = index_in_data_rows + 2
    # ------------------------------------------------------------------
    row_meta: Dict[int, Dict[str, Any]] = {}
    for i, raw in enumerate(data_rows):
        sheet_row = i + 2
        col_a = str(raw[0]).strip() if len(raw) >= 1 else ""
        col_b = str(raw[1]).strip() if len(raw) >= 2 else ""
        col_c = str(raw[2]).strip() if len(raw) >= 3 else ""

        # 仕様 6-2: A〜C がすべて空 → 完全空行、書き込みしない
        if not col_a and not col_b and not col_c:
            row_meta[sheet_row] = {"code": None, "skip": True, "raw": raw}
            continue

        # 仕様 6-2: A が空で B,C に文字 → メモ行、書き込みしない
        if not col_a and (col_b or col_c):
            row_meta[sheet_row] = {"code": None, "skip": True, "raw": raw}
            continue

        code = normalize_code(col_a)
        # E案: 財務更新日でない場合は前回シート値を復元してキャッシュとして持つ
        cached_scores = None
        if not is_fin_update_day and code is not None:
            cached_scores = restore_cached_scores_from_sheet(raw)
        row_meta[sheet_row] = {"code": code, "skip": False, "raw": raw, "cached_scores": cached_scores}

    # code → 対応するシート行番号のリスト（重複コード対応）
    # E案: 同一コードの最初の行の cached_scores を代表値として使う
    code_to_rows: Dict[str, List[int]] = {}
    code_cached_scores: Dict[str, Optional[Dict[str, Any]]] = {}
    for sheet_row, meta in row_meta.items():
        if meta.get("skip") or meta["code"] is None:
            continue
        c = meta["code"]
        code_to_rows.setdefault(c, []).append(sheet_row)
        if c not in code_cached_scores:
            code_cached_scores[c] = meta.get("cached_scores")

    # ベンチマーク取得（失敗時は上位の例外ハンドラへ）
    session = yf_session()
    benchmark_returns, benchmark_price_date = fetch_benchmark_returns(session)

    logger.info(f"START ts={started} rows={len(data_rows)}")
    logger.info(f"MARKET benchmark={BENCHMARK} price_date={benchmark_price_date}")
    logger.info(f"READ unique_codes={len(code_to_rows)}")

    # ------------------------------------------------------------------
    # D: yf.download で価格を一括取得し close_cache を構築する
    # PRICE_BATCH_SIZE 件ずつ分割してリクエストする。
    # 失敗したバッチはスキップし、該当銘柄は _fetch_one 内で個別取得にフォールバック。
    # FETCH_WORKERS を下げても価格取得効率は落ちない（キャッシュ参照のみ）。
    # ------------------------------------------------------------------
    all_symbols = [f"{code}.T" for code in code_to_rows.keys()]
    close_cache: Dict[str, pd.Series] = {}  # symbol → Close Series

    for batch_start in range(0, len(all_symbols), PRICE_BATCH_SIZE):
        batch = all_symbols[batch_start: batch_start + PRICE_BATCH_SIZE]
        try:
            bulk_df = with_retry(
                lambda b=batch: yf.download(
                    b,
                    period=PRICE_PERIOD,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                ),
                retries=2,
                retryable=(Exception,),
                label="price_bulk_fetch",
            )
            if bulk_df is None or bulk_df.empty:
                continue

            # Close 列を取り出す。単一銘柄でも複数銘柄でも同じ処理になるよう正規化する。
            if "Close" in bulk_df.columns:
                close_df = bulk_df["Close"]
            else:
                close_df = bulk_df.iloc[:, :len(batch)]

            # close_df が Series（銘柄1件のとき）→ DataFrame に統一する
            if isinstance(close_df, pd.Series):
                close_df = close_df.to_frame(name=batch[0])

            # MultiIndex の場合は1階層目（"Close"）を落として銘柄名だけにする
            if isinstance(close_df.columns, pd.MultiIndex):
                close_df.columns = close_df.columns.droplevel(0)

            for sym in batch:
                if sym in close_df.columns:
                    s = close_df[sym].dropna()
                    if not s.empty:
                        close_cache[sym] = s

        except Exception:
            # バッチ失敗は致命的にしない。該当銘柄は個別取得へフォールバック。
            logger.info(
                f"PRICE_BULK batch_start={batch_start} size={len(batch)} failed, fallback to individual"
            )

    logger.info(f"PRICE_BULK cached={len(close_cache)} / {len(all_symbols)}")

    # ------------------------------------------------------------------
    # 仕様 2-1: 同一コードは 1 回だけ計算する
    # A: ThreadPoolExecutor で並列取得
    # C: CHECKPOINT_EVERY 件ごとに途中書き込み
    # ------------------------------------------------------------------
    results: Dict[str, Dict[str, Any]] = {}
    codes = list(code_to_rows.keys())

    def _fetch_one(code: str) -> Dict[str, Any]:
        """1銘柄分の取得処理。スレッドから呼ばれる。"""
        symbol = f"{code}.T"
        result: Dict[str, Any] = {
            "SYMBOL": symbol,
            "RUN_STATUS": "OK",
            "UPDATED_AT_JST": datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S"),
            "PRICE_DATE": "",
            "SPEC_CHECK": "NA",
            "DD_CHECK": "NA",
            "DURATION_CHECK": "NA",
            "Q5_CHECK": "NA",
            "FINAL_CHECK": "FAIL",
            "SPEC_SCORE": None,
            "DD_SCORE": None,
            "DURATION_SCORE": None,
            "Q5_SCORE": None,
            "FINAL_SCORE": None,
            "BETA_252D": None,
            "IVOL_252D": None,
            "DD_RAW": None,
            "FCF_YIELD": None,
            "CFO_YIELD": None,
            "CAPEX_BURDEN": None,
            "ASSET_GROWTH_1Y": None,
            "ROE_TTM": None,
            "DATA_COVERAGE": 0,
            "REASON_CODES": [],
            "REASON_TEXT": "",
            "ERROR_MESSAGE": "",
        }
        try:
            try:
                pm = fetch_price_metrics(
                    symbol,
                    benchmark_returns,
                    benchmark_price_date,
                    session,
                    close_bulk=close_cache.get(symbol),  # D: キャッシュあれば使う、なければ個別取得
                )
            except Exception:
                result["RUN_STATUS"] = "ERROR"
                result["FINAL_CHECK"] = "ERROR"
                result["ERROR_MESSAGE"] = "PRICE_FETCH_FAILED"
                result["REASON_CODES"] = ["DATA_FETCH_ERROR"]
                result["REASON_TEXT"] = reason_text("ERROR", ["DATA_FETCH_ERROR"], has_error=True)
                return result

            # E案: 財務更新日でない かつ 前回スコアが復元できる場合は財務取得をスキップする。
            # 価格系（BETA, IVOL, PRICE_DATE）は毎日新しい値で上書きする。
            cached = code_cached_scores.get(code)
            if not is_fin_update_day and cached is not None:
                result["PRICE_DATE"]    = pm.price_date or ""
                result["BETA_252D"]     = pm.beta_252d
                result["IVOL_252D"]     = pm.ivol_252d
                result["SPEC_CHECK"]    = cached["SPEC_CHECK"]
                result["DD_CHECK"]      = cached["DD_CHECK"]
                result["DURATION_CHECK"]= cached["DURATION_CHECK"]
                result["Q5_CHECK"]      = cached["Q5_CHECK"]
                result["FINAL_CHECK"]   = cached["FINAL_CHECK"]
                result["SPEC_SCORE"]    = cached["SPEC_SCORE"]
                result["DD_SCORE"]      = cached["DD_SCORE"]
                result["DURATION_SCORE"]= cached["DURATION_SCORE"]
                result["Q5_SCORE"]      = cached["Q5_SCORE"]
                result["FINAL_SCORE"]   = cached["FINAL_SCORE"]
                result["DD_RAW"]        = cached["DD_RAW"]
                result["FCF_YIELD"]     = cached["FCF_YIELD"]
                result["CFO_YIELD"]     = cached["CFO_YIELD"]
                result["CAPEX_BURDEN"]  = cached["CAPEX_BURDEN"]
                result["ASSET_GROWTH_1Y"]= cached["ASSET_GROWTH_1Y"]
                result["ROE_TTM"]       = cached["ROE_TTM"]
                result["DATA_COVERAGE"] = cached["DATA_COVERAGE"]
                result["REASON_CODES"]  = cached["REASON_CODES"]
                result["REASON_TEXT"]   = cached["REASON_TEXT"]
                # スコア再計算不要フラグ（後段の pct_score ループでスキップするために使う）
                result["_use_cached_scores"] = True
                return result

            # 財務更新日 or 前回スコアなし → 通常通り財務取得
            try:
                fm = fetch_financial_metrics(symbol, pm.latest_price, session)
            except Exception:
                result["RUN_STATUS"] = "ERROR"
                result["FINAL_CHECK"] = "ERROR"
                result["ERROR_MESSAGE"] = "FIN_FETCH_FAILED"
                result["REASON_CODES"] = ["DATA_FETCH_ERROR"]
                result["REASON_TEXT"] = reason_text("ERROR", ["DATA_FETCH_ERROR"], has_error=True)
                return result

            result["PRICE_DATE"] = pm.price_date or ""
            result["BETA_252D"] = pm.beta_252d
            result["IVOL_252D"] = pm.ivol_252d

            price_ok = pm.price_days >= MIN_PRICE_DAYS and pm.price_date == benchmark_price_date

            # 仕様 3-2: debt を spec_ok_inputs の必須条件から外す
            spec_ok_inputs = all(v is not None for v in [
                pm.beta_252d, pm.ivol_252d, fm.total_assets, fm.operating_cf, fm.cash,
            ])
            dd_ok_inputs = (
                all(v is not None for v in [fm.market_cap, fm.total_debt, pm.ivol_252d])
                and price_ok
            )
            dur_ok_inputs = (
                all(v is not None for v in [fm.market_cap, fm.operating_cf])
                and (fm.free_cf is not None or fm.capex is not None)
            )
            q5_ok_inputs = all(v is not None for v in [
                fm.total_assets, fm.prev_total_assets,
                fm.equity, fm.prev_equity,
                fm.net_income, fm.prev_net_income,
                fm.operating_cf,
            ])

            result["DATA_COVERAGE"] = coverage_score(
                price_ok, spec_ok_inputs, dd_ok_inputs, dur_ok_inputs, q5_ok_inputs
            )
            if not price_ok:
                result["REASON_CODES"].append("DATA_PRICE_SHORT")
            if fm.total_debt is None:
                result["REASON_CODES"].append("DATA_DEBT_MISSING")
            if fm.market_cap is None:
                result["REASON_CODES"].append("DATA_MC_MISSING")
            if not (dur_ok_inputs and q5_ok_inputs):
                result["REASON_CODES"].append("PARTIAL_MODEL_DATA")

            result["_spec_inputs_ok"] = spec_ok_inputs
            result["_dd_inputs_ok"] = dd_ok_inputs
            result["_dur_inputs_ok"] = dur_ok_inputs
            result["_q5_inputs_ok"] = q5_ok_inputs
            result["_market_cap"] = fm.market_cap
            result["_total_debt"] = fm.total_debt
            result["_short_term_debt"] = fm.short_term_debt
            result["_long_term_debt"] = fm.long_term_debt
            result["_cash"] = fm.cash
            result["_operating_cf"] = fm.operating_cf
            result["_free_cf"] = fm.free_cf
            result["_capex"] = fm.capex
            result["_total_assets"] = fm.total_assets
            result["_prev_total_assets"] = fm.prev_total_assets
            result["_equity"] = fm.equity
            result["_prev_equity"] = fm.prev_equity
            result["_net_income"] = fm.net_income
            result["_prev_net_income"] = fm.prev_net_income
            result["_quarterly_ocf"] = fm.quarterly_ocf

        except Exception:
            result["RUN_STATUS"] = "ERROR"
            result["FINAL_CHECK"] = "ERROR"
            result["ERROR_MESSAGE"] = "UNKNOWN_ERROR"
            result["REASON_CODES"] = ["DATA_FETCH_ERROR"]
            result["REASON_TEXT"] = reason_text("ERROR", ["DATA_FETCH_ERROR"], has_error=True)

        return result

    def _flush_checkpoint(done_codes: List[str]) -> None:
        """C: 処理済みコードをシートに途中書き込みする。"""
        partial_outputs: List[Tuple[int, List[Any]]] = []
        for sheet_row, meta in row_meta.items():
            if meta.get("skip"):
                partial_outputs.append((sheet_row, [""] * len(OUTPUT_HEADERS)))
                continue
            c = meta["code"]
            if c is None or c not in done_codes:
                continue
            r = results[c]
            partial_outputs.append((sheet_row, _render_output_row(r)))
        if partial_outputs:
            try:
                batch_write_output(service, spreadsheet_id, cfg.worksheet_name, partial_outputs)
                logger.info(f"CHECKPOINT written={len(partial_outputs)}")
            except Exception:
                # チェックポイント失敗は致命的にしない（続行）
                logger.info("CHECKPOINT write failed, continuing")

    completed_codes: List[str] = []
    last_checkpoint_count = 0

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
        future_to_code = {executor.submit(_fetch_one, code): code for code in codes}
        for idx, future in enumerate(as_completed(future_to_code), start=1):
            code = future_to_code[future]
            results[code] = future.result()
            completed_codes.append(code)

            if idx % 500 == 0 or idx == len(codes):
                ok_n = sum(1 for r in results.values() if r["RUN_STATUS"] == "OK")
                err_n = sum(1 for r in results.values() if r["RUN_STATUS"] == "ERROR")
                logger.info(f"PROGRESS done={idx} ok={ok_n} error={err_n}")

            # C: CHECKPOINT_EVERY 件ごとに途中書き込み
            if idx - last_checkpoint_count >= CHECKPOINT_EVERY:
                _flush_checkpoint(completed_codes)
                last_checkpoint_count = idx

    # ------------------------------------------------------------------
    # パーセンタイルスコア計算（RUN_STATUS=OK のみ）
    # ------------------------------------------------------------------
    spec_badness: Dict[str, Optional[float]] = {}
    dd_raws: Dict[str, Optional[float]] = {}
    dur_fcf_yield: Dict[str, Optional[float]] = {}
    dur_cfo_yield: Dict[str, Optional[float]] = {}
    dur_capex_burden: Dict[str, Optional[float]] = {}
    dur_cf_stability: Dict[str, Optional[float]] = {}
    q5_asset_growth: Dict[str, Optional[float]] = {}
    q5_roe_ttm: Dict[str, Optional[float]] = {}
    q5_delta_roe: Dict[str, Optional[float]] = {}
    q5_ocf_assets: Dict[str, Optional[float]] = {}

    for code, r in results.items():
        if r["RUN_STATUS"] != "OK":
            continue
        # E案: 財務キャッシュ流用の銘柄はスコア再計算をスキップする
        if r.get("_use_cached_scores"):
            continue

        # --- SPEC ---
        if r["_spec_inputs_ok"]:
            debt = r["_total_debt"]
            assets = r["_total_assets"]
            ocf = r["_operating_cf"]
            cash = r["_cash"]

            # 仕様 3-1: debt <= 0 または None のとき DISTRESS_PROXY = 0.0
            if debt is None or debt <= 0:
                distress = 0.0
            elif assets and assets > 0 and ocf is not None and cash is not None:
                distress = (
                    0.4 * (debt / assets)
                    + 0.3 * (-(ocf / debt))
                    + 0.3 * (-(cash / debt))
                )
            else:
                distress = None

            beta_off = abs((r["BETA_252D"] or 1.0) - 1.0) if r["BETA_252D"] is not None else None
            if distress is not None and beta_off is not None and r["IVOL_252D"] is not None:
                spec_badness[code] = 0.5 * r["IVOL_252D"] + 0.2 * beta_off + 0.3 * distress
            else:
                spec_badness[code] = None
        else:
            spec_badness[code] = None

        # --- DD ---
        if r["_dd_inputs_ok"]:
            debt = r["_total_debt"] or 0.0
            mc = r["_market_cap"] or 0.0
            if debt <= 0:
                dd_raws[code] = 999.0
            else:
                std = r["_short_term_debt"]
                ltd = r["_long_term_debt"]
                if std is not None and ltd is not None:
                    dp_candidate = std + 0.5 * ltd
                    default_point = dp_candidate if dp_candidate > 0 else 0.75 * debt
                else:
                    default_point = 0.75 * debt
                sigma_e = r["IVOL_252D"]
                va = mc + debt
                sigma_a = None if va <= 0 or sigma_e is None else sigma_e * mc / va
                # 仕様 4-2: default_point <= 0 の場合は DD 計算しない
                if default_point > 0 and sigma_a and sigma_a > 0 and va > 0:
                    dd = (
                        math.log(va / default_point)
                        + (RISK_FREE_RATE + 0.5 * sigma_a * sigma_a)
                    ) / sigma_a
                    dd_raws[code] = dd
                else:
                    dd_raws[code] = None
        else:
            dd_raws[code] = None

        # --- Duration ---
        if r["_dur_inputs_ok"]:
            mc = r["_market_cap"]
            ocf = r["_operating_cf"]
            fcf = r["_free_cf"]
            capex = r["_capex"]
            if mc and mc > 0:
                dur_fcf_yield[code] = None if fcf is None else fcf / mc
                dur_cfo_yield[code] = None if ocf is None else ocf / mc
            else:
                dur_fcf_yield[code] = None
                dur_cfo_yield[code] = None
            if ocf is None or abs(ocf) < 1e-9 or capex is None:
                dur_capex_burden[code] = None
            else:
                dur_capex_burden[code] = abs(capex) / max(abs(ocf), 1e-9)
            qocf = r["_quarterly_ocf"]
            if len(qocf) >= 3:
                arr = np.array(qocf, dtype=float)
                mean_abs = max(abs(arr.mean()), 1e-9)
                dur_cf_stability[code] = -float(arr.std(ddof=1) / mean_abs)
            else:
                dur_cf_stability[code] = None
        else:
            dur_fcf_yield[code] = None
            dur_cfo_yield[code] = None
            dur_capex_burden[code] = None
            dur_cf_stability[code] = None

        # --- Q5 ---
        if r["_q5_inputs_ok"]:
            assets_t = r["_total_assets"]
            assets_p = r["_prev_total_assets"]
            eq_t = r["_equity"]
            eq_p = r["_prev_equity"]
            ni_t = r["_net_income"]
            ni_p = r["_prev_net_income"]
            ocf = r["_operating_cf"]
            q5_asset_growth[code] = (
                None if assets_t in (None, 0) or assets_p in (None, 0)
                else assets_t / assets_p - 1.0
            )
            avg_eq_t = None if eq_t is None or eq_p is None else (eq_t + eq_p) / 2.0
            q5_roe_ttm[code] = None if avg_eq_t in (None, 0) else ni_t / avg_eq_t
            # 前期 ROE: 2期前データが取れないため eq_p 単期ベース
            prev_roe = None if eq_p in (None, 0) else ni_p / eq_p
            q5_delta_roe[code] = (
                None if q5_roe_ttm[code] is None or prev_roe is None
                else q5_roe_ttm[code] - prev_roe
            )
            q5_ocf_assets[code] = None if assets_t in (None, 0) else ocf / assets_t
        else:
            q5_asset_growth[code] = None
            q5_roe_ttm[code] = None
            q5_delta_roe[code] = None
            q5_ocf_assets[code] = None

    spec_scores = pct_score(spec_badness, invert=True)
    dd_scores = pct_score(dd_raws, invert=False)
    fcf_scores = pct_score(dur_fcf_yield)
    cfo_scores = pct_score(dur_cfo_yield)
    capex_scores = pct_score(dur_capex_burden, invert=True)
    stability_scores = pct_score(dur_cf_stability)
    asset_growth_scores = pct_score(q5_asset_growth, invert=True)
    roe_scores = pct_score(q5_roe_ttm)
    delta_roe_scores = pct_score(q5_delta_roe)
    ocf_asset_scores = pct_score(q5_ocf_assets)

    reason_counter: Dict[str, int] = {}

    for code, r in results.items():
        if r["RUN_STATUS"] != "OK":
            continue
        # E案: 財務キャッシュ流用の銘柄はスコア再計算をスキップする
        if r.get("_use_cached_scores"):
            continue

        # SPEC
        ss = spec_scores.get(code)
        r["SPEC_SCORE"] = ss
        if ss is None:
            r["SPEC_CHECK"] = "NA"
        else:
            r["SPEC_CHECK"] = "PASS" if ss >= 35 else "FAIL"
            if r["SPEC_CHECK"] == "FAIL":
                if r["IVOL_252D"] is not None:
                    r["REASON_CODES"].append("SPEC_HIGH_IVOL")
                if r["BETA_252D"] is not None:
                    r["REASON_CODES"].append("SPEC_BETA_OFFSIDE")
                r["REASON_CODES"].append("SPEC_HIGH_DISTRESS")

        # DD
        dd_raw = dd_raws.get(code)
        ds = dd_scores.get(code)
        r["DD_RAW"] = dd_raw
        r["DD_SCORE"] = ds
        if dd_raw == 999.0:
            r["DD_CHECK"] = "PASS"
            r["DD_SCORE"] = 100.0
            r["DD_RAW"] = None
            r["REASON_CODES"].append("NO_DEBT")
        elif dd_raw is None or ds is None:
            r["DD_CHECK"] = "NA"
        else:
            r["DD_CHECK"] = "PASS" if (dd_raw >= 0.5 and ds >= 25) else "FAIL"
            if r["DD_CHECK"] == "FAIL":
                r["REASON_CODES"].append("DD_LOW")
                if r["IVOL_252D"] is not None:
                    r["REASON_CODES"].append("DD_HIGH_VOL")

        # Duration
        r["FCF_YIELD"] = dur_fcf_yield.get(code)
        r["CFO_YIELD"] = dur_cfo_yield.get(code)
        r["CAPEX_BURDEN"] = dur_capex_burden.get(code)
        dur_score = weighted_mean([
            (fcf_scores.get(code), 0.4 if stability_scores.get(code) is not None else 0.5),
            (cfo_scores.get(code), 0.3),
            (capex_scores.get(code), 0.2),
            (stability_scores.get(code), 0.1),
        ])
        r["DURATION_SCORE"] = dur_score
        if dur_score is None:
            r["DURATION_CHECK"] = "NA"
        else:
            r["DURATION_CHECK"] = "PASS" if dur_score >= 50 else "FAIL"
            if r["DURATION_CHECK"] == "FAIL":
                if (dur_fcf_yield.get(code) or -1) < 0:
                    r["REASON_CODES"].append("DUR_LOW_FCF")
                if (dur_cfo_yield.get(code) or -1) < 0:
                    r["REASON_CODES"].append("DUR_LOW_CFO")
                if (dur_capex_burden.get(code) or 0) > 1:
                    r["REASON_CODES"].append("DUR_HIGH_CAPEX")

        # Q5
        r["ASSET_GROWTH_1Y"] = q5_asset_growth.get(code)
        r["ROE_TTM"] = q5_roe_ttm.get(code)
        q5_score = weighted_mean([
            (asset_growth_scores.get(code), 0.35),
            (ocf_asset_scores.get(code), 0.25),
            (roe_scores.get(code), 0.25),
            (delta_roe_scores.get(code), 0.15),
        ])
        r["Q5_SCORE"] = q5_score
        if q5_score is None:
            r["Q5_CHECK"] = "NA"
        else:
            r["Q5_CHECK"] = "PASS" if q5_score >= 50 else "FAIL"
            if r["Q5_CHECK"] == "FAIL":
                if (q5_asset_growth.get(code) or 0) > 0.15:
                    r["REASON_CODES"].append("Q5_HIGH_ASSET_GROWTH")
                if (q5_roe_ttm.get(code) or 0) < 0.05:
                    r["REASON_CODES"].append("Q5_LOW_ROE")
                if (q5_delta_roe.get(code) or 0) < 0:
                    r["REASON_CODES"].append("Q5_NEG_DROE")

        # Final
        if r["SPEC_CHECK"] != "PASS" or r["DD_CHECK"] != "PASS":
            r["FINAL_CHECK"] = "FAIL"
            r["FINAL_SCORE"] = None
        else:
            final_score = weighted_mean([
                (r["DURATION_SCORE"], 0.6),
                (r["Q5_SCORE"], 0.4),
            ])
            r["FINAL_SCORE"] = final_score
            if final_score is None:
                r["FINAL_CHECK"] = "FAIL"
            elif final_score >= 95:
                r["FINAL_CHECK"] = "AA"
            elif final_score >= 90:
                r["FINAL_CHECK"] = "A"
            elif final_score >= 80:
                r["FINAL_CHECK"] = "B"
            elif final_score >= 70:
                r["FINAL_CHECK"] = "C"
            elif final_score >= 50:
                r["FINAL_CHECK"] = "D"
            else:
                r["FINAL_CHECK"] = "FAIL"

        # 重複除去・理由文
        dedup: List[str] = []
        for c in r["REASON_CODES"]:
            if c not in dedup:
                dedup.append(c)
        r["REASON_CODES"] = dedup
        for c in dedup:
            reason_counter[c] = reason_counter.get(c, 0) + 1
        r["REASON_TEXT"] = reason_text(r["FINAL_CHECK"], dedup, False)

    # ------------------------------------------------------------------
    # シート書き込み: D列以降のみ、行番号を保持して batchUpdate
    # ------------------------------------------------------------------
    row_outputs: List[Tuple[int, List[Any]]] = []

    for sheet_row, meta in row_meta.items():
        # 完全空行・メモ行は D〜AD を空欄で上書きする（古い結果を残さない）
        if meta.get("skip"):
            row_outputs.append((sheet_row, [""] * len(OUTPUT_HEADERS)))
            continue

        code = meta["code"]
        if code is None:
            # 仕様 1-5: 無効コード行 → PARSE_FAILED
            out = _make_error_output("PARSE_FAILED", "PARSE_ERROR")
            row_outputs.append((sheet_row, _render_output_row(out)))
            continue

        # 仕様 2-1: 同一コードの複数行に同じ結果を書く
        r = results[code]
        row_outputs.append((sheet_row, _render_output_row(r)))

    # 仕様 7-1: write 失敗は batch_write_output 内で SHEET_WRITE_FAILED として出す
    batch_write_output(service, spreadsheet_id, cfg.worksheet_name, row_outputs)
    logger.info(f"WRITE updated_rows={len(row_outputs)}")

    top_reasons = sorted(reason_counter.items(), key=lambda kv: kv[1], reverse=True)[:3]
    if top_reasons:
        logger.info("TOP_REASONS " + " ".join(f"{k}={v}" for k, v in top_reasons))
    else:
        logger.info("TOP_REASONS none=0")

    grade_counter: Dict[str, int] = {"AA": 0, "A": 0, "B": 0, "C": 0, "D": 0, "FAIL": 0, "error": 0, "skipped": 0}
    for sheet_row, meta in row_meta.items():
        if meta.get("skip"):
            grade_counter["skipped"] += 1
            continue
        code = meta["code"]
        if code is None:
            grade_counter["error"] += 1
            continue
        fc = results[code]["FINAL_CHECK"]
        rs = results[code]["RUN_STATUS"]
        if rs == "ERROR":
            grade_counter["error"] += 1
        elif fc in grade_counter:
            grade_counter[fc] += 1
        else:
            grade_counter["FAIL"] += 1
    logger.info(
        f"END AA={grade_counter['AA']} A={grade_counter['A']} B={grade_counter['B']} "
        f"C={grade_counter['C']} D={grade_counter['D']} FAIL={grade_counter['FAIL']} "
        f"error={grade_counter['error']} skipped={grade_counter['skipped']}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        msg = str(exc)
        if "Benchmark" in msg:
            logger.info("FATAL stage=benchmark error=PRICE_FETCH_FAILED")
        else:
            logger.info("FATAL stage=runtime error=UNKNOWN_ERROR")
        raise
