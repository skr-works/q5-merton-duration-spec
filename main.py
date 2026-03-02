import os
import json
import math
import time
import logging
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
RISK_FREE_RATE = 0.005
TRADING_DAYS = 252
MIN_PRICE_DAYS = 180
PRICE_PERIOD = "400d"
USER_AGENT = "Mozilla/5.0 (compatible; screening-bot/1.0)"

# Input/output columns
INPUT_COLS = ["CODE", "COMPANY_NAME", "SECTOR"]  # A-C
OUTPUT_HEADERS = [
    "SYMBOL",            # D
    "NAME_YF",           # E
    "RUN_STATUS",        # F
    "UPDATED_AT_JST",    # G
    "PRICE_DATE",        # H
    "SPEC_CHECK",        # I
    "DD_CHECK",          # J
    "DURATION_CHECK",    # K
    "Q5_CHECK",          # L
    "FINAL_CHECK",       # M
    "SPEC_SCORE",        # N
    "DD_SCORE",          # O
    "DURATION_SCORE",    # P
    "Q5_SCORE",          # Q
    "FINAL_SCORE",       # R
    "BETA_252D",         # S
    "IVOL_252D",         # T
    "DD_RAW",            # U
    "FCF_YIELD",         # V
    "CFO_YIELD",         # W
    "CAPEX_BURDEN",      # X
    "ASSET_GROWTH_1Y",   # Y
    "ROE_TTM",           # Z
    "DATA_COVERAGE",     # AA
    "REASON_CODES",      # AB
    "REASON_TEXT",       # AC
    "ERROR_MESSAGE",     # AD
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


def fetch_sheet_rows(service, spreadsheet_id: str, worksheet_name: str) -> List[List[str]]:
    range_name = f"{worksheet_name}!A:AD"
    resp = with_retry(
        lambda: service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id, range=range_name
        ).execute(),
        retries=2,
        retryable=(HttpError,),
        label="sheet_read",
    )
    return resp.get("values", [])


def batch_write_rows(service, spreadsheet_id: str, worksheet_name: str, rows: List[List[Any]]):
    body = {"values": rows}
    with_retry(
        lambda: service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=f"{worksheet_name}!A2:AD",
            valueInputOption="RAW",
            body=body,
        ).execute(),
        retries=3,
        retryable=(HttpError,),
        label="sheet_write",
    )


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
    s.headers.update({"User-Agent": USER_AGENT})
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


def last_nonnull(df: Optional[pd.DataFrame], labels: List[str]) -> Optional[float]:
    if df is None or df.empty:
        return None
    lower = {str(idx).strip().lower(): idx for idx in df.index}
    for label in labels:
        key = label.strip().lower()
        if key in lower:
            series = pd.to_numeric(df.loc[lower[key]], errors="coerce").dropna()
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
            series = pd.to_numeric(df.loc[lower[key]], errors="coerce").dropna()
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
            series = pd.to_numeric(df.loc[lower[key]], errors="coerce").dropna()
            return [float(v) for v in series.iloc[:limit].tolist()]
    return []


def fetch_benchmark_returns(session: requests.Session) -> Tuple[pd.Series, str]:
    df = with_retry(
        lambda: yf.download(
            BENCHMARK,
            period=PRICE_PERIOD,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            session=session,
        ),
        retries=2,
        retryable=(Exception,),
        label="benchmark_fetch",
    )
    if df is None or df.empty:
        raise RuntimeError("Benchmark fetch failed")
    close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    if len(close) < MIN_PRICE_DAYS:
        raise RuntimeError("Benchmark price history too short")
    price_date = close.index[-1].strftime("%Y-%m-%d")
    returns = np.log(close).diff().dropna().tail(TRADING_DAYS)
    return returns, price_date


def fetch_price_metrics(symbol: str, benchmark_returns: pd.Series, price_date: str, session: requests.Session) -> PriceMetrics:
    hist = with_retry(
        lambda: yf.Ticker(symbol, session=session).history(
            period=PRICE_PERIOD, interval="1d", auto_adjust=True, repair=True
        ),
        retries=2,
        retryable=(Exception,),
        label="price_fetch",
    )
    if hist is None or hist.empty or "Close" not in hist.columns:
        return PriceMetrics(None, None, None, None, 0)

    close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
    close = close[close.index.strftime("%Y-%m-%d") <= price_date]
    if close.empty:
        return PriceMetrics(None, None, None, None, 0)

    returns = np.log(close).diff().dropna()
    joined = pd.concat([returns.rename("stock"), benchmark_returns.rename("bench")], axis=1, join="inner").dropna()
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


def fetch_financial_metrics(symbol: str, latest_price: Optional[float], session: requests.Session) -> FinancialMetrics:
    ticker = yf.Ticker(symbol, session=session)

    info = {}
    fast = {}
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

    total_assets, prev_total_assets = first_two_nonnull(balance_sheet, [
        "Total Assets",
    ])
    total_debt = last_nonnull(balance_sheet, ["Total Debt", "Net Debt"])
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
        "Cash Flow From Continuing Operating Activities",
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
    )


def pct_score(values: Dict[int, Optional[float]], invert: bool = False) -> Dict[int, Optional[float]]:
    clean = {k: v for k, v in values.items() if v is not None and not math.isnan(v)}
    if not clean:
        return {k: None for k in values}
    arr = np.array(list(clean.values()), dtype=float)
    low, high = np.nanpercentile(arr, 2.5), np.nanpercentile(arr, 97.5)
    clipped = {k: float(min(max(v, low), high)) for k, v in clean.items()}
    sorted_items = sorted(clipped.items(), key=lambda kv: kv[1])
    n = len(sorted_items)
    out: Dict[int, Optional[float]] = {k: None for k in values}
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


def build_row_base(code: str, company_name: str, sector: str) -> List[Any]:
    return [code, company_name, sector] + [""] * len(OUTPUT_HEADERS)


def set_output(row: List[Any], header: str, value: Any) -> None:
    idx = 3 + OUTPUT_HEADERS.index(header)
    row[idx] = value


def get_output(row: List[Any], header: str) -> Any:
    idx = 3 + OUTPUT_HEADERS.index(header)
    return row[idx]


def coverage_score(price_ok: bool, spec_ok: bool, dd_ok: bool, dur_ok: bool, q5_ok: bool) -> int:
    score = 0
    score += 40 if price_ok else 0
    score += 20 if spec_ok else 0
    score += 20 if dd_ok else 0
    score += 10 if dur_ok else 0
    score += 10 if q5_ok else 0
    return score


def reason_text(final_check: str, reason_codes: List[str], has_error: bool) -> str:
    if has_error:
        return "取得エラーで未判定"
    if final_check == "PASS":
        return "全ゲート通過、総合評価上位"
    if final_check == "WATCH":
        return "安全条件は通過、総合評価は中位"
    if "DD_LOW" in reason_codes or "DD_HIGH_VOL" in reason_codes:
        return "信用リスクが高く見送り"
    if any(code.startswith("SPEC_") for code in reason_codes):
        return "投機性が高く見送り"
    if any(code.startswith("DUR_") for code in reason_codes):
        return "現金回収力が弱く見送り"
    if any(code.startswith("Q5_") for code in reason_codes):
        return "成長品質が弱く見送り"
    if any(code.startswith("DATA_") for code in reason_codes) or "PARTIAL_MODEL_DATA" in reason_codes:
        return "データ不足で判定精度低下"
    return "データ不足で判定精度低下"


def format_num(v: Optional[float], digits: int = 4) -> Any:
    if v is None or math.isnan(v):
        return ""
    return round(v, digits)


def main() -> None:
    started = datetime.now(JST).isoformat(timespec="seconds")
    cfg = parse_secret()
    service = get_sheets_service(cfg.credentials_info)
    spreadsheet_id = spreadsheet_id_from_url(cfg.spreadsheet_url)
    rows_raw = fetch_sheet_rows(service, spreadsheet_id, cfg.worksheet_name)

    if not rows_raw:
        raise RuntimeError("Sheet is empty")

    data_rows = rows_raw[1:] if len(rows_raw) > 1 else []
    prepared_rows: List[List[Any]] = []
    code_to_indices: Dict[str, List[int]] = {}

    for i, raw in enumerate(data_rows, start=0):
        code = normalize_code(raw[0] if len(raw) >= 1 else "")
        company_name = str(raw[1]).strip() if len(raw) >= 2 else ""
        sector = str(raw[2]).strip() if len(raw) >= 3 else ""
        if code is None:
            continue
        base_row = build_row_base(code, company_name, sector)
        prepared_rows.append(base_row)
        code_to_indices.setdefault(code, []).append(len(prepared_rows) - 1)

    logger.info(f"START ts={started} rows={len(prepared_rows)}")

    session = yf_session()
    benchmark_returns, benchmark_price_date = fetch_benchmark_returns(session)
    logger.info(f"MARKET benchmark={BENCHMARK} price_date={benchmark_price_date}")
    logger.info(f"READ unique_codes={len(code_to_indices)}")

    results: Dict[str, Dict[str, Any]] = {}
    codes = list(code_to_indices.keys())

    for idx, code in enumerate(codes, start=1):
        symbol = f"{code}.T"
        result: Dict[str, Any] = {
            "SYMBOL": symbol,
            "NAME_YF": "",
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
            "ERROR_MESSAGE": "",
        }
        try:
            pm = fetch_price_metrics(symbol, benchmark_returns, benchmark_price_date, session)
            fm = fetch_financial_metrics(symbol, pm.latest_price, session)

            result["PRICE_DATE"] = pm.price_date or ""
            result["BETA_252D"] = pm.beta_252d
            result["IVOL_252D"] = pm.ivol_252d
            try:
                result["NAME_YF"] = yf.Ticker(symbol, session=session).info.get("shortName", "")[:100]
            except Exception:
                result["NAME_YF"] = ""

            price_ok = pm.price_days >= MIN_PRICE_DAYS and pm.price_date == benchmark_price_date
            spec_ok_inputs = all(v is not None for v in [pm.beta_252d, pm.ivol_252d, fm.total_assets, fm.total_debt, fm.operating_cf, fm.cash])
            dd_ok_inputs = all(v is not None for v in [fm.market_cap, fm.total_debt, pm.ivol_252d]) and price_ok
            dur_ok_inputs = all(v is not None for v in [fm.market_cap, fm.operating_cf]) and (fm.free_cf is not None or fm.capex is not None)
            q5_ok_inputs = all(v is not None for v in [fm.total_assets, fm.prev_total_assets, fm.equity, fm.prev_equity, fm.net_income, fm.prev_net_income, fm.operating_cf])
            result["DATA_COVERAGE"] = coverage_score(price_ok, spec_ok_inputs, dd_ok_inputs, dur_ok_inputs, q5_ok_inputs)
            if not price_ok:
                result["REASON_CODES"].append("DATA_PRICE_SHORT")
            if fm.total_debt is None:
                result["REASON_CODES"].append("DATA_DEBT_MISSING")
            if fm.market_cap is None:
                result["REASON_CODES"].append("DATA_MC_MISSING")
            if not (dur_ok_inputs and q5_ok_inputs):
                result["REASON_CODES"].append("PARTIAL_MODEL_DATA")

            # Store raw fields for later percentile scoring
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
        results[code] = result

        if idx % 500 == 0 or idx == len(codes):
            logger.info(f"PROGRESS done={idx} ok={sum(r['RUN_STATUS']=='OK' for r in results.values())} pass={sum(r['FINAL_CHECK']=='PASS' for r in results.values())} watch={sum(r['FINAL_CHECK']=='WATCH' for r in results.values())} fail={sum(r['FINAL_CHECK']=='FAIL' for r in results.values())} error={sum(r['RUN_STATUS']=='ERROR' for r in results.values())}")

    # Build raw metrics for scoring
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
        if r["_spec_inputs_ok"]:
            debt = r["_total_debt"]
            assets = r["_total_assets"]
            ocf = r["_operating_cf"]
            cash = r["_cash"]
            distress = None
            if debt and assets and debt > 0 and assets > 0:
                distress = 0.4 * (debt / assets) + 0.3 * (-(ocf / debt)) + 0.3 * (-(cash / debt))
            beta_off = abs((r["BETA_252D"] or 1.0) - 1.0) if r["BETA_252D"] is not None else None
            if distress is not None and beta_off is not None and r["IVOL_252D"] is not None:
                spec_badness[code] = 0.5 * r["IVOL_252D"] + 0.2 * beta_off + 0.3 * distress
            else:
                spec_badness[code] = None
        else:
            spec_badness[code] = None

        if r["_dd_inputs_ok"]:
            debt = r["_total_debt"] or 0.0
            mc = r["_market_cap"] or 0.0
            if debt <= 0:
                dd_raws[code] = 999.0
            else:
                std = r["_short_term_debt"]
                ltd = r["_long_term_debt"]
                default_point = (std + 0.5 * ltd) if (std is not None and ltd is not None) else (0.75 * debt)
                sigma_e = r["IVOL_252D"]
                va = mc + debt
                sigma_a = None if va <= 0 or sigma_e is None else sigma_e * mc / va
                if default_point and default_point > 0 and sigma_a and sigma_a > 0 and va > 0:
                    dd = (math.log(va / default_point) + (RISK_FREE_RATE + 0.5 * sigma_a * sigma_a)) / sigma_a
                    dd_raws[code] = dd
                else:
                    dd_raws[code] = None
        else:
            dd_raws[code] = None

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

        if r["_q5_inputs_ok"]:
            assets_t = r["_total_assets"]
            assets_p = r["_prev_total_assets"]
            eq_t = r["_equity"]
            eq_p = r["_prev_equity"]
            ni_t = r["_net_income"]
            ni_p = r["_prev_net_income"]
            ocf = r["_operating_cf"]
            q5_asset_growth[code] = None if assets_t in (None, 0) or assets_p in (None, 0) else assets_t / assets_p - 1.0
            avg_eq_t = None if eq_t is None or eq_p is None else (eq_t + eq_p) / 2.0
            avg_eq_p = eq_p
            q5_roe_ttm[code] = None if avg_eq_t in (None, 0) else ni_t / avg_eq_t
            prev_roe = None if avg_eq_p in (None, 0) else ni_p / avg_eq_p
            q5_delta_roe[code] = None if q5_roe_ttm[code] is None or prev_roe is None else q5_roe_ttm[code] - prev_roe
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
            elif final_score >= 70:
                r["FINAL_CHECK"] = "PASS"
            elif final_score >= 50:
                r["FINAL_CHECK"] = "WATCH"
            else:
                r["FINAL_CHECK"] = "FAIL"

        # Deduplicate and finalize reasons
        dedup = []
        for c in r["REASON_CODES"]:
            if c not in dedup:
                dedup.append(c)
        r["REASON_CODES"] = dedup
        for c in dedup:
            reason_counter[c] = reason_counter.get(c, 0) + 1
        r["REASON_TEXT"] = reason_text(r["FINAL_CHECK"], dedup, False)

    # Reflect to all sheet rows
    for code, indices in code_to_indices.items():
        r = results[code]
        for i in indices:
            row = prepared_rows[i]
            for h in OUTPUT_HEADERS:
                val = r.get(h, "")
                if h == "REASON_CODES" and isinstance(val, list):
                    val = ",".join(val)
                elif isinstance(val, float):
                    val = format_num(val, 6 if h in {"BETA_252D", "IVOL_252D", "DD_RAW", "FCF_YIELD", "CFO_YIELD", "CAPEX_BURDEN", "ASSET_GROWTH_1Y", "ROE_TTM"} else 2)
                elif val is None:
                    val = ""
                set_output(row, h, val)

    batch_write_rows(service, spreadsheet_id, cfg.worksheet_name, prepared_rows)
    logger.info(f"WRITE updated_rows={len(prepared_rows)}")

    top_reasons = sorted(reason_counter.items(), key=lambda kv: kv[1], reverse=True)[:3]
    if top_reasons:
        logger.info("TOP_REASONS " + " ".join(f"{k}={v}" for k, v in top_reasons))
    else:
        logger.info("TOP_REASONS none=0")

    pass_n = sum(1 for r in results.values() if r["FINAL_CHECK"] == "PASS")
    watch_n = sum(1 for r in results.values() if r["FINAL_CHECK"] == "WATCH")
    fail_n = sum(1 for r in results.values() if r["FINAL_CHECK"] == "FAIL")
    error_n = sum(1 for r in results.values() if r["RUN_STATUS"] == "ERROR")
    logger.info(f"END pass={pass_n} watch={watch_n} fail={fail_n} error={error_n}")


if __name__ == "__main__":
    try:
        main()
    except HttpError:
        logger.info("FATAL stage=sheet error=SHEET_READ_FAILED")
        raise
    except Exception as exc:
        msg = str(exc)
        if "Benchmark" in msg:
            logger.info("FATAL stage=benchmark error=PRICE_FETCH_FAILED")
        else:
            logger.info("FATAL stage=runtime error=UNKNOWN_ERROR")
        raise
