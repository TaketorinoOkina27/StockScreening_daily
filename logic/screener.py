"""
連続上昇銘柄スクリーニングロジック（J-Quants API V2対応）
"""

import logging
import time
from datetime import date, timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd

import api.jquants as jq

logger = logging.getLogger(__name__)

FREE_PLAN_DELAY_DAYS = 84   # 無料プランのデータ遅延（12週間）
TRADING_DAY_BUFFER   = 1.6  # 週末・祝日考慮バッファ

# PER/PBRカラム候補（V2 fins/summary）
PER_COLS = ["PER", "ForwardPER", "PriceEarningsRatio"]
PBR_COLS = ["PBR", "PriceBookValueRatio", "PriceToBookRatio"]


# ─────────────────────────────────────────────────────────
# カラム検出ヘルパー
# ─────────────────────────────────────────────────────────

def _close_col(df: pd.DataFrame) -> str:
    """使用する終値カラム名を優先順に決定する。"""
    for c in ["AdjC", "AdjustmentClose", "C", "Close"]:
        if c in df.columns:
            return c
    return df.columns[0]


def _vol_col(df: pd.DataFrame) -> Optional[str]:
    """使用する出来高カラム名を優先順に決定する。
    V2: Vo=出来高, AdjVo=修正後出来高
    V1: Volume
    """
    for c in ["Vo", "AdjVo", "Volume"]:
        if c in df.columns:
            return c
    return None


# ─────────────────────────────────────────────────────────
# 日付計算
# ─────────────────────────────────────────────────────────

def get_end_date() -> date:
    """無料プラン遅延を考慮した取得可能最新日付。"""
    return date.today() - timedelta(days=FREE_PLAN_DELAY_DAYS)


def calc_required_calendar_days(timeframe: str, n_periods: int) -> int:
    """スクリーニングに必要なカレンダー日数を計算する。"""
    if timeframe == "日足":
        trading_days = (n_periods + 10) * TRADING_DAY_BUFFER
    elif timeframe == "週足":
        trading_days = (n_periods + 4) * 5 * TRADING_DAY_BUFFER
    else:
        trading_days = (n_periods + 3) * 21 * TRADING_DAY_BUFFER
    return int(trading_days * 7 / 5)


def generate_weekday_dates_desc(start: date, end: date) -> list[str]:
    """start〜endの平日（月〜金）をYYYYMMDD形式で降順に返す。"""
    dates: list[str] = []
    cur = end
    while cur >= start:
        if cur.weekday() < 5:
            dates.append(cur.strftime("%Y%m%d"))
        cur -= timedelta(days=1)
    return dates


# ─────────────────────────────────────────────────────────
# 株価データ取得
# ─────────────────────────────────────────────────────────

def fetch_price_data(
    api_key: str,
    timeframe: str,
    n_periods: int,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """必要な期間分の全銘柄株価データを新しい日付から順に取得する。"""
    end_date = get_end_date()
    start_date = end_date - timedelta(days=calc_required_calendar_days(timeframe, n_periods))

    if timeframe == "日足":
        min_trading = n_periods + 5
    elif timeframe == "週足":
        min_trading = (n_periods + 3) * 5
    else:
        min_trading = (n_periods + 2) * 21

    date_candidates = generate_weekday_dates_desc(start_date, end_date)
    collected = 0
    all_dfs: list[pd.DataFrame] = []

    for i, d in enumerate(date_candidates):
        if progress_callback:
            progress_callback(
                i + 1, len(date_candidates),
                f"📥 株価取得中: {d}（取引日 {collected}/{min_trading}）",
            )
        df = jq.get_daily_quotes_by_date(api_key, d)
        if not df.empty:
            df["_date"] = d
            all_dfs.append(df)
            collected += 1
            if collected >= min_trading:
                break
        time.sleep(jq.REQUEST_INTERVAL)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    if "Date" not in combined.columns:
        combined["Date"] = combined["_date"]
    return combined


# ─────────────────────────────────────────────────────────
# リサンプリング（週足・月足）
# ─────────────────────────────────────────────────────────

def resample_to_period(daily_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """日次データを週足または月足にリサンプリングする。"""
    freq = "W-FRI" if timeframe == "週足" else "ME"
    cc = _close_col(daily_df)
    vc = _vol_col(daily_df)

    df = daily_df.copy()
    date_col = "Date" if "Date" in df.columns else "_date"
    df["_dt"] = pd.to_datetime(df[date_col].astype(str), format="mixed", dayfirst=False)
    df[cc] = pd.to_numeric(df[cc], errors="coerce")
    if vc:
        df[vc] = pd.to_numeric(df[vc], errors="coerce")

    result: list[pd.DataFrame] = []
    for code, grp in df.groupby("Code"):
        grp = grp.set_index("_dt").sort_index()
        pf = pd.DataFrame({"Close": grp[cc].resample(freq).last()})
        if vc:
            pf["Volume"] = grp[vc].resample(freq).sum()
        pf = pf.dropna(subset=["Close"])
        pf["Code"] = code
        result.append(pf.reset_index().rename(columns={"_dt": "Date"}))

    return pd.concat(result, ignore_index=True) if result else pd.DataFrame()


# ─────────────────────────────────────────────────────────
# 連続上昇判定
# ─────────────────────────────────────────────────────────

def count_consecutive_rising(series: pd.Series) -> int:
    """終値の連続上昇期間数を直近から遡って数える。"""
    vals = pd.to_numeric(series, errors="coerce").dropna().values
    if len(vals) < 2:
        return 0
    count = 0
    for i in range(len(vals) - 1, 0, -1):
        if vals[i] > vals[i - 1]:
            count += 1
        else:
            break
    return count


def screen_consecutive_rising(
    period_df: pd.DataFrame, n_periods: int, close_col: str = "Close"
) -> pd.DataFrame:
    """n_periods以上連続上昇している銘柄を返す。"""
    results: list[dict] = []
    vc = _vol_col(period_df)

    for code, grp in period_df.groupby("Code"):
        grp = grp.sort_values("Date")
        count = count_consecutive_rising(grp[close_col])
        if count >= n_periods:
            latest = grp.iloc[-1]
            results.append({
                "Code": str(code),
                "株価": pd.to_numeric(latest[close_col], errors="coerce"),
                "出来高": pd.to_numeric(latest.get(vc, np.nan), errors="coerce") if vc else np.nan,
                "連続上昇数": count,
            })

    return pd.DataFrame(results) if results else pd.DataFrame()


# ─────────────────────────────────────────────────────────
# 銘柄情報マージ・フィルター
# ─────────────────────────────────────────────────────────

def merge_listed_info(screened: pd.DataFrame, listed: pd.DataFrame) -> pd.DataFrame:
    """
    スクリーニング結果に銘柄名・業種をマージする。

    V2 /equities/master のカラム名:
        CoName  : 銘柄名（V1: CompanyName）
        S33Nm   : 33業種名（V1: Sector33CodeName）
        MktNm   : 市場区分名（V1: MarketCodeName）
        ScaleCat: 規模区分（TOPIX Large70/Mid400/Small1/Small2）
        ※ Sharesカラム（上場株式数）が存在しないため時価総額の計算は不可。
    """
    li = listed.copy()
    sc = screened.copy()

    li["CodeKey"] = li["Code"].astype(str).str[:4]
    sc["CodeKey"] = sc["Code"].astype(str).str[:4]

    # ── V2カラム名 → V1カラム名へ正規化（どちらでも動くように） ──
    rename_to_canonical = {
        "CoName":   "CompanyName",       # V2→V1統一名
        "S33Nm":    "Sector33CodeName",  # V2→V1統一名
        "MktNm":    "MarketCodeName",    # V2→V1統一名
        "ScaleCat": "ScaleCat",          # V2のみ存在（規模区分）
    }
    li = li.rename(columns={k: v for k, v in rename_to_canonical.items() if k in li.columns})

    # マージするカラムを選択（存在するもののみ）
    info_cols = ["CodeKey"]
    for c in ["CompanyName", "Sector33CodeName", "MarketCodeName", "ScaleCat",
              "Shares", "MarketCapitalization"]:
        if c in li.columns:
            info_cols.append(c)

    merged = sc.merge(li[info_cols].drop_duplicates("CodeKey"), on="CodeKey", how="left")

    # 時価総額を億円換算（データがある場合のみ）
    if "MarketCapitalization" in merged.columns:
        # V1: MarketCapitalization が直接億円換算前の円単位で存在
        merged["時価総額(億円)"] = (
            pd.to_numeric(merged["MarketCapitalization"], errors="coerce") / 1e8
        ).round(1)
    elif "Shares" in merged.columns:
        # Shares × 株価 で推算
        merged["時価総額(億円)"] = (
            pd.to_numeric(merged["Shares"], errors="coerce") * merged["株価"] / 1e8
        ).round(1)
    else:
        # V2 /equities/master には株式数がないため計算不可 → N/A
        merged["時価総額(億円)"] = np.nan

    return merged


def apply_filters(
    df: pd.DataFrame,
    min_price: float, max_price: float,
    min_cap: float, max_cap: float,
) -> pd.DataFrame:
    """株価・時価総額フィルターを適用する。max=0 は上限なし。"""
    r = df.copy()
    if min_price > 0:
        r = r[r["株価"].fillna(0) >= min_price]
    if max_price > 0:
        r = r[r["株価"].fillna(np.inf) <= max_price]
    if "時価総額(億円)" in r.columns:
        if min_cap > 0:
            r = r[r["時価総額(億円)"].fillna(0) >= min_cap]
        if max_cap > 0:
            r = r[r["時価総額(億円)"].fillna(np.inf) <= max_cap]
    return r


# ─────────────────────────────────────────────────────────
# PER・PBR取得
# ─────────────────────────────────────────────────────────

def fetch_per_pbr(
    api_key: str,
    codes: list[str],
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """指定銘柄リストのPER・PBRを取得する。"""
    rows: list[dict] = []
    for i, code in enumerate(codes):
        if progress_callback:
            progress_callback(i + 1, len(codes), f"📊 PER/PBR取得中: {code} ({i+1}/{len(codes)})")
        df = jq.get_fins_summary(api_key, code)
        per, pbr = None, None
        if not df.empty:
            for dc in ["DisclosedDate", "Date", "PeriodEnd"]:
                if dc in df.columns:
                    df = df.sort_values(dc, ascending=False)
                    break
            for c in PER_COLS:
                if c in df.columns:
                    v = pd.to_numeric(df[c].iloc[0], errors="coerce")
                    if pd.notna(v) and v > 0:
                        per = round(float(v), 2)
                        break
            for c in PBR_COLS:
                if c in df.columns:
                    v = pd.to_numeric(df[c].iloc[0], errors="coerce")
                    if pd.notna(v) and v > 0:
                        pbr = round(float(v), 2)
                        break
        rows.append({"CodeKey": str(code)[:4], "PER": per, "PBR": pbr})
        time.sleep(jq.REQUEST_INTERVAL)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────
# メインスクリーニング関数
# ─────────────────────────────────────────────────────────

def run_screening(
    api_key: str,
    listed_info: pd.DataFrame,
    timeframe: str,
    n_periods: int,
    min_price: float,
    max_price: float,
    min_cap_oku: float,
    max_cap_oku: float,
    include_per_pbr: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """連続上昇銘柄スクリーニングを実行してDataFrameを返す。"""

    # STEP 1: 株価データ取得
    price_data = fetch_price_data(api_key, timeframe, n_periods, progress_callback)
    if price_data.empty:
        return pd.DataFrame()

    # STEP 2: 週足・月足リサンプリング
    if timeframe in ("週足", "月足"):
        if progress_callback:
            progress_callback(0, 1, f"📐 {timeframe}へリサンプリング中...")
        period_df = resample_to_period(price_data, timeframe)
        close_col = "Close"
    else:
        period_df = price_data.copy()
        date_col = "Date" if "Date" in period_df.columns else "_date"
        period_df["Date"] = pd.to_datetime(
            period_df[date_col].astype(str), format="mixed", dayfirst=False
        )
        close_col = _close_col(period_df)
        period_df[close_col] = pd.to_numeric(period_df[close_col], errors="coerce")

    if period_df.empty:
        return pd.DataFrame()

    # STEP 3: 連続上昇判定
    if progress_callback:
        progress_callback(0, 1, "🔍 連続上昇を判定中...")
    screened = screen_consecutive_rising(period_df, n_periods, close_col)
    if screened.empty:
        return pd.DataFrame()

    # STEP 4: 銘柄情報マージ
    if progress_callback:
        progress_callback(0, 1, "🔗 銘柄情報をマージ中...")
    merged = merge_listed_info(screened, listed_info)

    # STEP 5: フィルター
    merged = apply_filters(merged, min_price, max_price, min_cap_oku, max_cap_oku)
    if merged.empty:
        return pd.DataFrame()

    # STEP 6: PER・PBR（オプション）
    if include_per_pbr:
        per_pbr = fetch_per_pbr(api_key, merged["CodeKey"].tolist(), progress_callback)
        merged = merged.merge(per_pbr, on="CodeKey", how="left")
    else:
        merged["PER"] = np.nan
        merged["PBR"] = np.nan

    # STEP 7: 出力カラム整形
    unit = {"日足": "日", "週足": "週", "月足": "ヶ月"}[timeframe]
    merged["連続上昇期間"] = merged["連続上昇数"].apply(lambda x: f"{x} {unit}")
    merged["銘柄コード"] = merged["CodeKey"]
    merged = merged.rename(columns={
        "CompanyName": "銘柄名",
        "Sector33CodeName": "業種",
        "MarketCodeName": "市場",
        "株価": "株価(円)",
    })

    cols = ["銘柄コード", "銘柄名", "業種", "株価(円)", "時価総額(億円)",
            "出来高", "PER", "PBR", "連続上昇期間"]
    cols = [c for c in cols if c in merged.columns]
    return merged[cols].sort_values("連続上昇期間", ascending=False).reset_index(drop=True)