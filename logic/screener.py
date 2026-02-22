"""
連続上昇銘柄スクリーニングロジック（J-Quants API V2対応）
=============================================================
V2 API のカラム名略称に対応:
  AdjC  → 修正後終値（日足の終値として使用）
  AdjVo → 修正後出来高
"""

import logging
import time
from datetime import date, timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd

import api.jquants as jq

logger = logging.getLogger(__name__)

# 無料プランのデータ遅延日数（12週間 = 84日）
FREE_PLAN_DELAY_DAYS = 84

# 必要取引日数の計算時バッファ係数（週末・祝日考慮）
TRADING_DAY_BUFFER = 1.6

# V2カラム: 終値（修正後優先）
V2_CLOSE_COLS = ["AdjC", "C"]   # AdjC = 修正後終値, C = 生終値
V2_VOLUME_COL = "AdjVo"          # 修正後出来高

# PER/PBRカラム候補（V2 fins/summary の略称）
PER_CANDIDATE_COLS = ["PER", "ForwardPER", "PriceEarningsRatio", "EarningsPerShare"]
PBR_CANDIDATE_COLS = ["PBR", "PriceBookValueRatio", "PriceToBookRatio"]


# ─────────────────────────────────────────────────────────
# 日付計算ユーティリティ
# ─────────────────────────────────────────────────────────

def get_end_date() -> date:
    """
    無料プランのデータ遅延を考慮した取得可能な最新日付を返す。

    Returns:
        本日から FREE_PLAN_DELAY_DAYS 日前の日付
    """
    return date.today() - timedelta(days=FREE_PLAN_DELAY_DAYS)


def calc_required_calendar_days(timeframe: str, n_periods: int) -> int:
    """
    スクリーニングに必要なカレンダー日数を計算する。

    Args:
        timeframe: "日足" | "週足" | "月足"
        n_periods: 連続上昇期間数

    Returns:
        必要なカレンダー日数
    """
    if timeframe == "日足":
        trading_days = (n_periods + 10) * TRADING_DAY_BUFFER
    elif timeframe == "週足":
        trading_days = (n_periods + 4) * 5 * TRADING_DAY_BUFFER
    else:  # 月足
        trading_days = (n_periods + 3) * 21 * TRADING_DAY_BUFFER

    # 取引日 → カレンダー日（週5日営業: ×7/5）
    return int(trading_days * 7 / 5)


def generate_weekday_dates_desc(start: date, end: date) -> list[str]:
    """
    start〜end の範囲で平日（月〜金）の日付リストを降順で生成する。

    Args:
        start: 開始日
        end: 終了日

    Returns:
        YYYYMMDD形式の日付文字列リスト（新しい順）
    """
    dates: list[str] = []
    current = end
    while current >= start:
        if current.weekday() < 5:  # 0=月〜4=金
            dates.append(current.strftime("%Y%m%d"))
        current -= timedelta(days=1)
    return dates


def _detect_close_col(df: pd.DataFrame) -> str:
    """
    DataFrameから使用する終値カラム名を決定する。
    V2 AdjC → V1 AdjustmentClose → C → Close の優先順で検索する。

    Args:
        df: 株価DataFrame

    Returns:
        使用する終値カラム名
    """
    priority = ["AdjC", "AdjustmentClose", "C", "Close"]
    for col in priority:
        if col in df.columns:
            return col
    # フォールバック: 最初のカラムを返す（異常系）
    logger.warning("終値カラムが見つかりません。最初のカラムを使用します。")
    return df.columns[0]


def _detect_volume_col(df: pd.DataFrame) -> Optional[str]:
    """
    DataFrameから使用する出来高カラム名を決定する。

    Args:
        df: 株価DataFrame

    Returns:
        出来高カラム名 または None
    """
    priority = ["AdjVo", "Volume", "Vo"]
    for col in priority:
        if col in df.columns:
            return col
    return None


# ─────────────────────────────────────────────────────────
# 株価データ取得
# ─────────────────────────────────────────────────────────

def fetch_price_data(
    api_key: str,
    timeframe: str,
    n_periods: int,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """
    スクリーニングに必要な期間分の全銘柄株価データを取得する。
    新しい日付から降順にフェッチし、必要な取引日数が揃い次第終了する。

    Args:
        api_key: J-Quants APIキー
        timeframe: "日足" | "週足" | "月足"
        n_periods: 連続上昇期間数
        progress_callback: 進捗コールバック (current, total, message)

    Returns:
        全銘柄・全日付の株価DataFrame（失敗時は空のDataFrame）
    """
    end_date = get_end_date()
    calendar_days = calc_required_calendar_days(timeframe, n_periods)
    start_date = end_date - timedelta(days=calendar_days)

    # 必要な最低取引日数（バッファなし）
    if timeframe == "日足":
        min_trading_days = n_periods + 5
    elif timeframe == "週足":
        min_trading_days = (n_periods + 3) * 5
    else:
        min_trading_days = (n_periods + 2) * 21

    date_candidates = generate_weekday_dates_desc(start_date, end_date)
    total_candidates = len(date_candidates)

    all_dfs: list[pd.DataFrame] = []
    trading_days_collected = 0

    for i, date_str in enumerate(date_candidates):
        if progress_callback:
            progress_callback(
                i + 1,
                total_candidates,
                f"📥 株価データ取得中: {date_str}（取引日 {trading_days_collected}/{min_trading_days}）",
            )

        df = jq.get_daily_quotes_by_date(api_key, date_str)

        if not df.empty:
            df["_date_key"] = date_str  # 統一日付キー
            all_dfs.append(df)
            trading_days_collected += 1

            if trading_days_collected >= min_trading_days:
                break

        time.sleep(jq.REQUEST_INTERVAL)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # V2 は Date カラムが "YYYY-MM-DD" or "YYYYMMDD"（どちらかに統一）
    if "Date" not in combined.columns and "_date_key" in combined.columns:
        combined["Date"] = combined["_date_key"]

    logger.info(f"株価取得完了: {trading_days_collected}取引日, {len(combined)}行")
    return combined


# ─────────────────────────────────────────────────────────
# リサンプリング（週足・月足）
# ─────────────────────────────────────────────────────────

def resample_to_period(
    daily_df: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:
    """
    日次データを週足または月足にリサンプリングする。
    終値は期間最終営業日の終値、出来高は期間合計を使用。

    Args:
        daily_df: 日次株価DataFrame
        timeframe: "週足" | "月足"

    Returns:
        リサンプリング済みDataFrame（Code, Date, Close, Volume）
    """
    freq = "W-FRI" if timeframe == "週足" else "ME"
    close_col = _detect_close_col(daily_df)
    vol_col = _detect_volume_col(daily_df)

    df = daily_df.copy()

    # 日付カラムを datetime 型に変換
    date_col = "Date" if "Date" in df.columns else "_date_key"
    df["_dt"] = pd.to_datetime(df[date_col].astype(str), format="mixed", dayfirst=False)
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    if vol_col:
        df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")

    result_dfs: list[pd.DataFrame] = []

    for code, group in df.groupby("Code"):
        group = group.set_index("_dt").sort_index()

        resampled_close = group[close_col].resample(freq).last()
        period_df = pd.DataFrame({"Close": resampled_close})

        if vol_col:
            resampled_vol = group[vol_col].resample(freq).sum()
            period_df["Volume"] = resampled_vol

        period_df = period_df.dropna(subset=["Close"])
        period_df["Code"] = code
        period_df = period_df.reset_index().rename(columns={"_dt": "Date"})
        result_dfs.append(period_df)

    return pd.concat(result_dfs, ignore_index=True) if result_dfs else pd.DataFrame()


# ─────────────────────────────────────────────────────────
# 連続上昇判定
# ─────────────────────────────────────────────────────────

def count_consecutive_rising(close_series: pd.Series) -> int:
    """
    終値の連続上昇期間数を返す。
    直近から遡り「当日終値 > 前日終値」が連続している足数を数える。

    Args:
        close_series: 終値Series（古い順）

    Returns:
        連続上昇期間数（0以上）
    """
    values = pd.to_numeric(close_series, errors="coerce").dropna().values
    if len(values) < 2:
        return 0

    count = 0
    for i in range(len(values) - 1, 0, -1):
        if values[i] > values[i - 1]:
            count += 1
        else:
            break
    return count


def screen_consecutive_rising(
    period_df: pd.DataFrame,
    n_periods: int,
    close_col: str = "Close",
) -> pd.DataFrame:
    """
    銘柄ごとに連続上昇数を計算し、n_periods 以上の銘柄を返す。

    Args:
        period_df: 足種変換済みDataFrame（Code, Date, Close, [Volume]）
        n_periods: 最低連続上昇期間数
        close_col: 終値カラム名

    Returns:
        通過銘柄DataFrame（Code, 株価, 出来高, 連続上昇数）
    """
    results: list[dict] = []

    for code, group in period_df.groupby("Code"):
        sorted_group = group.sort_values("Date")
        consecutive = count_consecutive_rising(sorted_group[close_col])

        if consecutive >= n_periods:
            latest = sorted_group.iloc[-1]
            vol_col = _detect_volume_col(sorted_group) or "Volume"
            results.append(
                {
                    "Code": str(code),
                    "株価": pd.to_numeric(latest[close_col], errors="coerce"),
                    "出来高": pd.to_numeric(
                        latest.get(vol_col, np.nan), errors="coerce"
                    ),
                    "連続上昇数": consecutive,
                }
            )

    return pd.DataFrame(results) if results else pd.DataFrame()


# ─────────────────────────────────────────────────────────
# 銘柄情報マージ・フィルター
# ─────────────────────────────────────────────────────────

def merge_listed_info(
    screened_df: pd.DataFrame,
    listed_info: pd.DataFrame,
) -> pd.DataFrame:
    """
    スクリーニング結果に銘柄名・業種・時価総額をマージする。
    V2カラム名（CompanyName, Sector33CodeName, Shares等）に対応。

    Args:
        screened_df: スクリーニング通過銘柄DataFrame
        listed_info: get_listed_info() で取得した銘柄一覧DataFrame

    Returns:
        マージ済みDataFrame
    """
    listed = listed_info.copy()
    screened = screened_df.copy()

    # コードを4桁に正規化（V2は末尾に"0"が付く場合がある）
    listed["CodeKey"] = listed["Code"].astype(str).str[:4]
    screened["CodeKey"] = screened["Code"].astype(str).str[:4]

    # マージするカラムを選択
    info_cols = ["CodeKey"]
    for col in ["CompanyName", "Sector33CodeName", "Shares", "MarketCodeName"]:
        if col in listed.columns:
            info_cols.append(col)

    listed_unique = listed[info_cols].drop_duplicates("CodeKey")
    merged = screened.merge(listed_unique, on="CodeKey", how="left")

    # 時価総額 = 株価 × 上場株式数（億円単位）
    if "Shares" in merged.columns:
        merged["Shares"] = pd.to_numeric(merged["Shares"], errors="coerce")
        merged["時価総額(億円)"] = (merged["株価"] * merged["Shares"] / 1e8).round(1)
    else:
        merged["時価総額(億円)"] = np.nan

    return merged


def apply_filters(
    df: pd.DataFrame,
    min_price: float,
    max_price: float,
    min_cap_oku: float,
    max_cap_oku: float,
) -> pd.DataFrame:
    """
    株価・時価総額フィルターを適用する。
    max 値が 0 のとき上限なし。

    Args:
        df: マージ済みDataFrame
        min_price: 株価最小値（円）
        max_price: 株価最大値（円）。0=上限なし
        min_cap_oku: 時価総額最小値（億円）
        max_cap_oku: 時価総額最大値（億円）。0=上限なし

    Returns:
        フィルター済みDataFrame
    """
    result = df.copy()

    if min_price > 0:
        result = result[result["株価"].fillna(0) >= min_price]
    if max_price > 0:
        result = result[result["株価"].fillna(np.inf) <= max_price]

    if "時価総額(億円)" in result.columns:
        if min_cap_oku > 0:
            result = result[result["時価総額(億円)"].fillna(0) >= min_cap_oku]
        if max_cap_oku > 0:
            result = result[result["時価総額(億円)"].fillna(np.inf) <= max_cap_oku]

    return result


# ─────────────────────────────────────────────────────────
# PER・PBR 取得
# ─────────────────────────────────────────────────────────

def fetch_per_pbr(
    api_key: str,
    codes: list[str],
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """
    指定銘柄のPER・PBRを財務サマリーから取得する。

    Args:
        api_key: J-Quants APIキー
        codes: 銘柄コードリスト（4桁）
        progress_callback: 進捗コールバック

    Returns:
        CodeKey, PER, PBR カラムのDataFrame
    """
    rows: list[dict] = []
    total = len(codes)

    for i, code in enumerate(codes):
        if progress_callback:
            progress_callback(
                i + 1, total, f"📊 PER・PBR取得中: {code} ({i+1}/{total})"
            )

        summary_df = jq.get_fins_summary(api_key, code)
        per: Optional[float] = None
        pbr: Optional[float] = None

        if not summary_df.empty:
            # 最新開示日で並べ替え
            for date_col in ["DisclosedDate", "Date", "PeriodEnd"]:
                if date_col in summary_df.columns:
                    summary_df = summary_df.sort_values(date_col, ascending=False)
                    break

            for col in PER_CANDIDATE_COLS:
                if col in summary_df.columns:
                    val = pd.to_numeric(summary_df[col].iloc[0], errors="coerce")
                    if not pd.isna(val) and val > 0:
                        per = round(float(val), 2)
                        break

            for col in PBR_CANDIDATE_COLS:
                if col in summary_df.columns:
                    val = pd.to_numeric(summary_df[col].iloc[0], errors="coerce")
                    if not pd.isna(val) and val > 0:
                        pbr = round(float(val), 2)
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
    """
    連続上昇銘柄スクリーニングを実行する。

    処理フロー:
        1. 株価データを日付単位で一括取得
        2. 週足・月足はpandasでリサンプリング
        3. 銘柄ごとに連続上昇期間数を判定
        4. 銘柄情報をマージ（名称・業種・時価総額）
        5. 株価・時価総額フィルター適用
        6. （オプション）PER・PBR取得
        7. 出力カラム整形

    Args:
        api_key: J-Quants APIキー
        listed_info: get_listed_info() で取得した銘柄一覧DataFrame
        timeframe: "日足" | "週足" | "月足"
        n_periods: 最低連続上昇期間数
        min_price: 株価最小値（円）
        max_price: 株価最大値（円）。0=上限なし
        min_cap_oku: 時価総額最小値（億円）
        max_cap_oku: 時価総額最大値（億円）。0=上限なし
        include_per_pbr: PER・PBRを取得するか
        progress_callback: 進捗コールバック (current, total, message)

    Returns:
        スクリーニング結果DataFrame。該当なしは空のDataFrame。
    """
    # ── STEP 1: 株価データ取得 ──────────────────────────
    price_data = fetch_price_data(
        api_key, timeframe, n_periods, progress_callback
    )
    if price_data.empty:
        logger.warning("株価データが取得できませんでした。")
        return pd.DataFrame()

    # ── STEP 2: リサンプリング（週足・月足） ─────────────
    if timeframe in ("週足", "月足"):
        if progress_callback:
            progress_callback(0, 1, f"📐 {timeframe}へリサンプリング中...")
        period_df = resample_to_period(price_data, timeframe)
        close_col = "Close"
    else:
        period_df = price_data.copy()
        date_col = "Date" if "Date" in period_df.columns else "_date_key"
        period_df["Date"] = pd.to_datetime(
            period_df[date_col].astype(str), format="mixed", dayfirst=False
        )
        close_col = _detect_close_col(period_df)
        period_df[close_col] = pd.to_numeric(period_df[close_col], errors="coerce")
        vol_col = _detect_volume_col(period_df)
        if vol_col:
            period_df[vol_col] = pd.to_numeric(period_df[vol_col], errors="coerce")

    if period_df.empty:
        return pd.DataFrame()

    # ── STEP 3: 連続上昇判定 ───────────────────────────
    if progress_callback:
        progress_callback(0, 1, "🔍 連続上昇を判定中...")
    screened = screen_consecutive_rising(period_df, n_periods, close_col)

    if screened.empty:
        return pd.DataFrame()

    # ── STEP 4: 銘柄情報マージ ───────────────────────────
    if progress_callback:
        progress_callback(0, 1, "🔗 銘柄情報をマージ中...")
    merged = merge_listed_info(screened, listed_info)

    # ── STEP 5: フィルター ───────────────────────────────
    merged = apply_filters(merged, min_price, max_price, min_cap_oku, max_cap_oku)
    if merged.empty:
        return pd.DataFrame()

    # ── STEP 6: PER・PBR取得（オプション） ─────────────
    if include_per_pbr and not merged.empty:
        codes = merged["CodeKey"].astype(str).tolist()
        per_pbr_df = fetch_per_pbr(api_key, codes, progress_callback)
        merged = merged.merge(per_pbr_df, on="CodeKey", how="left")
    else:
        merged["PER"] = np.nan
        merged["PBR"] = np.nan

    # ── STEP 7: 出力カラム整形 ───────────────────────────
    unit_map = {"日足": "日", "週足": "週", "月足": "ヶ月"}
    unit = unit_map[timeframe]
    merged["連続上昇期間"] = merged["連続上昇数"].apply(lambda x: f"{x} {unit}")
    merged["銘柄コード"] = merged["CodeKey"]

    rename_map = {
        "CompanyName": "銘柄名",
        "Sector33CodeName": "業種",
        "MarketCodeName": "市場",
        "株価": "株価(円)",
    }
    merged = merged.rename(columns=rename_map)

    output_cols = [
        "銘柄コード", "銘柄名", "業種",
        "株価(円)", "時価総額(億円)", "出来高",
        "PER", "PBR", "連続上昇期間",
    ]
    output_cols = [c for c in output_cols if c in merged.columns]
    result = merged[output_cols].sort_values(
        "連続上昇期間", ascending=False
    ).reset_index(drop=True)

    logger.info(f"スクリーニング完了: {len(result)}件")
    return result
