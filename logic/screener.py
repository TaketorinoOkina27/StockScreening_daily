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

# ── 定数 ──────────────────────────────────────────────────
FREE_PLAN_DELAY_DAYS      = 84   # 無料プランのデータ遅延（12週間）
DAYS_PER_WEEK             = 5    # 1週 = 5取引日
DAYS_PER_MONTH            = 21   # 1ヶ月 = 21取引日

PER_COLS = ["PER", "ForwardPER", "PriceEarningsRatio"]
PBR_COLS = ["PBR", "PriceBookValueRatio", "PriceToBookRatio"]


# ─────────────────────────────────────────────────────────
# カラム検出
# ─────────────────────────────────────────────────────────

def _close_col(df: pd.DataFrame) -> str:
    """終値カラムを優先順に検出する。"""
    for c in ["AdjC", "AdjustmentClose", "C", "Close"]:
        if c in df.columns:
            return c
    return df.columns[0]


def _vol_col(df: pd.DataFrame) -> Optional[str]:
    """出来高カラムを優先順に検出する（V2: Vo が通常出来高）。"""
    for c in ["Vo", "AdjVo", "Volume"]:
        if c in df.columns:
            return c
    return None


# ─────────────────────────────────────────────────────────
# 必要取引日数の定義
# ─────────────────────────────────────────────────────────

def _min_trading_days(timeframe: str, n_periods: int) -> int:
    """足種・連続上昇期間から、収集すべき取引日数を返す。"""
    if timeframe == "日足":
        return n_periods + 1          # n回の上昇比較にはn+1個のデータが必要
    elif timeframe == "週足":
        return (n_periods + 1) * DAYS_PER_WEEK
    else:  # 月足
        return (n_periods + 1) * DAYS_PER_MONTH


def get_end_date() -> date:
    """無料プラン遅延を考慮した取得可能最新日付。"""
    return date.today() - timedelta(days=FREE_PLAN_DELAY_DAYS)


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
    スクリーニングに必要な取引日数分の株価を取得する。

    動作:
      取得可能最新日（本日 - 84日）から1日ずつ過去へ遡り、
      実データが返った日だけ counted する。
      min_trading 日分のデータが揃った時点で終了する。
      カレンダー日数の事前計算は行わない。

    終了条件:
      (1) collected >= min_trading  → 正常終了
      (2) tried_trading >= min_trading + 15  → 安全打ち切り（警告ログ）

    429（レートリミット）: 30秒待機して同じ日を再試行する。
    非営業日（空レスポンス）: カウントせず前日へ進む。
    """
    min_trading     = _min_trading_days(timeframe, n_periods)
    end_date        = get_end_date()
    cur             = end_date
    collected       = 0   # 実データが取れた取引日数
    tried_trading   = 0   # APIを叩いた取引日数（安全打ち切り用）
    all_dfs: list[pd.DataFrame] = []

    logger.info(
        f"株価取得開始: {timeframe} n={n_periods}, "
        f"必要取引日数={min_trading}, 取得可能最新日={end_date}"
    )

    while collected < min_trading:
        # 安全上限チェック
        if tried_trading >= min_trading + 15:
            logger.warning(
                f"安全打ち切り: {tried_trading}取引日遡ったが "
                f"収集={collected}/{min_trading} 日しか取得できませんでした。"
            )
            break

        # 土日はAPIを叩かずスキップ
        if cur.weekday() >= 5:
            cur -= timedelta(days=1)
            continue

        d = cur.strftime("%Y%m%d")
        tried_trading += 1

        try:
            df = jq.get_daily_quotes_by_date(api_key, d)
            if not df.empty:
                df["_date"] = d
                all_dfs.append(df)
                collected += 1
            # 空 = 非営業日（祝日など）→ 前日へ進む
            cur -= timedelta(days=1)
            if progress_callback:
                progress_callback(
                    collected, min_trading,
                    f"📥 株価取得中: {d}（取引日 {collected}/{min_trading}）"
                )

        except RuntimeError as e:
            if "429" in str(e):
                # レートリミット: cur を進めず同じ日を再試行
                wait = 30
                logger.warning(f"429 レートリミット（{d}）。{wait}秒待機後に再試行...")
                if progress_callback:
                    progress_callback(
                        collected, min_trading,
                        f"⏳ レートリミット中  {wait}秒待機します"
                    )
                time.sleep(wait)
                tried_trading -= 1  # この試行はカウントしない
            else:
                logger.warning(f"株価取得エラー（{d}）: {str(e)[:80]}")
                cur -= timedelta(days=1)

        time.sleep(jq.REQUEST_INTERVAL)

    if not all_dfs:
        logger.warning("株価データを1日分も取得できませんでした")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    if "Date" not in combined.columns:
        combined["Date"] = combined["_date"]
    logger.info(f"株価取得完了: {collected}取引日, {len(combined):,}行")
    return combined




# ─────────────────────────────────────────────────────────
# リサンプリング（週足・月足）
# ─────────────────────────────────────────────────────────

def resample_to_period(daily_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """日次データを週足（W-FRI）または月足（ME）にリサンプリングする。"""
    freq = "W-FRI" if timeframe == "週足" else "ME"
    cc = _close_col(daily_df)
    vc = _vol_col(daily_df)

    df = daily_df.copy()
    date_col = "Date" if "Date" in df.columns else "_date"
    df["_dt"] = pd.to_datetime(df[date_col].astype(str), format="mixed", dayfirst=False)
    df[cc] = pd.to_numeric(df[cc], errors="coerce")
    if vc:
        df[vc] = pd.to_numeric(df[vc], errors="coerce")

    result_dfs = []
    for code, grp in df.groupby("Code"):
        grp = grp.set_index("_dt").sort_index()
        resampled = pd.DataFrame({"Close": grp[cc].resample(freq).last()})
        if vc:
            resampled["Volume"] = grp[vc].resample(freq).sum()
        resampled = resampled.dropna(subset=["Close"])
        resampled["Code"] = code
        result_dfs.append(resampled.reset_index().rename(columns={"_dt": "Date"}))

    return pd.concat(result_dfs, ignore_index=True) if result_dfs else pd.DataFrame()


# ─────────────────────────────────────────────────────────
# 連続上昇判定
# ─────────────────────────────────────────────────────────

def count_consecutive_rising(close_series: pd.Series) -> int:
    """終値の連続上昇本数を直近から逆順で数える。"""
    vals = pd.to_numeric(close_series, errors="coerce").dropna().values
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
    period_df: pd.DataFrame,
    n_periods: int,
    close_col: str,
) -> pd.DataFrame:
    """n_periods 以上連続上昇している銘柄を抽出する。"""
    results = []
    vc = _vol_col(period_df)
    for code, grp in period_df.groupby("Code"):
        grp = grp.sort_values("Date")
        cnt = count_consecutive_rising(grp[close_col])
        if cnt >= n_periods:
            latest = grp.iloc[-1]
            results.append({
                "Code": str(code),
                "株価": pd.to_numeric(latest[close_col], errors="coerce"),
                "出来高": pd.to_numeric(latest.get(vc, np.nan), errors="coerce") if vc else np.nan,
                "連続上昇数": cnt,
            })
    return pd.DataFrame(results) if results else pd.DataFrame()


# ─────────────────────────────────────────────────────────
# 銘柄情報マージ・フィルター
# ─────────────────────────────────────────────────────────

def merge_listed_info(screened: pd.DataFrame, listed: pd.DataFrame) -> pd.DataFrame:
    """
    スクリーニング結果に銘柄名・業種をマージする。

    V2 カラム: CoName / S33Nm / MktNm / ScaleCat
    → 内部で V1 互換名（CompanyName / Sector33CodeName / MarketCodeName）に統一する。
    ※ V2 に Shares（株式数）なし → 時価総額は N/A
    """
    li = listed.copy()
    sc = screened.copy()

    # V2 略称 → 内部統一名
    li = li.rename(columns={
        "CoName": "CompanyName",
        "S33Nm":  "Sector33CodeName",
        "MktNm":  "MarketCodeName",
    })

    li["CodeKey"] = li["Code"].astype(str).str[:4]
    sc["CodeKey"] = sc["Code"].astype(str).str[:4]

    info_cols = ["CodeKey"]
    for c in ["CompanyName", "Sector33CodeName", "MarketCodeName", "ScaleCat",
              "Shares", "MarketCapitalization"]:
        if c in li.columns:
            info_cols.append(c)

    merged = sc.merge(li[info_cols].drop_duplicates("CodeKey"), on="CodeKey", how="left")

    # 時価総額（データがある場合のみ計算）
    if "MarketCapitalization" in merged.columns:
        merged["時価総額(億円)"] = (
            pd.to_numeric(merged["MarketCapitalization"], errors="coerce") / 1e8
        ).round(1)
    elif "Shares" in merged.columns:
        merged["時価総額(億円)"] = (
            pd.to_numeric(merged["Shares"], errors="coerce") * merged["株価"] / 1e8
        ).round(1)
    else:
        merged["時価総額(億円)"] = np.nan  # V2では計算不可

    return merged


def apply_filters(df: pd.DataFrame,
                  min_price: float, max_price: float,
                  min_cap: float, max_cap: float) -> pd.DataFrame:
    """株価・時価総額フィルター（max=0 は上限なし）。"""
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
    """指定銘柄の PER・PBR を財務サマリーから取得する。"""
    rows = []
    for i, code in enumerate(codes):
        if progress_callback:
            progress_callback(i + 1, len(codes),
                              f"📊 PER/PBR取得中: {code} ({i+1}/{len(codes)})")
        df = jq.get_fins_summary(api_key, code)
        per, pbr = None, None
        if not df.empty:
            for dc in ["DiscDate", "DisclosedDate", "Date"]:
                if dc in df.columns:
                    df = df.sort_values(dc, ascending=False)
                    break
            for col in PER_COLS:
                if col in df.columns:
                    v = pd.to_numeric(df[col].iloc[0], errors="coerce")
                    if pd.notna(v) and v > 0:
                        per = round(float(v), 2)
                        break
            for col in PBR_COLS:
                if col in df.columns:
                    v = pd.to_numeric(df[col].iloc[0], errors="coerce")
                    if pd.notna(v) and v > 0:
                        pbr = round(float(v), 2)
                        break
        rows.append({"CodeKey": str(code)[:4], "PER": per, "PBR": pbr})
        time.sleep(jq.REQUEST_INTERVAL)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────
# メインスクリーニング
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
    """連続上昇銘柄スクリーニングを実行して結果 DataFrame を返す。"""

    # STEP 1: 株価取得
    price_data = fetch_price_data(api_key, timeframe, n_periods, progress_callback)
    if price_data.empty:
        return pd.DataFrame()

    # STEP 2: リサンプリング
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

    # STEP 7: 出力整形
    unit = {"日足": "日", "週足": "週", "月足": "ヶ月"}[timeframe]
    merged["連続上昇期間"] = merged["連続上昇数"].apply(lambda x: f"{x} {unit}")
    merged["銘柄コード"] = merged["CodeKey"]
    merged = merged.rename(columns={
        "CompanyName": "銘柄名",
        "Sector33CodeName": "業種",
        "MarketCodeName": "市場",
        "株価": "株価(円)",
    })

    cols = ["銘柄コード", "銘柄名", "業種", "市場", "株価(円)",
            "時価総額(億円)", "出来高", "PER", "PBR", "連続上昇期間"]
    cols = [c for c in cols if c in merged.columns]
    return merged[cols].sort_values("連続上昇期間", ascending=False).reset_index(drop=True)