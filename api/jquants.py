"""
J-Quants API 呼び出しモジュール（V2専用）
==========================================
認証: x-api-key ヘッダー

確認済みエンドポイント:
  GET /v2/equities/master     銘柄一覧（dateパラメータ必須）
  GET /v2/equities/bars/daily 株価四本値
  GET /v2/fins/summary        財務サマリー
"""

import logging
import time
from datetime import date as _date, timedelta
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_V2              = "https://api.jquants.com/v2"
ENDPOINT_MASTER      = f"{BASE_V2}/equities/master"
ENDPOINT_BARS_DAILY  = f"{BASE_V2}/equities/bars/daily"
ENDPOINT_FIN_SUMMARY = f"{BASE_V2}/fins/summary"

REQUEST_INTERVAL     = 0.5  # リクエスト間スリープ秒（429対策：0.2→0.5に増加）
RETRY_WAIT_SEC       = 1.0
FREE_PLAN_DELAY_DAYS = 84   # Freeプランデータ遅延日数（12週間）


def _headers(api_key: str) -> dict:
    return {"x-api-key": api_key}


def _get_with_retry(url: str, headers: dict, params: dict,
                    retries: int = 3, timeout: int = 30) -> requests.Response:
    """
    GETリクエストをリトライ付きで実行する。
    400/401/403/404 は即 raise（リトライ無意味）。
    呼び出し元が 400 を受け取り、dateを変えて再試行する。
    """
    last_error = ""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp

            body = resp.text[:400]
            last_error = f"HTTP {resp.status_code} — {url}\nレスポンス: {body}"

            if resp.status_code == 429:
                # レートリミット → 待機してリトライ
                wait = 10.0 * (attempt + 1)
                logger.warning(f"429 レートリミット。{wait}秒待機後リトライ...")
                time.sleep(wait)
                continue
            if resp.status_code in (400, 401, 403, 404):
                raise RuntimeError(last_error)

        except RuntimeError:
            raise
        except requests.exceptions.RequestException as e:
            last_error = f"通信エラー: {e}"

        if attempt < retries - 1:
            time.sleep(RETRY_WAIT_SEC * (attempt + 1))

    raise RuntimeError(f"リトライ上限超過: {last_error}")


def _fetch_all_pages(url: str, api_key: str,
                     data_keys: list[str],
                     params: Optional[dict] = None) -> pd.DataFrame:
    """ページネーション対応 GET。全ページを結合して DataFrame を返す。"""
    hdrs = _headers(api_key)
    all_rows: list[dict] = []
    pagination_key: Optional[str] = None

    while True:
        p = dict(params or {})
        if pagination_key:
            p["pagination_key"] = pagination_key

        resp = _get_with_retry(url, hdrs, p)
        body = resp.json()

        rows: list = []
        for key in data_keys:
            if key in body:
                rows = body[key]
                break
        all_rows.extend(rows)

        pagination_key = body.get("pagination_key")
        if not pagination_key:
            break
        time.sleep(REQUEST_INTERVAL)

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────
# 公開関数
# ─────────────────────────────────────────────────────────

def get_listed_info(api_key: str) -> pd.DataFrame:
    """
    上場銘柄一覧を取得して DataFrame を返す（V2: /v2/equities/master）。
    Freeプランは本日から84日以上前のデータしか取得できないため、84日前を指定する。
    """
    target_date = (_date.today() - timedelta(days=84)).strftime("%Y%m%d")
    df = _fetch_all_pages(
        url=ENDPOINT_MASTER,
        api_key=api_key,
        data_keys=["data"],
        params={"date": target_date},
    )
    if df.empty:
        raise RuntimeError("銘柄一覧が空でした。しばらく待ってから再試行してください。")
    logger.info(f"銘柄一覧取得完了: {len(df)}件 (date={target_date})")
    return df


def get_daily_quotes_by_date(api_key: str, date_str: str) -> pd.DataFrame:
    """
    指定日の全銘柄株価を取得する（V2: /v2/equities/bars/daily）。

    戻り値:
      - データあり → DataFrame
      - 非営業日（HTTP 200でデータ空、または404）→ 空 DataFrame
    例外:
      - 429（レートリミット）→ RuntimeError を re-raise する。
        呼び出し元で待機処理を行うこと。
    """
    try:
        return _fetch_all_pages(
            url=ENDPOINT_BARS_DAILY,
            api_key=api_key,
            data_keys=["data"],
            params={"date": date_str},
        )
    except RuntimeError as e:
        if "429" in str(e):
            raise   # 429は呼び出し元に伝播させて待機処理を委ねる
        # 非営業日・404など → 空DataFrameで正常扱い
        logger.info(f"株価スキップ（非営業日等）: {date_str}")
        return pd.DataFrame()


def get_fins_summary(api_key: str, code: str) -> pd.DataFrame:
    """
    財務サマリーを取得する（V2: /v2/fins/summary）。
    失敗時は空 DataFrame を返す（PER/PBR を N/A 扱い）。
    """
    try:
        return _fetch_all_pages(
            url=ENDPOINT_FIN_SUMMARY,
            api_key=api_key,
            data_keys=["data"],
            params={"code": code},
        )
    except RuntimeError as e:
        logger.warning(f"財務サマリースキップ ({code}): {str(e)[:100]}")
        return pd.DataFrame()