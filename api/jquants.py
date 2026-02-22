"""
J-Quants API V2 呼び出しモジュール
====================================
2025年12月リリースの V2 API に対応。
認証方式: x-api-key ヘッダー（APIキー方式）

V1（メール/パスワード→トークン方式）は廃止予定のため、
V2 API キー方式のみをサポートする。

J-Quants ダッシュボードからAPIキーを取得してください。
https://jpx-jquants.com/dashboard/api-keys

V2主要エンドポイント（Freeプランで利用可能）:
  - GET /v2/equities/list        銘柄一覧（業種・時価総額等）
  - GET /v2/equities/bars/daily  株価四本値（日次）
  - GET /v2/fins/summary         財務サマリー（PER・PBR等）

レスポンス形式: { "data": [...], "pagination_key": "..." }
"""

import logging
import time
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── エンドポイント定数 ─────────────────────────────
BASE_URL_V2 = "https://api.jquants.com/v2"
ENDPOINT_LIST       = f"{BASE_URL_V2}/equities/list"        # 銘柄一覧
ENDPOINT_BARS_DAILY = f"{BASE_URL_V2}/equities/bars/daily"  # 株価日次
ENDPOINT_FIN_SUMMARY = f"{BASE_URL_V2}/fins/summary"        # 財務サマリー

# APIリクエスト間のスリープ（レートリミット対策）
REQUEST_INTERVAL = 0.2
# リトライ待機（秒 × attempt番号）
RETRY_WAIT_SEC = 1.0


def _make_headers(api_key: str) -> dict[str, str]:
    """
    V2 API 認証ヘッダーを生成する。

    Args:
        api_key: J-Quants ダッシュボードで発行したAPIキー

    Returns:
        x-api-key ヘッダーを含む辞書
    """
    return {"x-api-key": api_key}


def validate_api_key(api_key: str) -> bool:
    """
    APIキーの有効性を /equities/list への試験リクエストで確認する。

    Args:
        api_key: 確認対象のAPIキー

    Returns:
        True: 有効なAPIキー / False: 無効（認証失敗）

    Raises:
        requests.exceptions.ConnectionError: ネットワーク接続エラー時
    """
    headers = _make_headers(api_key)
    resp = requests.get(
        ENDPOINT_LIST,
        headers=headers,
        params={"date": "20240101"},  # 固定日付で試験
        timeout=15,
    )
    # 401/403 = 認証失敗、200 = 成功
    return resp.status_code == 200


# ─────────────────────────────────────────────────────────
# 銘柄情報取得
# ─────────────────────────────────────────────────────────

def get_listed_info(api_key: str, retries: int = 3) -> pd.DataFrame:
    """
    上場銘柄一覧を取得する（V2: /equities/list）。
    ページネーション対応（全ページを結合して返す）。

    V2 主要カラム（略称 → 意味）:
        Code          : 銘柄コード
        CompanyName   : 銘柄名
        Sector33Code  : 33業種コード
        Sector33CodeName : 33業種名
        MarketCode    : 市場区分コード
        MarketCodeName : 市場区分名
        Shares        : 上場株式数

    Args:
        api_key: J-Quants APIキー
        retries: リトライ回数

    Returns:
        銘柄情報DataFrame（取得失敗時は空のDataFrame）
    """
    headers = _make_headers(api_key)
    all_rows: list[dict] = []
    pagination_key: Optional[str] = None

    while True:
        params: dict[str, str] = {}
        if pagination_key:
            params["pagination_key"] = pagination_key

        for attempt in range(retries):
            try:
                resp = requests.get(
                    ENDPOINT_LIST,
                    headers=headers,
                    params=params,
                    timeout=30,
                )
                resp.raise_for_status()
                body = resp.json()
                rows = body.get("data", body.get("info", []))  # V2="data", V1互換="info"
                all_rows.extend(rows)
                pagination_key = body.get("pagination_key")
                break
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    wait = RETRY_WAIT_SEC * (attempt + 1)
                    logger.warning(f"銘柄一覧取得失敗 attempt {attempt+1}: {e}, {wait}秒後リトライ")
                    time.sleep(wait)
                else:
                    logger.error(f"銘柄一覧取得を断念: {e}")
                    return pd.DataFrame()

        if not pagination_key:
            break
        time.sleep(REQUEST_INTERVAL)

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────
# 株価データ取得
# ─────────────────────────────────────────────────────────

def get_daily_quotes_by_date(
    api_key: str,
    date_str: str,
    retries: int = 3,
) -> pd.DataFrame:
    """
    指定日の全銘柄株価データを取得する（V2: /equities/bars/daily）。
    ページネーション対応。

    V2 主要カラム（略称 → 意味）:
        Code  : 銘柄コード
        Date  : 日付
        O     : 始値
        H     : 高値
        L     : 安値
        C     : 終値
        Vo    : 出来高
        AdjO  : 修正後始値
        AdjH  : 修正後高値
        AdjL  : 修正後安値
        AdjC  : 修正後終値（株式分割等を考慮）
        AdjVo : 修正後出来高

    Args:
        api_key: J-Quants APIキー
        date_str: 取得日付（YYYYMMDD形式 または YYYY-MM-DD形式）
        retries: リトライ回数

    Returns:
        株価DataFrame（非営業日や取得失敗時は空のDataFrame）
    """
    headers = _make_headers(api_key)
    all_rows: list[dict] = []
    pagination_key: Optional[str] = None

    while True:
        params: dict[str, str] = {"date": date_str}
        if pagination_key:
            params["pagination_key"] = pagination_key

        for attempt in range(retries):
            try:
                resp = requests.get(
                    ENDPOINT_BARS_DAILY,
                    headers=headers,
                    params=params,
                    timeout=30,
                )
                resp.raise_for_status()
                body = resp.json()
                # V2 = "data", V1互換 = "daily_quotes"
                rows = body.get("data", body.get("daily_quotes", []))
                all_rows.extend(rows)
                pagination_key = body.get("pagination_key")
                break
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    wait = RETRY_WAIT_SEC * (attempt + 1)
                    logger.warning(
                        f"株価 {date_str} 取得失敗 attempt {attempt+1}: {e}, {wait}秒後リトライ"
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"株価 {date_str} の取得を断念: {e}")
                    return pd.DataFrame()

        if not pagination_key:
            break
        time.sleep(REQUEST_INTERVAL)

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────
# 財務データ取得（PER・PBR）
# ─────────────────────────────────────────────────────────

def get_fins_summary(
    api_key: str,
    code: str,
    retries: int = 3,
) -> pd.DataFrame:
    """
    財務サマリーデータを取得する（V2: /fins/summary）。
    PER・PBRの算出に使用する。

    Args:
        api_key: J-Quants APIキー
        code: 銘柄コード（4桁）
        retries: リトライ回数

    Returns:
        財務サマリーDataFrame（取得失敗時は空のDataFrame）
    """
    headers = _make_headers(api_key)
    params = {"code": code}

    for attempt in range(retries):
        try:
            resp = requests.get(
                ENDPOINT_FIN_SUMMARY,
                headers=headers,
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            body = resp.json()
            data = body.get("data", body.get("statements", []))
            return pd.DataFrame(data) if data else pd.DataFrame()
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(RETRY_WAIT_SEC * (attempt + 1))
            else:
                logger.warning(f"銘柄 {code} 財務サマリー取得失敗: {e}")
                return pd.DataFrame()

    return pd.DataFrame()
