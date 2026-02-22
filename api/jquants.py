"""
J-Quants API 呼び出しモジュール（V2専用）
==========================================
認証方式: x-api-key ヘッダー（V2 APIキー方式）

診断済み正常エンドポイント（2026-02時点）:
  GET /v2/equities/master     銘柄一覧（dateパラメータ必須）
  GET /v2/equities/bars/daily 株価四本値
  GET /v2/fins/summary        財務サマリー

V1（/v1/...）はAPIキー認証非対応のため使用しない。
エラーは必ず呼び出し元まで raise し、サイレントに握り潰さない。
"""

import logging
import time
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── エンドポイント ─────────────────────────────────────────
BASE_V2 = "https://api.jquants.com/v2"

# 確定済み正しいエンドポイント（診断済み 2026-02）
ENDPOINT_LIST_V2     = f"{BASE_V2}/equities/master"     # 銘柄一覧（dateパラメータ必須）
ENDPOINT_BARS_DAILY  = f"{BASE_V2}/equities/bars/daily" # 株価四本値
ENDPOINT_FIN_SUMMARY = f"{BASE_V2}/fins/summary"        # 財務サマリー

REQUEST_INTERVAL = 0.2   # リクエスト間スリープ（レートリミット対策）
RETRY_WAIT_SEC   = 1.0   # リトライ待機基礎秒数


# ─────────────────────────────────────────────────────────
# ヘルパー
# ─────────────────────────────────────────────────────────

def _headers(api_key: str) -> dict:
    """V2 APIキー認証ヘッダーを返す。"""
    return {"x-api-key": api_key}


def _get_with_retry(
    url: str,
    headers: dict,
    params: dict,
    retries: int = 3,
    timeout: int = 30,
) -> requests.Response:
    """
    GETリクエストをリトライ付きで実行する。
    成功(200)のレスポンスを返す。失敗時は RuntimeError を raise する。
    RuntimeError のメッセージには HTTPステータスと APIレスポンス本文を含める。
    """
    last_error: str = ""

    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)

            if resp.status_code == 200:
                return resp

            # --- エラー応答の詳細をメッセージに含める ---
            body_preview = resp.text[:400]
            last_error = (
                f"HTTP {resp.status_code} — URL: {url}\n"
                f"レスポンス本文: {body_preview}"
            )

            # 認証エラーはリトライ不要
            if resp.status_code in (401, 403):
                raise RuntimeError(
                    f"認証エラー (HTTP {resp.status_code})\n"
                    f"APIキーが正しいか、プランが登録済みか確認してください。\n"
                    f"詳細: {body_preview}"
                )

            # 404 = エンドポイントが存在しない
            if resp.status_code == 404:
                raise RuntimeError(
                    f"エンドポイントが見つかりません (HTTP 404) — {url}\n"
                    f"詳細: {body_preview}"
                )

        except RuntimeError:
            raise  # 上位へ伝播

        except requests.exceptions.ConnectionError as e:
            last_error = f"接続エラー: {e}"
        except requests.exceptions.Timeout:
            last_error = f"タイムアウト ({timeout}秒): {url}"
        except requests.exceptions.RequestException as e:
            last_error = f"リクエストエラー: {e}"

        if attempt < retries - 1:
            wait = RETRY_WAIT_SEC * (attempt + 1)
            logger.warning(f"リトライ {attempt+1}/{retries}（{wait}秒後）: {last_error}")
            time.sleep(wait)

    raise RuntimeError(
        f"最大リトライ回数 ({retries}) 超過\n"
        f"最後のエラー: {last_error}"
    )


def _fetch_all_pages(
    url: str,
    api_key: str,
    data_key_candidates: list[str],
    extra_params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    ページネーションを処理して全ページのデータを結合する。
    data_key_candidates に指定したキーを順番に探してデータを取り出す。
    """
    hdrs = _headers(api_key)
    all_rows: list[dict] = []
    pagination_key: Optional[str] = None

    while True:
        params: dict = dict(extra_params or {})
        if pagination_key:
            params["pagination_key"] = pagination_key

        resp = _get_with_retry(url, hdrs, params)
        body = resp.json()

        rows: list = []
        for key in data_key_candidates:
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
# 銘柄一覧取得
# ─────────────────────────────────────────────────────────

def get_listed_info(api_key: str) -> tuple[pd.DataFrame, str]:
    """
    上場銘柄一覧を取得する（V2: GET /v2/equities/master）。

    エンドポイント仕様:
      URL  : https://api.jquants.com/v2/equities/master
      認証 : x-api-key ヘッダー
      必須 : date または code パラメータのいずれか
             → 本関数では本日日付を自動付与して全銘柄を取得する

    V2レスポンスカラム:
      Code     : 銘柄コード（5桁）
      CoName   : 銘柄名
      CoNameEn : 銘柄名（英語）
      S33      : 33業種コード
      S33Nm    : 33業種名
      S17      : 17業種コード
      S17Nm    : 17業種名
      Mkt      : 市場コード
      MktNm    : 市場区分名
      ScaleCat : 規模区分
      ※ Shares（上場株式数）は含まれない → 時価総額の算出不可

    Returns:
        (DataFrame, "v2")  ※常にV2を使用

    Raises:
        RuntimeError: 取得失敗時。HTTPステータスとレスポンス本文を含む。
    """
    from datetime import date as _date
    today_str = _date.today().strftime("%Y%m%d")

    df = _fetch_all_pages(
        url=ENDPOINT_LIST_V2,
        api_key=api_key,
        data_key_candidates=["data"],
        extra_params={"date": today_str},
    )
    if df.empty:
        raise RuntimeError(
            "銘柄一覧が空でした（HTTP 200だがデータなし）。\n"
            "しばらく待ってから再試行してください。"
        )
    logger.info(f"銘柄一覧取得成功: {len(df)}件")
    return df, "v2"


# ─────────────────────────────────────────────────────────
# 株価データ取得
# ─────────────────────────────────────────────────────────

def get_daily_quotes_by_date(
    api_key: str,
    date_str: str,
) -> pd.DataFrame:
    """
    指定日の全銘柄株価データを取得する (V2: /equities/bars/daily)。
    非営業日など正常に空の場合は空DataFrameを返す。
    ネットワークエラーなど異常時もログを出して空DataFrameを返す（スクリーニング継続）。

    V2カラム: Code, Date, O, H, L, C, Vo, AdjO, AdjH, AdjL, AdjC, AdjVo
    """
    try:
        return _fetch_all_pages(
            url=ENDPOINT_BARS_DAILY,
            api_key=api_key,
            data_key_candidates=["data", "daily_quotes"],
            extra_params={"date": date_str},
        )
    except RuntimeError as e:
        # 非営業日=データなし(404 or 空) は正常。それ以外はログだけ出してスキップ
        logger.warning(f"株価 {date_str} スキップ: {str(e)[:120]}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────
# 財務データ取得（PER・PBR）
# ─────────────────────────────────────────────────────────

def get_fins_summary(api_key: str, code: str) -> pd.DataFrame:
    """
    財務サマリーを取得する (V2: /fins/summary)。
    取得失敗時は空DataFrameを返す（PER/PBRをN/Aとして扱う）。
    """
    try:
        return _fetch_all_pages(
            url=ENDPOINT_FIN_SUMMARY,
            api_key=api_key,
            data_key_candidates=["data", "statements"],
            extra_params={"code": code},
        )
    except RuntimeError as e:
        logger.warning(f"銘柄 {code} 財務サマリー取得失敗: {str(e)[:120]}")
        return pd.DataFrame()