"""
J-Quants API 呼び出しモジュール（V2 + V1フォールバック）
==========================================================
認証方式: x-api-key ヘッダー（V2 APIキー方式）

銘柄一覧は V2 → V1 の順でフォールバックを試みる。
エラーは必ず呼び出し元まで raise し、サイレントに握り潰さない。
"""

import logging
import time
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── エンドポイント ─────────────────────────────────────────
BASE_V1 = "https://api.jquants.com/v1"
BASE_V2 = "https://api.jquants.com/v2"

# 銘柄一覧: /v2/equities/master（dateパラメータ必須）
# 注意: /v2/equities/list は存在しない（403 "endpoint does not exist"）
ENDPOINT_LIST_V2     = f"{BASE_V2}/equities/master"     # V2 正しいエンドポイント
ENDPOINT_LIST_V1     = f"{BASE_V1}/listed/info"         # V1 フォールバック

# 株価・財務
ENDPOINT_BARS_DAILY  = f"{BASE_V2}/equities/bars/daily" # V2
ENDPOINT_FIN_SUMMARY = f"{BASE_V2}/fins/summary"        # V2

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
# 銘柄一覧取得（V2 → V1 フォールバック付き）
# ─────────────────────────────────────────────────────────

def get_listed_info(api_key: str) -> tuple[pd.DataFrame, str]:
    """
    上場銘柄一覧を取得する。
    V2(/v2/equities/master) を優先し、失敗した場合は V1(/v1/listed/info) にフォールバックする。

    V2エンドポイント: /v2/equities/master
      - dateパラメータが必須（省略するとエラー）
      - 本日日付を渡して当日時点の全銘柄を取得する
    V2主要カラム（V1と名称が異なる）:
      Code     : 銘柄コード（5桁）
      CoName   : 銘柄名（≒ V1の CompanyName）
      CoNameEn : 銘柄名（英語）
      S33      : 33業種コード
      S33Nm    : 33業種名（≒ V1の Sector33CodeName）
      S17      : 17業種コード
      S17Nm    : 17業種名
      Mkt      : 市場コード
      MktNm    : 市場区分名（≒ V1の MarketCodeName）
      ScaleCat : 規模区分（TOPIX Large70 / Mid400 / Small1 / Small2 / New Index）
      ※ Shares（上場株式数）カラムは存在しないため時価総額の計算不可

    Returns:
        (DataFrame, 使用したエンドポイントを示す文字列 "v2" | "v1")

    Raises:
        RuntimeError: V2・V1 ともに失敗した場合。詳細なエラーメッセージを含む。
    """
    from datetime import date as _date
    v2_error_msg: str = ""

    # ── V2 を試す ──────────────────────────────────────
    # /v2/equities/master は date パラメータが必須
    today_str = _date.today().strftime("%Y%m%d")
    try:
        df = _fetch_all_pages(
            url=ENDPOINT_LIST_V2,
            api_key=api_key,
            data_key_candidates=["data", "info"],
            extra_params={"date": today_str},
        )
        if not df.empty:
            logger.info(f"V2銘柄一覧取得成功: {len(df)}件 ({ENDPOINT_LIST_V2})")
            return df, "v2"
        # 200が返ったがデータが空 → V1を試す
        v2_error_msg = "V2エンドポイントは200を返したがデータが空でした。"
        logger.warning(v2_error_msg)

    except RuntimeError as e:
        v2_error_msg = str(e)
        # 認証エラー(401/403)はフォールバック不要
        if "HTTP 401" in v2_error_msg or "HTTP 403" in v2_error_msg:
            raise RuntimeError(
                f"認証エラー: APIキーが正しくないか、プランが未登録です。\n\n"
                f"詳細:\n{v2_error_msg}"
            )
        logger.warning(f"V2失敗。V1にフォールバック。V2エラー: {v2_error_msg[:200]}")

    # ── V1 フォールバック ─────────────────────────────
    try:
        df = _fetch_all_pages(
            url=ENDPOINT_LIST_V1,
            api_key=api_key,
            data_key_candidates=["info", "data"],
        )
        if not df.empty:
            logger.info(f"V1銘柄一覧取得成功: {len(df)}件 ({ENDPOINT_LIST_V1})")
            return df, "v1"
        v1_error_msg = "V1エンドポイントは200を返したがデータが空でした。"

    except RuntimeError as e:
        v1_error_msg = str(e)

    # ── 両方失敗 ─────────────────────────────────────
    raise RuntimeError(
        f"銘柄一覧の取得に失敗しました（V2・V1ともにエラー）\n\n"
        f"【V2エラー詳細】\n{v2_error_msg}\n\n"
        f"【V1フォールバックエラー詳細】\n{v1_error_msg}\n\n"
        f"確認事項:\n"
        f"  1. APIキーをJ-Quantsダッシュボードで再確認・再発行してください\n"
        f"  2. サブスクリプションプラン（Freeプラン含む）に登録済みか確認してください\n"
        f"  3. インターネット接続が正常か確認してください"
    )


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