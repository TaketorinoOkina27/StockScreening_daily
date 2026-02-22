#!/usr/bin/env python3
"""
J-Quants API エンドポイント診断スクリプト
==========================================
APIキーと各エンドポイントの動作を詳細にチェックし、
どのエンドポイントが使えてどれがエラーになるかを明確に表示する。

使い方:
    python diagnose.py
    # 実行後、APIキーの入力を求めるプロンプトが出ます

または環境変数で渡す場合:
    JQUANTS_API_KEY=xxxxxxxx python diagnose.py
"""

import json
import os
import sys
import time


try:
    import requests
except ImportError:
    print("❌ requests ライブラリが必要です: pip install requests")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════
# テスト対象エンドポイント一覧
# 各エンドポイントに対して GET リクエストを送り、
# HTTPステータスコード・レスポンス本文・カラム名を表示する
# ══════════════════════════════════════════════════════════════

ENDPOINTS_TO_TEST = [
    # ── V2 エンドポイント群 ───────────────────────────────
    {
        "label": "[V2] 銘柄一覧 (equities/master)",
        "url":   "https://api.jquants.com/v2/equities/master",
        "params": {"date": "20250101"},   # dateパラメータが必須
        "auth":  "v2_apikey",
        "data_keys": ["data", "info"],
        "note": "正しいV2エンドポイント。/v2/equities/listは存在しない（403）"
    },
    {
        "label": "[V2] 株価（1日・全銘柄）",
        "url":   "https://api.jquants.com/v2/equities/bars/daily",
        "params": {"date": "20241101"},
        "auth":  "v2_apikey",
        "data_keys": ["data", "daily_quotes"],
        "note": ""
    },
    {
        "label": "[V2] 財務サマリー",
        "url":   "https://api.jquants.com/v2/fins/summary",
        "params": {"code": "86970"},
        "auth":  "v2_apikey",
        "data_keys": ["data", "statements"],
        "note": ""
    },
    # ── 旧URL（誤り確認用） ─────────────────────────────────
    {
        "label": "[V2誤] 銘柄一覧 (equities/list) ※存在しない",
        "url":   "https://api.jquants.com/v2/equities/list",
        "params": {},
        "auth":  "v2_apikey",
        "data_keys": ["data"],
        "note": "このエンドポイントは存在しない（403 endpoint does not exist）"
    },
    # ── V1 エンドポイント群 ─────────────────────────────────
    {
        "label": "[V1] 銘柄一覧",
        "url":   "https://api.jquants.com/v1/listed/info",
        "params": {},
        "auth":  "v2_apikey",
        "data_keys": ["info", "data"],
        "note": "V1はAPIキー認証非対応（401が返る）"
    },
]


def _separator(title: str = "") -> None:
    width = 70
    if title:
        print(f"\n{'─' * 3} {title} {'─' * (width - 5 - len(title))}")
    else:
        print("─" * width)


def _status_emoji(code: int) -> str:
    if code == 200:
        return "✅"
    if code in (401, 403):
        return "🔑"   # 認証エラー
    if code == 404:
        return "❓"   # エンドポイント不存在
    if code == 429:
        return "⏳"   # レートリミット
    return "❌"


def test_endpoint(ep: dict, api_key: str) -> dict:
    """
    1エンドポイントをテストして結果を返す。

    Returns:
        {
            "label": str,
            "url": str,
            "status": int | None,
            "ok": bool,
            "row_count": int,
            "columns": list[str],
            "body_preview": str,
            "error": str,
        }
    """
    result = {
        "label":        ep["label"],
        "url":          ep["url"],
        "status":       None,
        "ok":           False,
        "row_count":    0,
        "columns":      [],
        "body_preview": "",
        "error":        "",
    }

    headers = {"x-api-key": api_key}

    try:
        resp = requests.get(
            ep["url"],
            headers=headers,
            params=ep["params"],
            timeout=20,
        )
        result["status"] = resp.status_code

        # レスポンスボディを安全に取得
        try:
            body = resp.json()
            body_str = json.dumps(body, ensure_ascii=False, indent=2)
        except Exception:
            body = {}
            body_str = resp.text

        # プレビュー（最初の 600 文字）
        result["body_preview"] = body_str[:600]

        if resp.status_code == 200:
            # データキーを探してレコード数・カラムを確認
            rows = []
            for key in ep["data_keys"]:
                if key in body:
                    rows = body[key]
                    break

            result["row_count"] = len(rows)
            if rows and isinstance(rows[0], dict):
                result["columns"] = list(rows[0].keys())

            result["ok"] = len(rows) > 0
        else:
            result["error"] = f"HTTP {resp.status_code}"

    except requests.exceptions.ConnectionError as e:
        result["error"] = f"接続失敗: {e}"
    except requests.exceptions.Timeout:
        result["error"] = "タイムアウト（20秒）"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def main() -> None:
    # ── APIキー取得 ─────────────────────────────────────
    api_key = os.environ.get("JQUANTS_API_KEY", "").strip()

    if not api_key:
        print("=" * 70)
        print("J-Quants API エンドポイント診断ツール")
        print("=" * 70)
        print()
        api_key = input("🔑 APIキーを入力してください: ").strip()

    if not api_key:
        print("❌ APIキーが空です。終了します。")
        sys.exit(1)

    masked = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "****"
    print()
    print("=" * 70)
    print(f"J-Quants API 診断開始  (APIキー: {masked})")
    print("=" * 70)

    results = []

    for ep in ENDPOINTS_TO_TEST:
        print(f"\n🔍 テスト中: {ep['label']}")
        print(f"   URL: {ep['url']}")
        if ep["params"]:
            print(f"   Params: {ep['params']}")

        r = test_endpoint(ep, api_key)
        results.append(r)

        emoji = _status_emoji(r["status"] or 0)
        if r["ok"]:
            print(f"   {emoji} HTTP {r['status']} → {r['row_count']:,}件取得成功")
            if r["columns"]:
                print(f"   📋 カラム: {', '.join(r['columns'][:10])}"
                      + (" ..." if len(r["columns"]) > 10 else ""))
        else:
            print(f"   {emoji} HTTP {r['status']} → 失敗: {r['error']}")
            print(f"   📄 レスポンス本文（先頭600文字）:")
            for line in r["body_preview"].splitlines():
                print(f"      {line}")

        time.sleep(0.3)   # 連続リクエスト防止

    # ── サマリー ─────────────────────────────────────────
    _separator()
    print("\n📊 診断サマリー")
    _separator()

    ok_count  = sum(1 for r in results if r["ok"])
    ng_count  = len(results) - ok_count

    print(f"  テスト数: {len(results)}  ✅成功: {ok_count}  ❌失敗: {ng_count}")
    print()

    for r in results:
        emoji = "✅" if r["ok"] else "❌"
        status_str = f"HTTP {r['status']}" if r["status"] else "接続不可"
        detail = f"{r['row_count']:,}件" if r["ok"] else r["error"]
        print(f"  {emoji}  {r['label']:<30} {status_str:<12} {detail}")

    # ── 推奨アクション ────────────────────────────────────
    _separator()
    print("\n💡 推奨アクション")
    _separator()

    all_401_403 = all(
        r["status"] in (401, 403) for r in results if r["status"] is not None
    )
    v2_list_ok = any(r["ok"] for r in results if r["label"] == "[V2] 銘柄一覧")
    v2_bars_ok = any(r["ok"] for r in results if r["label"] == "[V2] 株価（1日・全銘柄）")

    if all_401_403:
        print("""
  ❌ 全エンドポイントで認証エラー (401/403) が発生しています。
  → APIキーが正しくないか、期限切れの可能性があります。
  → 対処法:
     1. https://jpx-jquants.com/dashboard/api-keys でAPIキーを再発行
     2. サブスクリプションプランに登録済みか確認（Freeプラン含む）
     3. 再発行したキーをアプリに入力し直してください
        """)
    elif v2_list_ok and v2_bars_ok:
        print("""
  ✅ V2 APIが正常に動作しています。
  → アプリをそのまま使用できます。
        """)
    elif not v2_list_ok and not v2_bars_ok:
        v2_list = next((r for r in results if r["label"] == "[V2] 銘柄一覧"), {})
        print(f"""
  ❌ V2エンドポイントに問題があります。
  → /v2/equities/list の応答: HTTP {v2_list.get('status')} / {v2_list.get('error')}
  → 考えられる原因:
     ・APIキーが正しくない
     ・エンドポイントURLが変更されている可能性
     ・レートリミット超過（HTTP 429）
  → J-Quants公式ドキュメントで最新エンドポイントを確認してください:
     https://jpx-jquants.com/ja/spec/quickstart
        """)
    else:
        print("""
  ⚠️  一部のエンドポイントのみ動作しています。
  → 上記の詳細結果を確認して、失敗しているエンドポイントを特定してください。
        """)

    print()
    print("診断完了。この出力をそのまま共有して問題の特定に役立てられます。")
    print("=" * 70)


if __name__ == "__main__":
    main()