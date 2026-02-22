"""
連続上昇銘柄スクリーナー（J-Quants API V2対応版）
=====================================================
V2 API（2025年12月リリース）に対応。
認証: APIキー方式（x-api-key ヘッダー）

起動方法:
    streamlit run app.py
"""

import logging
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from api.jquants import validate_api_key, get_listed_info
from logic.screener import FREE_PLAN_DELAY_DAYS, run_screening

# ──────────────────────────────────────────────────────────────────────
# ロギング設定
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

# ──────────────────────────────────────────────────────────────────────
# ページ設定（最初に呼び出す必要がある）
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="連続上昇銘柄スクリーナー",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# セッション状態の初期化
# ──────────────────────────────────────────────────────────────────────
_SESSION_DEFAULTS: dict = {
    "api_key": None,
    "listed_info": None,
    "screening_result": None,
    "last_run_params": {},
    "error_message": None,
}
for _key, _val in _SESSION_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


def is_logged_in() -> bool:
    """APIキーと銘柄情報が両方取得済みかを返す。"""
    return (
        st.session_state["api_key"] is not None
        and st.session_state["listed_info"] is not None
    )


# ──────────────────────────────────────────────────────────────────────
# サイドバー
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── 認証セクション ─────────────────────────────────
    st.header("🔐 J-Quants 接続設定")

    if is_logged_in():
        info_df: pd.DataFrame = st.session_state["listed_info"]
        st.success("✅ 接続済み")
        st.caption(f"取得銘柄数: {len(info_df):,} 件")
        if st.button("切断（ログアウト）", use_container_width=True):
            for k in _SESSION_DEFAULTS:
                st.session_state[k] = _SESSION_DEFAULTS[k]
            st.rerun()
    else:
        # ── APIキー入力（V2方式） ─────────────────────
        st.markdown(
            "**APIキー**（V2認証）を入力してください。\n\n"
            "取得場所: [J-Quantsダッシュボード](https://jpx-jquants.com/dashboard/api-keys)"
        )
        api_key_input = st.text_input(
            "APIキー",
            type="password",
            placeholder="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            key="api_key_input",
            help="J-Quantsダッシュボード の [設定 > APIキー] から発行・取得できます。",
        )

        connect_btn = st.button(
            "接続する",
            use_container_width=True,
            type="primary",
        )

        if connect_btn:
            if not api_key_input or api_key_input.strip() == "":
                st.error("APIキーを入力してください。")
            else:
                with st.spinner("APIキーを確認中..."):
                    try:
                        # APIキーの有効性チェック＋銘柄一覧取得
                        listed_info = get_listed_info(api_key_input.strip())

                        if listed_info.empty:
                            st.error(
                                "❌ APIキーが無効か、銘柄一覧を取得できませんでした。\n\n"
                                "・APIキーが正しいか確認してください。\n"
                                "・J-Quantsのサブスクリプションプランに登録済みか確認してください。"
                            )
                        else:
                            st.session_state["api_key"] = api_key_input.strip()
                            st.session_state["listed_info"] = listed_info
                            st.success(f"接続成功！（{len(listed_info):,}銘柄を取得）")
                            st.rerun()

                    except Exception as exc:
                        st.error(
                            f"❌ 接続エラー: {exc}\n\n"
                            "ネットワーク接続とAPIキーを確認してください。"
                        )

        # ── APIキーの取得方法ガイド ─────────────────────
        with st.expander("🔑 APIキーの取得方法"):
            st.markdown(
                """
                1. [J-Quants公式サイト](https://jpx-jquants.com/) でアカウント登録
                2. Freeプラン（無料）を選択して登録
                3. **ダッシュボード → 設定 → APIキー** からキーを発行
                4. 発行されたAPIキーをここに貼り付け

                > ⚠️ **注意**: 2025年12月以降のアカウントはV2 APIのみ対応です。
                > 旧バージョン（V1）のメール/パスワードログインは使用できません。
                """
            )

    st.divider()

    # ── スクリーニング条件 ───────────────────────────────
    st.header("🔍 スクリーニング条件")

    timeframe = st.selectbox(
        "足種",
        options=["日足", "週足", "月足"],
        index=0,
        help=(
            "日足: 当日終値 > 前営業日終値\n"
            "週足: その週の終値（金曜）> 前週の終値\n"
            "月足: その月の終値（最終営業日）> 前月の終値"
        ),
    )

    max_period_map = {"日足": 60, "週足": 52, "月足": 24}
    default_period_map = {"日足": 5, "週足": 4, "月足": 3}

    n_periods = st.number_input(
        f"連続上昇期間（{timeframe}）",
        min_value=2,
        max_value=max_period_map[timeframe],
        value=default_period_map[timeframe],
        step=1,
        help=f"N期間連続して終値が上昇している銘柄を抽出します（最大{max_period_map[timeframe]}）",
    )

    st.subheader("フィルター設定")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        min_price = st.number_input("株価 最小（円）", min_value=0, value=0, step=100)
    with col_p2:
        max_price = st.number_input(
            "株価 最大（円）", min_value=0, value=0, step=100, help="0 = 上限なし"
        )

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        min_cap = st.number_input("時価総額 最小（億円）", min_value=0, value=0, step=10)
    with col_c2:
        max_cap = st.number_input(
            "時価総額 最大（億円）", min_value=0, value=0, step=10, help="0 = 上限なし"
        )

    include_per_pbr = st.checkbox(
        "PER・PBRも取得する",
        value=False,
        help="ONにすると該当銘柄ごとに財務データを追加取得します。件数が多いと時間がかかります。",
    )

    st.divider()

    run_btn = st.button(
        "🚀 スクリーニング開始",
        use_container_width=True,
        type="primary",
        disabled=not is_logged_in(),
        help="先にAPIキーで接続してください。" if not is_logged_in() else "",
    )


# ──────────────────────────────────────────────────────────────────────
# メインエリア: タイトル
# ──────────────────────────────────────────────────────────────────────
st.title("📈 連続上昇銘柄スクリーナー")

data_as_of = (date.today() - timedelta(days=FREE_PLAN_DELAY_DAYS)).strftime(
    "%Y年%m月%d日"
)
st.caption(
    f"🕐 J-Quants 無料プラン（12週間遅延）使用　"
    f"参照データ: おおよそ **{data_as_of}** 頃まで"
)
st.divider()


# ──────────────────────────────────────────────────────────────────────
# スクリーニング実行
# ──────────────────────────────────────────────────────────────────────
if run_btn:
    st.session_state["screening_result"] = None
    st.session_state["error_message"] = None

    status_box = st.status("スクリーニングを実行中...", expanded=True)
    progress_bar = st.progress(0.0)
    elapsed_text = st.empty()
    start_ts = time.time()

    def update_progress(current: int, total: int, message: str) -> None:
        """スクリーニングロジックから呼ばれる進捗コールバック。"""
        ratio = current / max(total, 1)
        progress_bar.progress(min(ratio, 1.0))
        elapsed = time.time() - start_ts
        with status_box:
            st.write(message)
        elapsed_text.caption(f"⏱ 経過時間: {elapsed:.0f}秒")

    try:
        result_df = run_screening(
            api_key=st.session_state["api_key"],
            listed_info=st.session_state["listed_info"],
            timeframe=timeframe,
            n_periods=int(n_periods),
            min_price=float(min_price),
            max_price=float(max_price),
            min_cap_oku=float(min_cap),
            max_cap_oku=float(max_cap),
            include_per_pbr=include_per_pbr,
            progress_callback=update_progress,
        )
        st.session_state["screening_result"] = result_df
        st.session_state["last_run_params"] = {
            "timeframe": timeframe,
            "n_periods": n_periods,
        }
        progress_bar.progress(1.0)
        total_elapsed = time.time() - start_ts
        elapsed_text.caption(f"✅ 完了（所要時間: {total_elapsed:.0f}秒）")
        status_box.update(label="スクリーニング完了！", state="complete")

    except Exception as exc:
        st.session_state["error_message"] = str(exc)
        status_box.update(label="エラーが発生しました", state="error")
        logging.exception("スクリーニング中に例外が発生")


# ──────────────────────────────────────────────────────────────────────
# エラー表示
# ──────────────────────────────────────────────────────────────────────
if st.session_state["error_message"]:
    st.error(
        f"❌ エラーが発生しました:\n\n{st.session_state['error_message']}\n\n"
        "**確認事項:**\n"
        "- APIキーが正しいか\n"
        "- インターネット接続が有効か\n"
        "- J-Quantsのプランが有効か（Freeプランを含む）"
    )


# ──────────────────────────────────────────────────────────────────────
# 結果表示
# ──────────────────────────────────────────────────────────────────────
if st.session_state["screening_result"] is not None:
    result: pd.DataFrame = st.session_state["screening_result"]
    params = st.session_state.get("last_run_params", {})

    if result.empty:
        st.warning(
            "⚠️ 条件に合致する銘柄が見つかりませんでした。\n"
            "連続上昇期間を短くするか、フィルター条件を緩めて再試行してください。"
        )
    else:
        tf = params.get("timeframe", "")
        np_ = params.get("n_periods", "")
        st.success(
            f"✅ **{tf} {np_}期間連続上昇** ｜ 該当銘柄: **{len(result):,} 件**"
        )

        # 表示用フォーマット（元のDataFrameは変えない）
        display_df = result.copy()

        fmt_map = {
            "株価(円)":    lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A",
            "時価総額(億円)": lambda x: f"{x:,.1f}" if pd.notna(x) else "N/A",
            "出来高":      lambda x: f"{int(x):,}" if pd.notna(x) and not np.isnan(float(x)) else "N/A",
            "PER":         lambda x: f"{x:.2f}倍" if pd.notna(x) else "N/A",
            "PBR":         lambda x: f"{x:.2f}倍" if pd.notna(x) else "N/A",
        }
        for col, fmt in fmt_map.items():
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: fmt(x) if x != "" else "N/A"
                )

        st.dataframe(
            display_df,
            use_container_width=True,
            height=min(600, 56 + 36 * len(display_df)),
            hide_index=True,
        )

        # CSV ダウンロード（数値形式の生データで出力）
        csv_data = result.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 CSVダウンロード",
            data=csv_data,
            file_name=f"screening_{tf}_{np_}period_{date.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

elif not is_logged_in():
    # ログイン前の案内
    st.info(
        "👈 左のサイドバーから **J-Quants APIキー** を入力して接続してください。\n\n"
        "アカウントをお持ちでない場合は [J-Quants公式サイト](https://jpx-jquants.com/) から無料登録できます。"
    )

    with st.expander("📖 使い方"):
        st.markdown(
            """
            1. **APIキー取得**: J-Quantsダッシュボード → 設定 → APIキー から発行
            2. **接続**: サイドバーにAPIキーを入力して「接続する」をクリック
            3. **条件設定**: 足種・連続上昇期間・フィルターを設定
            4. **実行**: 「スクリーニング開始」をクリック
            5. **結果確認**: 結果テーブルを確認・CSVダウンロード

            > ⚠️ **V2 APIについて**: 2025年12月以降に登録したアカウントはAPIキー方式（V2）のみ対応です。
            > 旧来のメール/パスワードログイン（V1）は使用できません。
            >
            > ⚠️ **無料プランの制限**: データは約12週間遅延して配信されます。
            """
        )

    with st.expander("⚠️ 旧バージョン（V1）からの移行について"):
        st.markdown(
            """
            **「Missing Authentication Token」エラーが発生していた方へ**

            このエラーはJ-Quants APIのV2移行（2025年12月）により、
            旧来のメール/パスワードによる認証方式が廃止されたことが原因です。

            **解決方法**:
            1. [J-Quantsダッシュボード](https://jpx-jquants.com/dashboard/api-keys) にログイン
            2. 「設定」→「APIキー」からAPIキーを発行
            3. 発行されたAPIキーをサイドバーに入力して接続

            V1 APIは並走期間中は引き続き使用できますが、廃止日が決まり次第アナウンスされます。
            早めにV2（APIキー方式）への移行をお勧めします。
            """
        )
