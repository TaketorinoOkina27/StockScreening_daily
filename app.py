"""
連続上昇銘柄スクリーナー（J-Quants API V2対応）
起動: streamlit run app.py
"""

import logging
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from api.jquants import get_listed_info
from logic.screener import FREE_PLAN_DELAY_DAYS, run_screening

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

st.set_page_config(
    page_title="連続上昇銘柄スクリーナー",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── セッション状態 ─────────────────────────────────────────
_DEFAULTS = {
    "api_key":          None,
    "listed_info":      None,
    "screening_result": None,
    "last_run_params":  {},
    "connect_error":    None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def is_logged_in() -> bool:
    return (
        st.session_state["api_key"] is not None
        and st.session_state["listed_info"] is not None
    )


# ── サイドバー ─────────────────────────────────────────────
with st.sidebar:
    st.header("🔐 J-Quants 接続設定")

    if is_logged_in():
        info_df: pd.DataFrame = st.session_state["listed_info"]
        st.success("✅ 接続済み")
        st.caption(f"取得銘柄数: {len(info_df):,} 件")
        if st.button("切断", use_container_width=True):
            for k in _DEFAULTS:
                st.session_state[k] = _DEFAULTS[k]
            st.rerun()

    else:
        st.markdown(
            "**J-Quants APIキー**を入力してください。\n\n"
            "取得場所: [J-Quantsダッシュボード](https://jpx-jquants.com/dashboard/api-keys)"
            " → 設定 → APIキー"
        )
        api_key_input = st.text_input(
            "APIキー",
            type="password",
            placeholder="api key",
            key="api_key_input",
        )
        connect_btn = st.button("接続する", use_container_width=True, type="primary")

        if connect_btn:
            st.session_state["connect_error"] = None
            if not api_key_input or not api_key_input.strip():
                st.error("APIキーを入力してください。")
            else:
                with st.spinner("銘柄一覧を取得中..."):
                    try:
                        # get_listed_info は DataFrame を直接返す（タプルではない）
                        df = get_listed_info(api_key_input.strip())
                        st.session_state["api_key"] = api_key_input.strip()
                        st.session_state["listed_info"] = df
                        st.success(f"接続成功！（{len(df):,}銘柄）")
                        st.rerun()

                    except RuntimeError as e:
                        st.session_state["connect_error"] = str(e)

                    except Exception as e:
                        st.session_state["connect_error"] = (
                            f"予期しないエラー: {type(e).__name__}: {e}"
                        )

        if st.session_state.get("connect_error"):
            st.error("❌ 接続に失敗しました")
            with st.expander("▶ エラー詳細（クリックで展開）", expanded=True):
                st.code(st.session_state["connect_error"], language="text")

        with st.expander("🔑 APIキーの取得方法"):
            st.markdown(
                """
                1. [J-Quants公式サイト](https://jpx-jquants.com/) でアカウント登録
                2. **Freeプラン**（無料）を選択
                3. ダッシュボード → **設定 → APIキー** で発行
                4. 発行されたキーをここに貼り付け
                """
            )

    st.divider()

    # ── スクリーニング条件 ──────────────────────────────────
    st.header("🔍 スクリーニング条件")

    timeframe = st.selectbox("足種", ["日足", "週足", "月足"])
    max_n = {"日足": 60, "週足": 52, "月足": 24}
    default_n = {"日足": 5, "週足": 4, "月足": 3}

    n_periods = st.number_input(
        f"連続上昇期間（{timeframe}）",
        min_value=2, max_value=max_n[timeframe],
        value=default_n[timeframe], step=1,
    )

    st.subheader("フィルター")
    c1, c2 = st.columns(2)
    with c1:
        min_price = st.number_input("株価 最小(円)", min_value=0, value=0, step=100)
    with c2:
        max_price = st.number_input("株価 最大(円)", min_value=0, value=0, step=100,
                                    help="0=上限なし")
    c3, c4 = st.columns(2)
    with c3:
        min_cap = st.number_input("時価総額 最小(億円)", min_value=0, value=0, step=10,
                                   help="※V2 APIでは時価総額データがないためフィルター無効")
    with c4:
        max_cap = st.number_input("時価総額 最大(億円)", min_value=0, value=0, step=10,
                                   help="0=上限なし ※V2 APIでは無効")

    include_per_pbr = st.checkbox("PER・PBRも取得する", value=False,
                                   help="ONにすると各銘柄の財務データを追加取得します")

    st.divider()
    run_btn = st.button(
        "🚀 スクリーニング開始",
        use_container_width=True, type="primary",
        disabled=not is_logged_in(),
    )


# ── メインエリア ───────────────────────────────────────────
st.title("📈 連続上昇銘柄スクリーナー")
data_as_of = (date.today() - timedelta(days=FREE_PLAN_DELAY_DAYS)).strftime("%Y年%m月%d日")
st.caption(f"🕐 J-Quants 無料プラン（12週間遅延）｜参照データ: **{data_as_of}** 頃まで")
st.divider()

# ── スクリーニング実行 ─────────────────────────────────────
if run_btn:
    st.session_state["screening_result"] = None

    status_box = st.status("スクリーニング実行中...", expanded=True)
    progress_bar = st.progress(0.0)
    elapsed_text = st.empty()
    start_ts = time.time()

    def update_progress(current: int, total: int, msg: str) -> None:
        progress_bar.progress(min(current / max(total, 1), 1.0))
        elapsed_text.caption(f"⏱ {time.time() - start_ts:.0f}秒")
        with status_box:
            st.write(msg)

    try:
        result = run_screening(
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
        st.session_state["screening_result"] = result
        st.session_state["last_run_params"] = {
            "timeframe": timeframe, "n_periods": n_periods,
        }
        progress_bar.progress(1.0)
        elapsed_text.caption(f"✅ 完了（{time.time() - start_ts:.0f}秒）")
        status_box.update(label="スクリーニング完了！", state="complete")

    except Exception as exc:
        status_box.update(label="エラー発生", state="error")
        st.error(f"❌ スクリーニング中にエラーが発生しました:\n\n{exc}")
        logging.exception("スクリーニングエラー")

# ── 結果表示 ───────────────────────────────────────────────
if st.session_state["screening_result"] is not None:
    result: pd.DataFrame = st.session_state["screening_result"]
    params = st.session_state.get("last_run_params", {})
    tf = params.get("timeframe", "")
    np_ = params.get("n_periods", "")

    if result.empty:
        st.warning("⚠️ 条件に合致する銘柄が見つかりませんでした。条件を緩めて再試行してください。")
    else:
        st.success(f"✅ **{tf} {np_}期間連続上昇** ｜ 該当: **{len(result):,} 件**")

        disp = result.copy()
        fmt = {
            "株価(円)":      lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A",
            "時価総額(億円)": lambda x: f"{x:,.1f}" if pd.notna(x) else "N/A",
            "出来高":        lambda x: f"{int(x):,}" if pd.notna(x) and not np.isnan(float(x)) else "N/A",
            "PER":           lambda x: f"{x:.2f}倍" if pd.notna(x) else "N/A",
            "PBR":           lambda x: f"{x:.2f}倍" if pd.notna(x) else "N/A",
        }
        for col, fn in fmt.items():
            if col in disp.columns:
                disp[col] = disp[col].apply(fn)

        st.dataframe(disp, use_container_width=True,
                     height=min(600, 56 + 36 * len(disp)), hide_index=True)

        csv = result.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            "📥 CSVダウンロード", csv,
            f"screening_{tf}_{np_}period_{date.today():%Y%m%d}.csv",
            mime="text/csv",
        )

elif not is_logged_in():
    st.info("👈 左のサイドバーからAPIキーを入力して接続してください。")
    with st.expander("📖 使い方"):
        st.markdown("""
        1. [J-Quants公式サイト](https://jpx-jquants.com/) で無料登録 → Freeプラン選択
        2. ダッシュボード → 設定 → **APIキー** を発行
        3. サイドバーにAPIキーを貼り付けて「接続する」
        4. 足種・期間・フィルターを設定して「スクリーニング開始」
        5. 結果を確認・CSVダウンロード

        > ⚠️ 無料プランではデータは約12週間遅延します
        """)