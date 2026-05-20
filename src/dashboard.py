# src/dashboard.py
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_METRICS = ROOT / "results/eval_metrics_1_50.json"
DEFAULT_QUERY_REPORT = ROOT / "results/eval_query_report_1_50.csv"
DEFAULT_EVENTS = ROOT / "results/eval_quarantine_events_1_50.csv"
DEFAULT_QUARANTINE_DB = ROOT / "data/quarantine.sqlite3"


st.set_page_config(
    page_title="RAG Assembly Detector",
    page_icon="RAG",
    layout="wide",
)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_quarantine_db(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(str(path))
        df = pd.read_sql_query(
            """
            SELECT doc_id, label, status, reason, query, created_at
            FROM quarantined_documents
            ORDER BY created_at DESC
            """,
            conn,
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def metric_card(label: str, value, help_text: str | None = None):
    st.metric(label=label, value=value, help=help_text)


def main() -> None:
    st.title("RAG Assembly Detector Dashboard")
    st.caption("Top-k 조립 문맥 탐지, 격리, 재검색, 평가 결과를 확인하는 대시보드")

    with st.sidebar:
        st.header("Data Sources")

        metrics_path = Path(st.text_input("Metrics JSON", str(DEFAULT_METRICS)))
        query_report_path = Path(st.text_input("Query Report CSV", str(DEFAULT_QUERY_REPORT)))
        events_path = Path(st.text_input("Quarantine Events CSV", str(DEFAULT_EVENTS)))
        quarantine_db_path = Path(st.text_input("Quarantine DB", str(DEFAULT_QUARANTINE_DB)))

        st.divider()
        st.caption("결과 파일이 없으면 먼저 평가 스크립트를 실행해야 합니다.")

    metrics = load_json(metrics_path)
    query_df = load_csv(query_report_path)
    events_df = load_csv(events_path)
    quarantine_df = load_quarantine_db(quarantine_db_path)

    if not metrics and query_df.empty and events_df.empty:
        st.warning(
            "평가 결과 파일이 없습니다. 먼저 scripts/evaluate_queries.py 와 "
            "scripts/analyze_eval_results.py 를 실행하세요."
        )
        st.code(
            "python3 scripts/analyze_eval_results.py "
            "--input results/eval_summary_1_50_fast.csv",
            language="bash",
        )
        return

    st.subheader("Overview")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Queries", metrics.get("total_queries", len(query_df)))
    with c2:
        metric_card("Answered", metrics.get("answered_queries", "-"))
    with c3:
        metric_card("Final Clean", metrics.get("final_clean_queries", "-"))
    with c4:
        attack_total = metrics.get("attack_queries", "-")
        attack_success = metrics.get("query_level_attack_success_count", "-")
        metric_card("Attack Success", f"{attack_success}/{attack_total}")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        metric_card(
            "Normal Query Quarantine",
            metrics.get("normal_query_quarantine_event_count", "-"),
            "정상 질의에서 격리 이벤트가 발생한 횟수",
        )
    with c6:
        metric_card(
            "Normal Doc False Positive",
            metrics.get("document_level_false_positive_normal_doc_count", "-"),
            "정상 문서를 잘못 격리한 수",
        )
    with c7:
        metric_card(
            "Quarantined Attack Docs",
            metrics.get("quarantined_attack_docs_count", "-"),
        )
    with c8:
        metric_card(
            "Blocked/Error",
            metrics.get("blocked_or_error_queries", "-"),
        )

    st.divider()

    tab_summary, tab_queries, tab_events, tab_db, tab_raw = st.tabs(
        [
            "Summary",
            "Queries",
            "Quarantine Events",
            "Quarantine DB",
            "Raw Metrics",
        ]
    )

    with tab_summary:
        st.subheader("Detection Summary")

        left, right = st.columns(2)

        with left:
            st.markdown("#### Normal Query Quarantine Events")
            normal_events = metrics.get("normal_query_quarantine_event_query_ids", [])
            if normal_events:
                st.write(normal_events)
            else:
                st.success("없음")

            st.markdown("#### False Positive Normal Documents")
            fp_docs = metrics.get("document_level_false_positive_normal_doc_ids", [])
            if fp_docs:
                st.error(fp_docs)
            else:
                st.success("정상 문서 오탐 없음")

        with right:
            st.markdown("#### Attack Document Misses")
            misses = metrics.get("attack_doc_misses", [])
            if misses:
                st.dataframe(pd.DataFrame(misses), use_container_width=True)
            else:
                st.success("공격 문서 단위 미탐 없음")

            st.markdown("#### Attack Docs Found During Normal Queries")
            found_in_normal = metrics.get("quarantined_attack_docs_during_normal_queries", [])
            if found_in_normal:
                st.dataframe(pd.DataFrame(found_in_normal), use_container_width=True)
            else:
                st.info("없음")

        if not query_df.empty:
            st.markdown("#### Query Status Counts")
            status_counts = query_df["status"].value_counts().rename_axis("status").reset_index(name="count")
            st.bar_chart(status_counts.set_index("status"))

            st.markdown("#### Query Kind Counts")
            kind_counts = query_df["kind"].value_counts().rename_axis("kind").reset_index(name="count")
            st.bar_chart(kind_counts.set_index("kind"))

    with tab_queries:
        st.subheader("Query-Level Report")

        if query_df.empty:
            st.info("질의별 리포트 CSV가 없습니다.")
        else:
            kinds = ["all"] + sorted(query_df["kind"].dropna().unique().tolist())
            selected_kind = st.selectbox("Kind", kinds)

            filtered = query_df.copy()
            if selected_kind != "all":
                filtered = filtered[filtered["kind"] == selected_kind]

            only_quarantine = st.checkbox("격리 발생 질의만 보기")
            if only_quarantine and "has_quarantine" in filtered.columns:
                filtered = filtered[filtered["has_quarantine"].astype(str).str.lower().isin(["true", "1"])]

            only_fp = st.checkbox("정상 문서 오탐 후보만 보기")
            if only_fp and "has_normal_doc_false_positive" in filtered.columns:
                filtered = filtered[
                    filtered["has_normal_doc_false_positive"].astype(str).str.lower().isin(["true", "1"])
                ]

            st.dataframe(filtered, use_container_width=True, height=520)

    with tab_events:
        st.subheader("Quarantine Events")

        if events_df.empty:
            st.info("격리 이벤트 CSV가 없습니다.")
        else:
            doc_types = ["all"] + sorted(events_df["doc_type"].dropna().unique().tolist())
            selected_doc_type = st.selectbox("Doc Type", doc_types)

            filtered_events = events_df.copy()
            if selected_doc_type != "all":
                filtered_events = filtered_events[filtered_events["doc_type"] == selected_doc_type]

            st.dataframe(filtered_events, use_container_width=True, height=520)

            st.markdown("#### Quarantine Events by Query")
            event_counts = (
                filtered_events.groupby(["query_id", "kind"])
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            if not event_counts.empty:
                st.bar_chart(event_counts.set_index("query_id")["count"])

    with tab_db:
        st.subheader("Persistent Quarantine DB")

        if quarantine_df.empty:
            st.info("격리 DB 기록이 없거나 DB 파일이 없습니다.")
        else:
            st.dataframe(quarantine_df, use_container_width=True, height=520)

    with tab_raw:
        st.subheader("Raw Metrics JSON")
        st.json(metrics)


if __name__ == "__main__":
    main()
