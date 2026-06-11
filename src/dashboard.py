# src/dashboard.py
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components


QUERY_LOG = ROOT / "results/query_runs.jsonl"
QUARANTINE_DB = ROOT / "data/quarantine.sqlite3"
GROUND_TRUTH = ROOT / "data/ground_truth.csv"
RAW_DOCS = ROOT / "data/raw_docs"


st.set_page_config(
    page_title="RAG Detector Admin",
    page_icon="shield",
    layout="wide",
)

st_autorefresh(interval=3000, key="dashboard_autorefresh")


def load_query_runs(path: Path = QUERY_LOG) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    rows.reverse()
    return rows


def load_ground_truth() -> pd.DataFrame:
    if not GROUND_TRUTH.exists():
        return pd.DataFrame()
    return pd.read_csv(GROUND_TRUTH)


def load_quarantine_db() -> pd.DataFrame:
    if not QUARANTINE_DB.exists():
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(str(QUARANTINE_DB))
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


def restore_doc(doc_id: str) -> None:
    conn = sqlite3.connect(str(QUARANTINE_DB))
    conn.execute(
        "UPDATE quarantined_documents SET status='restored' WHERE doc_id=?",
        (doc_id,),
    )
    conn.commit()
    conn.close()


def reactivate_doc(doc_id: str) -> None:
    conn = sqlite3.connect(str(QUARANTINE_DB))
    conn.execute(
        "UPDATE quarantined_documents SET status='active' WHERE doc_id=?",
        (doc_id,),
    )
    conn.commit()
    conn.close()


def first_round_label(run: dict[str, Any]) -> str:
    rounds = run.get("rounds", []) or []
    if not rounds:
        return run.get("final_label", "") or "unknown"
    return rounds[0].get("label", "") or "unknown"


def label_text(label: str) -> str:
    label = str(label or "")
    if label == "clean":
        return "정상"
    if label == "suspicious":
        return "의심"
    if label == "malicious":
        return "악성"
    return label or "unknown"


def label_color(label: str) -> str:
    if label == "clean":
        return "#16a34a"
    if label == "suspicious":
        return "#d97706"
    if label == "malicious":
        return "#dc2626"
    return "#64748b"


def doc_type(doc_id: str, gt: pd.DataFrame) -> str:
    if gt.empty or "doc_id" not in gt.columns:
        return "unknown"
    row = gt[gt["doc_id"].astype(str) == str(doc_id)]
    if row.empty:
        return "unknown"
    if "label" in row.columns:
        return str(row.iloc[0]["label"])
    return "unknown"


def resolve_doc_path(doc_id: str, gt: pd.DataFrame) -> Path | None:
    candidates: list[Path] = []

    if not gt.empty and "doc_id" in gt.columns:
        row = gt[gt["doc_id"].astype(str) == str(doc_id)]
        if not row.empty:
            item = row.iloc[0]
            for col in ["path", "original_path", "neutral_file", "file_name"]:
                if col in row.columns and pd.notna(item.get(col)):
                    value = str(item.get(col))
                    candidates.append(ROOT / value)
                    candidates.append(RAW_DOCS / value)

    candidates.extend([
        RAW_DOCS / f"{doc_id}.txt",
        RAW_DOCS / f"{doc_id.replace('kb_', 'doc')}.txt",
    ])

    for path in candidates:
        if path.exists() and path.is_file():
            return path

    return None


def load_doc_text(doc_id: str, gt: pd.DataFrame) -> str:
    import ast

    source = resolve_doc_path(doc_id, gt)

    if not source:
        return ""

    p = Path(source)
    if not p.is_absolute():
        p = ROOT / p

    if not p.exists():
        return ""

    raw = p.read_text(encoding="utf-8", errors="ignore").strip()

    # raw_docs가 ('본문', '경로') 같은 tuple 문자열로 저장된 경우 본문만 추출
    try:
        parsed = ast.literal_eval(raw)

        if isinstance(parsed, tuple) or isinstance(parsed, list):
            if parsed and isinstance(parsed[0], str):
                return parsed[0].strip()

        if isinstance(parsed, dict):
            for key in ["text", "content", "page_content", "document"]:
                value = parsed.get(key)
                if isinstance(value, str):
                    return value.strip()
    except Exception:
        pass

    return raw

def show_doc_view(doc_id: str, gt: pd.DataFrame, quarantine_df: pd.DataFrame, key_prefix: str = "doc"):
    text = load_doc_text(doc_id, gt)

    st.markdown(
        f"""
        <div style="font-size:1.12rem; font-weight:750; margin:.25rem 0 1rem 0;">
            doc_id: <code>{doc_id}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.text_area(
        "문서 내용",
        text or "문서 내용을 찾지 못했습니다.",
        height=360,
        key=f"{key_prefix}_doc_text_{doc_id}",
    )

    if not quarantine_df.empty and doc_id in quarantine_df["doc_id"].astype(str).values:
        row = quarantine_df[quarantine_df["doc_id"].astype(str) == doc_id].iloc[0]
        status = row.get("status", "-")
        label = row.get("label", "-")
        created_at = row.get("created_at", "-")

        st.write(f"격리 상태: `{status}` / 판정: `{label}` / 시간: `{created_at}`")

        if status == "active":
            if st.button("정상 문서로 복구", key=f"{key_prefix}_restore_{doc_id}"):
                restore_doc(doc_id)
                st.success(f"{doc_id} 복구 완료")
                st.rerun()
        else:
            if st.button("다시 격리 상태로 변경", key=f"{key_prefix}_reactivate_{doc_id}"):
                reactivate_doc(doc_id)
                st.success(f"{doc_id} 다시 격리 처리 완료")
                st.rerun()

def top_chunks_table(chunks: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(chunks, start=1):
        rows.append({
            "rank": i,
            "doc_id": c.get("doc_id"),
            "chunk_id": c.get("chunk_id"),
            "chunk_index": c.get("chunk_index"),
            "score": c.get("score"),
            "preview": str(c.get("text", "")).replace("\n", " ")[:180],
        })
    return pd.DataFrame(rows)



def korean_reason_summary(reason: str) -> str:
    reason = str(reason or "").strip()
    if not reason:
        return "기록된 판단 근거가 없습니다."

    code_map = {
        "A1": "같은 출처 문서가 반복 검색됨",
        "A2": "상위 검색 결과가 특정 문서에 과도하게 치우침",
        "A3": "검색 결과 전체가 특정 출처에 집중됨",
        "A4": "답변을 특정 결론으로 유도하는 표현이 있음",
        "A5": "상위 청크에 직접적인 결론 표현이 있음",
        "A6": "같은 결론이 여러 청크에서 반복됨",
        "A7": "청크들 사이에 서로 모순되는 법리나 결론이 있음",
        "A11": "질의 핵심어와 과도하게 맞춰진 문장이 있음",
        "A12": "상위 위치의 결론 문장이 판단을 강하게 유도함",
        "A13": "must, should 같은 단정적/규범적 표현이 과도함",
        "A14": "기존 법적 기준을 예외나 단정 문장으로 덮어씀",
        "A15": "서로 다른 청크가 같은 방향의 결론을 반복 강화함",
        "A16": "may/can 같은 가능 표현이 must/required 같은 의무 표현으로 바뀜",
        "A17": "인접 문맥 안에서 사실관계나 법리가 충돌함",
        "A18": "검색된 문맥이 한쪽 결론으로 과도하게 불균형함",
        "B1": "같은 문서의 멀리 떨어진 청크들이 함께 검색됨",
        "B2": "검색 점수 분포가 비정상적으로 튐",
        "B3": "결론 유도형 제목이나 마커가 있음",
        "B4": "질의의 법적 주체와 문맥의 주체가 혼동됨",
        "B5": "근거보다 결론이 먼저 제시되는 패턴이 있음",
    }

    found = [f"{code}: {desc}" for code, desc in code_map.items() if code in reason]

    lower = reason.lower()
    extra = []

    if "answer steering" in lower:
        extra.append("문맥이 LLM의 답변을 특정 방향으로 유도한다고 판단했습니다.")
    if "contradict" in lower or "contradiction" in lower:
        extra.append("청크들 사이에 서로 충돌하는 설명이나 결론이 있다고 판단했습니다.")
    if "fabricated" in lower or "unsupported" in lower:
        extra.append("근거가 부족하거나 조작된 법적 결론이 포함됐을 가능성이 있다고 판단했습니다.")
    if "directional redundancy" in lower or "redundancy" in lower:
        extra.append("여러 청크가 같은 결론을 반복해서 강화한다고 판단했습니다.")
    if "legal standard" in lower or "override" in lower:
        extra.append("정상적인 법적 판단 기준을 단정적 문장으로 덮어쓰는 흐름이 있다고 판단했습니다.")
    if "query-specific" in lower:
        extra.append("일반 법률 문서에 질의의 특정 인물이나 상황이 직접 들어가 있어 의심스럽다고 판단했습니다.")
    if "balanced" in lower and "accurate" in lower:
        extra.append("문맥이 균형 있고 법적으로 일관되어 최종적으로 정상으로 판단했습니다.")
    if "no malicious" in lower or "no answer steering" in lower:
        extra.append("악성 조립 의미나 답변 유도 흐름은 확인되지 않았다고 판단했습니다.")

    if found or extra:
        parts = []
        if found:
            parts.append("탐지된 주요 기준:\n- " + "\n- ".join(found))
        if extra:
            parts.append("해석:\n- " + "\n- ".join(extra))
        return "\n\n".join(parts)

    return "이 라운드의 판단 근거를 보면, 검색된 청크들이 조립됐을 때 답변을 잘못된 방향으로 유도하는지, 서로 모순되는지, 법적 기준을 단정적으로 바꾸는지 확인한 것으로 해석할 수 있습니다."


def reason_bullets(reason: str) -> list[str]:
    reason = str(reason or "").strip()
    if not reason:
        return ["기록된 판단 근거가 없습니다."]

    import re

    cleaned = re.sub(r"\s+", " ", reason)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9가-힣])", cleaned)
    bullets = [p.strip() for p in parts if p.strip()]

    return bullets or [cleaned]


def reason_codes(reason: str) -> list[str]:
    import re

    known = {
        "A1", "A2", "A3", "A4", "A5", "A6", "A7",
        "A10", "A11", "A12", "A13", "A14", "A15",
        "A16", "A17", "A18", "B1", "B2", "B3", "B4", "B5",
    }
    found = re.findall(r"\b[AB]\d+\b", str(reason or ""))
    return [code for code in dict.fromkeys(found) if code in known]


def render_reason_cards(reason: str, label: str):
    bullets = reason_bullets(reason)
    codes = reason_codes(reason)

    color = {
        "malicious": "#dc2626",
        "suspicious": "#d97706",
        "clean": "#16a34a",
    }.get(str(label), "#64748b")

    bg = {
        "malicious": "#fff1f2",
        "suspicious": "#fffbeb",
        "clean": "#f0fdf4",
    }.get(str(label), "#f8fafc")

    border = {
        "malicious": "#fecdd3",
        "suspicious": "#fde68a",
        "clean": "#bbf7d0",
    }.get(str(label), "#e2e8f0")

    if codes:
        badges = " ".join(
            f"<span style='display:inline-block; padding:.18rem .45rem; margin:.08rem .2rem .08rem 0; "
            f"border-radius:999px; background:{color}; color:white; font-size:.78rem; font-weight:750;'>{code}</span>"
            for code in codes
        )
        st.markdown(badges, unsafe_allow_html=True)

    for idx, bullet in enumerate(bullets, start=1):
        st.markdown(
            f"""
            <div style="
                display:flex;
                gap:.75rem;
                align-items:flex-start;
                padding:.9rem 1rem;
                margin:.55rem 0;
                border:1px solid {border};
                border-left:5px solid {color};
                border-radius:8px;
                background:{bg};
            ">
                <div style="
                    min-width:1.75rem;
                    height:1.75rem;
                    border-radius:999px;
                    background:{color};
                    color:white;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    font-size:.85rem;
                    font-weight:800;
                ">{idx}</div>
                <div style="line-height:1.65; font-size:.98rem;">
                    {bullet}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def show_rounds(run: dict[str, Any], gt: pd.DataFrame, quarantine_df: pd.DataFrame):
    rounds = run.get("rounds", []) or []

    st.markdown("#### Round별 탐지 근거 보기")

    if not rounds:
        st.info("라운드 기록이 없습니다.")
        return

    for idx, round_item in enumerate(rounds, start=1):
        detection = round_item.get("detection") or {}

        if isinstance(detection, dict):
            label = detection.get("label") or round_item.get("label") or round_item.get("detection_label") or "unknown"
            via = detection.get("via") or round_item.get("via") or "-"
            judge_label = detection.get("judge_label") or detection.get("judge") or "-"
            rule_label = detection.get("rule_label") or "-"
            score = detection.get("score")
            reason = detection.get("reason") or round_item.get("reason") or ""
            quarantine_doc_ids = (
                detection.get("quarantine_doc_ids")
                or detection.get("quarantine_candidates")
                or round_item.get("quarantine_doc_ids")
                or round_item.get("quarantine_candidates")
                or []
            )
        else:
            label = round_item.get("label") or round_item.get("detection_label") or "unknown"
            via = round_item.get("via") or "-"
            judge_label = round_item.get("judge_label") or "-"
            rule_label = round_item.get("rule_label") or "-"
            score = round_item.get("score")
            reason = round_item.get("reason") or ""
            quarantine_doc_ids = (
                round_item.get("quarantine_doc_ids")
                or round_item.get("quarantine_candidates")
                or []
            )

        title_parts = [
            f"Round {idx}",
            label_text(label),
            f"via={via}",
        ]
        if judge_label and judge_label != "-":
            title_parts.append(f"judge={label_text(judge_label)}")
        if rule_label and rule_label != "-":
            title_parts.append(f"rule={label_text(rule_label)}")
        if score is not None:
            try:
                title_parts.append(f"score={float(score):.3f}")
            except Exception:
                title_parts.append(f"score={score}")

        with st.expander(" | ".join(title_parts), expanded=False):
            st.markdown("**판단 근거**")
            render_reason_cards(reason, label)

            top_chunks = (
                round_item.get("top_chunks")
                or round_item.get("chunks")
                or round_item.get("results")
                or []
            )

            with st.expander("청크 내용", expanded=False):
                if top_chunks:
                    for cidx, chunk in enumerate(top_chunks, start=1):
                        if isinstance(chunk, dict):
                            doc_id = chunk.get("doc_id") or chunk.get("source_doc_id") or "unknown"
                            chunk_id = chunk.get("chunk_id") or "unknown"
                            score = chunk.get("score") or chunk.get("retrieval_score")
                            text = chunk.get("text") or chunk.get("page_content") or ""
                        else:
                            doc_id = getattr(chunk, "doc_id", "unknown")
                            chunk_id = getattr(chunk, "chunk_id", "unknown")
                            score = getattr(chunk, "score", None)
                            text = getattr(chunk, "text", "") or getattr(chunk, "page_content", "")

                        score_txt = ""
                        if score is not None:
                            try:
                                score_txt = f" | score={float(score):.4f}"
                            except Exception:
                                score_txt = f" | score={score}"

                        with st.expander(f"{cidx}. {doc_id} / {chunk_id}{score_txt}", expanded=False):
                            st.write(text or "청크 내용이 로그에 저장되어 있지 않습니다.")
                else:
                    st.write("청크 내용이 로그에 저장되어 있지 않습니다.")

            with st.expander("격리 문서", expanded=False):
                if quarantine_doc_ids:
                    for doc_id in quarantine_doc_ids:
                        with st.expander(str(doc_id), expanded=False):
                            show_doc_view(str(doc_id), gt, quarantine_df, key_prefix=f"round_{idx}_{doc_id}")
                else:
                    st.write("없음")

def per_query_quarantined_docs(run):
    """
    이 질의에서 새로 격리된 문서만 계산한다.
    기존 quarantine DB에 이미 있던 누적 격리 문서는 질의 목록에 섞지 않는다.
    """
    direct_keys = [
        "run_quarantined_doc_ids",
        "new_quarantined_doc_ids",
        "quarantined_this_run",
    ]

    for key in direct_keys:
        value = run.get(key)
        if isinstance(value, list) and value:
            return sorted(dict.fromkeys(str(x) for x in value if x))

    docs = []

    for round_item in run.get("rounds", []) or []:
        candidates = []

        detection = round_item.get("detection")
        if isinstance(detection, dict):
            candidates.extend(detection.get("quarantine_doc_ids") or [])
            candidates.extend(detection.get("quarantine_candidates") or [])

        candidates.extend(round_item.get("quarantine_doc_ids") or [])
        candidates.extend(round_item.get("quarantine_candidates") or [])

        label = (
            round_item.get("label")
            or round_item.get("detection_label")
            or (detection.get("label") if isinstance(detection, dict) else None)
        )

        if label in {"suspicious", "malicious"}:
            for doc_id in candidates:
                if doc_id and doc_id != "unknown":
                    docs.append(str(doc_id))

    return sorted(dict.fromkeys(docs))


def short_query(query, max_chars=110):
    query = " ".join(str(query or "").split())
    if len(query) <= max_chars:
        return query
    return query[: max_chars - 3].rstrip() + "..."


def query_rows(runs):
    rows = []

    for idx, run in enumerate(runs):
        rounds = run.get("rounds", []) or []
        first_label = first_round_label(run)
        final_label = run.get("final_label") or run.get("label") or "unknown"
        qdocs = per_query_quarantined_docs(run)

        rows.append({
            "run_index": idx,
            "time": run.get("created_at") or run.get("time") or "-",
            "query": run.get("query") or "",
            "status": run.get("status") or "-",
            "first_label": first_label,
            "final_label": final_label,
            "rounds": len(rounds),
            "run_quarantined_doc_ids": qdocs,
        })

    return rows





def render_query_detail(run_index: int, run: dict[str, Any], gt: pd.DataFrame, quarantine_df: pd.DataFrame):
    qdocs = per_query_quarantined_docs(run)
    query = run.get("query") or "질의 내용 없음"

    st.markdown(f"#### {query}")

    st.markdown("#### 격리된 문서")
    if qdocs:
        st.write(", ".join(qdocs))
    else:
        st.write("없음")

    show_rounds(run, gt, quarantine_df)

    answer = run.get("answer") or run.get("final_answer")
    if answer:
        with st.expander("최종 답변", expanded=False):
            st.write(answer)

def render_query_list(rows, runs, gt, quarantine_df):
    st.markdown("### 질의 목록")

    st.markdown(
        """
        <style>
        .query-row-spacer {
            border-bottom: 1px solid #eef0f4;
            margin: 0.55rem 0 0.9rem 0;
        }

        /* 질의 칸: 토글처럼 보이는 버튼 */
        div[data-testid="column"]:nth-child(2) div.stButton > button {
            justify-content: flex-start !important;
            text-align: left !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
            min-height: 2.9rem !important;
            padding-left: .85rem !important;
            padding-right: .85rem !important;
            border-radius: 8px !important;
        }

        div[data-testid="column"]:nth-child(2) div.stButton > button p {
            width: 100% !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
            text-align: left !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # 순서 | 질의 | 시간 | 첫 판정 | 최종 판정 | 상태 | 라운드 | 격리 문서
    col_spec = [0.45, 3.75, 1.55, 0.8, 0.8, 0.95, 0.6, 2.2]

    header = st.columns(col_spec)
    header[0].markdown("**순서**")
    header[1].markdown("**질의**")
    header[2].markdown("**시간**")
    header[3].markdown("**첫 판정**")
    header[4].markdown("**최종 판정**")
    header[5].markdown("**상태**")
    header[6].markdown("**라운드**")
    header[7].markdown("**격리 문서**")

    if not rows:
        st.info("저장된 질의 로그가 없습니다.")
        return

    if "open_run_index" not in st.session_state:
        st.session_state["open_run_index"] = None

    for display_no, row in enumerate(rows, start=1):
        run_index = row["run_index"]
        run = runs[run_index]
        qdocs = row.get("run_quarantined_doc_ids", []) or []

        cols = st.columns(col_spec)

        cols[0].markdown(f"**{display_no}**")

        query_text = row.get("query", "")
        is_open = st.session_state.get("open_run_index") == run_index
        arrow = "▾" if is_open else "▸"
        title = f"{arrow} {short_query(query_text, 95)}"

        if cols[1].button(
            title,
            key=f"toggle_query_{run_index}",
            use_container_width=True,
            help=query_text,
        ):
            if is_open:
                st.session_state["open_run_index"] = None
                st.session_state["open_doc_id"] = None
            else:
                st.session_state["open_run_index"] = run_index
                st.session_state["open_doc_id"] = None
            st.rerun()

        cols[2].write(row.get("time", "-"))

        first_label = row.get("first_label")
        final_label = row.get("final_label")

        cols[3].markdown(
            f"<span style='color:{label_color(first_label)}; font-weight:800'>{label_text(first_label)}</span>",
            unsafe_allow_html=True,
        )
        cols[4].markdown(
            f"<span style='color:{label_color(final_label)}; font-weight:800'>{label_text(final_label)}</span>",
            unsafe_allow_html=True,
        )

        cols[5].write(row.get("status", "-"))
        cols[6].markdown(f"`{row.get('rounds', 0)}`")

        qdoc_text = ", ".join(qdocs) if qdocs else "없음"
        cols[7].write(qdoc_text)

        if st.session_state.get("open_run_index") == run_index:
            render_query_detail(run_index, run, gt, quarantine_df)

        st.markdown("<div class='query-row-spacer'></div>", unsafe_allow_html=True)

def main() -> None:
    st.title("RAG Detector Admin")
    st.caption("사용자 질의 로그를 확인하고, 첫 판정/최종 판정과 격리 문서를 검토하는 관리자 대시보드")

    gt = load_ground_truth()
    runs = load_query_runs()
    quarantine_df = load_quarantine_db()

    with st.sidebar:
        st.header("상태")
        st.metric("저장된 질의", len(runs))

        active_count = 0
        if not quarantine_df.empty:
            active_count = int((quarantine_df["status"].astype(str) == "active").sum())
        st.metric("Active 격리 문서", active_count)
        restored_count = 0
        if not quarantine_df.empty and "status" in quarantine_df.columns:
            restored_count = int((quarantine_df["status"].astype(str) == "restored").sum())
        st.metric("Restored 문서", restored_count)


    tab_queries, tab_quarantine, tab_docs = st.tabs(
        ["질의 로그", "격리 문서 관리", "문서 검색"]
    )

    with tab_queries:
        if not runs:
            return

        rows = query_rows(runs)

        first_clean = sum(1 for r in rows if r["first_label"] == "clean")
        first_suspicious = sum(1 for r in rows if r["first_label"] == "suspicious")
        first_malicious = sum(1 for r in rows if r["first_label"] == "malicious")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("전체 질의", len(rows))
        c2.metric("첫 판정 정상", first_clean)
        c3.metric("첫 판정 의심", first_suspicious)
        c4.metric("첫 판정 악성", first_malicious)

        col_a, col_b = st.columns([1, 2])
        with col_a:
            first_filter = st.selectbox("첫 판정 필터", ["all", "clean", "suspicious", "malicious"])
        with col_b:
            status_filter = st.selectbox("상태 필터", ["all", "answered", "blocked", "error"])

        filtered_rows = rows
        if first_filter != "all":
            filtered_rows = [r for r in filtered_rows if r["first_label"] == first_filter]
        if status_filter != "all":
            filtered_rows = [r for r in filtered_rows if r["status"] == status_filter]

        render_query_list(filtered_rows, runs, gt, quarantine_df)

    with tab_quarantine:
        st.subheader("격리 문서 관리")

        qdf_all = quarantine_df.copy() if isinstance(quarantine_df, pd.DataFrame) else pd.DataFrame()

        if qdf_all.empty or "doc_id" not in qdf_all.columns:
            st.info("격리된 문서가 없습니다.")
        else:
            qdf_all["doc_id"] = qdf_all["doc_id"].astype(str)

            if "status" in qdf_all.columns:
                qdf_all["status"] = qdf_all["status"].astype(str).str.strip().str.lower()
                qdf = qdf_all[qdf_all["status"] == "active"].copy()
            else:
                qdf = qdf_all.copy()

            doc_ids = sorted(qdf["doc_id"].dropna().astype(str).unique().tolist())

            selected_doc_id = st.session_state.get("selected_quarantine_doc_id")
            if selected_doc_id and selected_doc_id not in doc_ids:
                st.session_state["selected_quarantine_doc_id"] = None
                selected_doc_id = None

            if not doc_ids:
                st.info("현재 active 상태인 격리 문서가 없습니다.")
            else:
                st.caption(f"active 격리 문서 {len(doc_ids)}개")

                for start in range(0, len(doc_ids), 8):
                    row_doc_ids = doc_ids[start:start + 8]
                    cols = st.columns(8)

                    for col, doc_id in zip(cols, row_doc_ids):
                        if col.button(doc_id, key=f"quarantine_doc_btn_{doc_id}", use_container_width=True):
                            if st.session_state.get("selected_quarantine_doc_id") == doc_id:
                                st.session_state["selected_quarantine_doc_id"] = None
                            else:
                                st.session_state["selected_quarantine_doc_id"] = doc_id
                            st.rerun()

                    selected_doc_id = st.session_state.get("selected_quarantine_doc_id")

                    if selected_doc_id in row_doc_ids:
                        st.markdown("---")

                        selected_rows = qdf[qdf["doc_id"].astype(str) == selected_doc_id]

                        if selected_rows.empty:
                            st.warning("선택한 문서의 active 격리 기록을 찾지 못했습니다.")
                        else:
                            row = selected_rows.iloc[0]
                            label = row.get("label", "-")
                            status = row.get("status", "-")
                            created_at = row.get("created_at", "-")
                            query = row.get("query", "-")
                            reason = row.get("reason", "-")

                            st.markdown(
                                f"""
                                <div style="font-size:1.12rem; font-weight:750; margin:.25rem 0 1rem 0;">
                                    doc_id: <code>{selected_doc_id}</code>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            c1, c2, c3 = st.columns(3)
                            c1.metric("라벨", str(label))
                            c2.metric("상태", str(status))
                            c3.metric("격리 시간", str(created_at))

                            st.markdown("**질의**")
                            st.write(query if query else "-")

                            st.markdown("**이유**")
                            st.write(reason if reason else "-")

                            text = load_doc_text(selected_doc_id, gt)
                            st.markdown("**문서 내용**")
                            st.text_area(
                                "문서 내용",
                                text or "문서 내용을 찾지 못했습니다.",
                                height=420,
                                key=f"quarantine_doc_text_{selected_doc_id}",
                                label_visibility="collapsed",
                            )

                            if st.button("정상 문서로 복구", key=f"quarantine_restore_{selected_doc_id}"):
                                restore_doc(selected_doc_id)
                                st.session_state["selected_quarantine_doc_id"] = None
                                try:
                                    st.cache_data.clear()
                                except Exception:
                                    pass
                                st.success(f"{selected_doc_id} 복구 완료")
                                st.rerun()

                        st.markdown("---")
    with tab_docs:
        st.subheader("문서 검색")

        docs = gt.copy() if isinstance(gt, pd.DataFrame) else pd.DataFrame()

        if docs.empty or "doc_id" not in docs.columns:
            st.info("문서 목록을 찾지 못했습니다.")
        else:
            docs["doc_id"] = docs["doc_id"].astype(str)

            active_quarantined = set()
            if isinstance(quarantine_df, pd.DataFrame) and not quarantine_df.empty:
                if "doc_id" in quarantine_df.columns and "status" in quarantine_df.columns:
                    active_quarantined = set(
                        quarantine_df.loc[
                            quarantine_df["status"].astype(str).str.lower() == "active",
                            "doc_id",
                        ].astype(str).tolist()
                    )

            doc_ids = sorted(
                doc_id
                for doc_id in docs["doc_id"].dropna().astype(str).unique().tolist()
                if doc_id not in active_quarantined
            )

            if not doc_ids:
                st.info("표시할 문서가 없습니다.")
            else:
                st.caption(f"문서 {len(doc_ids)}개")

                selected_doc_id = st.session_state.get("selected_search_doc_id")

                for start in range(0, len(doc_ids), 8):
                    row_doc_ids = doc_ids[start:start + 8]
                    cols = st.columns(8)

                    for col, doc_id in zip(cols, row_doc_ids):
                        if col.button(doc_id, key=f"doc_search_btn_{doc_id}", use_container_width=True):
                            if st.session_state.get("selected_search_doc_id") == doc_id:
                                st.session_state["selected_search_doc_id"] = None
                            else:
                                st.session_state["selected_search_doc_id"] = doc_id
                            st.rerun()

                    selected_doc_id = st.session_state.get("selected_search_doc_id")

                    if selected_doc_id in row_doc_ids:
                        st.markdown("---")
                        show_doc_view(
                            selected_doc_id,
                            gt,
                            quarantine_df,
                            key_prefix=f"doc_search_{selected_doc_id}",
                        )
                        st.markdown("---")
if __name__ == "__main__":
    main()
