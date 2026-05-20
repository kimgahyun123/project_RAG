# src/detector.py
#
# 단일 문서 업로드 시점 탐지기
#
# 적용 기법 (업로드 시점, 쿼리 없음):
#   [1] PPL        - PoisonedRAG (Zou et al., USENIX Security 2025)
#   [2] Rule       - PoisonedRAG (Zou et al., USENIX Security 2025)
#   [3] PD Score   - RAGuard (arxiv:2510.25025)
#   [4] Corpus Sim - RAGuard (arxiv:2510.25025)
#
# 최종 판정: 가중 합산 >= final_threshold → 악성
#
# 임계값 (legal-rag-bench 정상 문서 200개 기준, 오탐 0개):
#   ppl_threshold        = 55.0   (정상 최대 50.95)
#   pd_threshold         = 95.0   (정상 최대 89.74)
#   corpus_sim_threshold = 0.98   (정상 최대 0.973)
#   final_threshold      = 0.65   (정상 최고 점수 0.630)

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── 데이터 클래스 ─────────────────────────────────────────────

@dataclass
class RuleMatch:
    name: str
    weight: float
    description: str
    matched_text: str


@dataclass
class DetectionResult:
    label: str
    ppl: float
    ppl_threshold: float
    ppl_flag: bool
    rule_score: float
    rule_threshold: float
    rule_flag: bool
    pd_score: float
    pd_threshold: float
    pd_flag: bool
    corpus_sim: float
    corpus_sim_threshold: float
    corpus_sim_flag: bool
    final_score: float
    final_threshold: float
    matched_rules: List[RuleMatch] = field(default_factory=list)
    reason: str = ""
    text_preview: str = ""
    is_malicious: bool = False
    doc_id: str = ""


@dataclass
class RulePattern:
    name: str
    pattern: str
    weight: float
    description: str


# ── 탐지기 ────────────────────────────────────────────────────

class SingleDocPoisonDetector:
    """
    단일 문서 업로드 시점 탐지기 (문서 단위, 쿼리 없음)

    [1] PPL 탐지 (PoisonedRAG, USENIX Security 2025)
        distilgpt2 stride 기반 perplexity
        비자연스러운 텍스트 → PPL 높음

    [2] Rule 탐지 (PoisonedRAG, USENIX Security 2025)
        Regex 패턴으로 prompt injection 키워드 탐지
        법률 도메인 오탐 패턴 3개 제거:
          override_priority, tool_or_action_command, always_answer_bias

    [3] PD Score (RAGuard, arxiv:2510.25025)
        전반부/후반부 PPL 차이 측정
        구조적 불균형 문서 탐지

    [4] Corpus Similarity (RAGuard, arxiv:2510.25025)
        정상 corpus와 코사인 유사도
        정상 문서를 재구성한 공격 문서 탐지

    최종 판정 (가중 합산):
    final = 0.30*PPL + 0.30*Rule + 0.20*PD + 0.20*Sim >= 0.65
    """

    def __init__(
        self,
        ppl_model_name: str = "distilgpt2",
        ppl_threshold: float = 55.0,
        rule_threshold: float = 1.5,
        pd_threshold: float = 95.0,
        corpus_sim_threshold: float = 0.98,
        final_threshold: float = 0.65,
        weight_ppl: float = 0.30,
        weight_rule: float = 0.30,
        weight_pd: float = 0.20,
        weight_corpus_sim: float = 0.20,
        stride: int = 256,
        device: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        extra_patterns: Optional[List[Dict[str, Any]]] = None,
        max_preview_chars: int = 180,
    ) -> None:
        self.ppl_threshold = ppl_threshold
        self.rule_threshold = rule_threshold
        self.pd_threshold = pd_threshold
        self.corpus_sim_threshold = corpus_sim_threshold
        self.final_threshold = final_threshold
        self.weight_ppl = weight_ppl
        self.weight_rule = weight_rule
        self.weight_pd = weight_pd
        self.weight_corpus_sim = weight_corpus_sim
        self.stride = stride
        self.max_preview_chars = max_preview_chars
        self.device = device or self._auto_device()
        self._corpus_embeddings: Optional[np.ndarray] = None

        print(f"[Detector] PPL 모델 로드 중: {ppl_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(ppl_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(ppl_model_name)
        self.model.eval()
        self.model.to(self.device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_positions = (
            getattr(self.model.config, "n_positions", None)
            or getattr(self.model.config, "max_position_embeddings", None)
            or 1024
        )

        print(f"[Detector] 임베딩 모델 로드 중: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        self.rule_patterns = self._compile_patterns(extra_patterns)

        print(f"[Detector] device={self.device}")
        print(f"[Detector] ppl_threshold={self.ppl_threshold}")
        print(f"[Detector] pd_threshold={self.pd_threshold}")
        print(f"[Detector] corpus_sim_threshold={self.corpus_sim_threshold}")
        print(f"[Detector] final_threshold={self.final_threshold}")

    # ── Corpus 학습 ───────────────────────────────────────────

    def fit_corpus(self, corpus_texts: List[str]) -> None:
        """
        [4] Corpus Similarity 계산을 위해 정상 문서 임베딩 저장
        업로드 탐지 전에 반드시 호출해야 함
        """
        print(f"[Detector] corpus fit 중: {len(corpus_texts)}개 문서")
        embeddings = self.embedder.encode(
            corpus_texts,
            normalize_embeddings=False,
            show_progress_bar=True,
            batch_size=32,
        )
        embeddings = np.array(embeddings, dtype=np.float64)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self._corpus_embeddings = embeddings / (norms + 1e-8)
        print(f"[Detector] corpus fit 완료")

    def fit_corpus_from_dir(self, corpus_dir: str | Path) -> None:
        """정상 문서 디렉토리에서 corpus 학습"""
        texts = self._load_texts_from_dir(corpus_dir)
        self.fit_corpus(texts)

    # ── 핵심 탐지 메서드 ──────────────────────────────────────

    def detect(self, text: str, doc_id: str = "unknown") -> DetectionResult:
        """
        단일 문서 탐지 (업로드 시점)
        is_malicious=True 이면 업로드 차단
        """
        raw_text = text or ""

        # [1] PPL
        ppl = self.compute_perplexity(raw_text)
        ppl_flag = ppl >= self.ppl_threshold
        ppl_score = min(1.0, ppl / self.ppl_threshold)

        # [2] Rule
        rule_score_raw, matched_rules = self._rule_score(raw_text)
        rule_flag = rule_score_raw >= self.rule_threshold
        rule_score_norm = min(1.0, rule_score_raw / max(self.rule_threshold, 1e-8))

        # [3] PD Score
        pd_result = self._pd_score(raw_text)
        pd_flag = pd_result["pd"] >= self.pd_threshold
        pd_score_norm = min(1.0, pd_result["pd"] / self.pd_threshold)

        # [4] Corpus Similarity
        emb = self.embedder.encode(raw_text[:8192], normalize_embeddings=False)
        emb = np.array(emb, dtype=np.float64)
        corpus_sim = self._compute_corpus_sim(emb)
        corpus_sim_flag = corpus_sim >= self.corpus_sim_threshold

        # 최종 점수
        final_score = (
            self.weight_ppl        * ppl_score +
            self.weight_rule       * rule_score_norm +
            self.weight_pd         * pd_score_norm +
            self.weight_corpus_sim * corpus_sim
        )
        is_malicious = final_score >= self.final_threshold
        label = "malicious" if is_malicious else "benign"

        reasons = []
        if ppl_flag:         reasons.append(f"PPL {ppl:.1f}>={self.ppl_threshold}")
        if rule_flag:        reasons.append(f"Rule {rule_score_raw:.1f}>={self.rule_threshold}")
        if pd_flag:          reasons.append(f"PD {pd_result['pd']:.1f}>={self.pd_threshold}")
        if corpus_sim_flag:  reasons.append(f"CorpusSim {corpus_sim:.3f}>={self.corpus_sim_threshold}")
        if not reasons:      reasons.append("No threshold exceeded")

        print(f"\n[단일 문서 탐지] {doc_id}")
        print("-" * 60)
        print(f"[1] PPL     (PoisonedRAG): {ppl:7.2f}  "
              f"임계값={self.ppl_threshold:.0f}  점수={ppl_score:.3f}  "
              f"{'⚠️' if ppl_flag else '✅'}")
        print(f"[2] Rule    (PoisonedRAG): {rule_score_raw:7.2f}  "
              f"임계값={self.rule_threshold:.1f}  점수={rule_score_norm:.3f}  "
              f"{'⚠️' if rule_flag else '✅'}")
        if matched_rules:
            for m in matched_rules:
                print(f"    → 탐지 규칙: {m.name}")
        print(f"[3] PD Score (RAGuard):   {pd_result['pd']:7.2f}  "
              f"임계값={self.pd_threshold:.0f}  점수={pd_score_norm:.3f}  "
              f"{'⚠️' if pd_flag else '✅'}  "
              f"(전반={pd_result['ppl_pre']:.1f} 후반={pd_result['ppl_post']:.1f})")
        cs_note = "(corpus 미학습)" if self._corpus_embeddings is None else ""
        print(f"[4] Corpus Sim (RAGuard): {corpus_sim:7.4f}  "
              f"임계값={self.corpus_sim_threshold:.2f}  점수={corpus_sim:.4f}  "
              f"{'⚠️' if corpus_sim_flag else '✅'} {cs_note}")
        print(f"최종 점수: {final_score:.4f}  (임계값: {self.final_threshold})")
        print(f"최종 판정: {'🚨 악성' if is_malicious else '✅ 정상'}")
        print("-" * 60)

        return DetectionResult(
            label=label,
            ppl=float(ppl), ppl_threshold=float(self.ppl_threshold), ppl_flag=bool(ppl_flag),
            rule_score=float(rule_score_raw), rule_threshold=float(self.rule_threshold), rule_flag=bool(rule_flag),
            pd_score=float(pd_result["pd"]), pd_threshold=float(self.pd_threshold), pd_flag=bool(pd_flag),
            corpus_sim=float(corpus_sim), corpus_sim_threshold=float(self.corpus_sim_threshold),
            corpus_sim_flag=bool(corpus_sim_flag),
            final_score=float(final_score), final_threshold=float(self.final_threshold),
            matched_rules=matched_rules,
            reason="; ".join(reasons),
            text_preview=self._preview(raw_text),
            is_malicious=bool(is_malicious),
            doc_id=doc_id,
        )

    def detect_file(self, file_path: str | Path) -> DetectionResult:
        """파일 경로로 탐지"""
        text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        return self.detect(text, Path(file_path).name)

    def scan_path(self, path: str | Path) -> List[Tuple[str, DetectionResult]]:
        """파일 또는 디렉토리 전체 탐지"""
        path = Path(path)
        if path.is_file():
            return [(str(path), self.detect_file(path))]
        return [(str(fp), self.detect_file(fp))
                for fp in sorted(path.rglob("*.txt"))]

    def evaluate_from_dirs(
        self,
        benign_dir: str | Path,
        malicious_dir: str | Path,
        save_csv_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """정상/악성 디렉토리로 성능 평가"""
        rows, y_true, y_pred = [], [], []

        for fp in sorted(Path(benign_dir).rglob("*.txt")):
            r = self.detect_file(fp)
            y_true.append(0)
            y_pred.append(1 if r.is_malicious else 0)
            rows.append(self._result_row(fp, 0, r))

        for fp in sorted(Path(malicious_dir).rglob("*.txt")):
            r = self.detect_file(fp)
            y_true.append(1)
            y_pred.append(1 if r.is_malicious else 0)
            rows.append(self._result_row(fp, 1, r))

        metrics = self._binary_metrics(y_true, y_pred)
        summary = {
            "num_samples": len(y_true),
            "metrics": metrics,
            "config": {
                "ppl_threshold": self.ppl_threshold,
                "rule_threshold": self.rule_threshold,
                "pd_threshold": self.pd_threshold,
                "corpus_sim_threshold": self.corpus_sim_threshold,
                "final_threshold": self.final_threshold,
            },
            "rows": rows,
        }

        print(f"\n[평가 결과]")
        print(f"  정확도: {metrics['accuracy']:.3f}")
        print(f"  정밀도: {metrics['precision']:.3f}")
        print(f"  재현율: {metrics['recall']:.3f}")
        print(f"  F1:     {metrics['f1']:.3f}")
        print(f"  TP={int(metrics['tp'])}  TN={int(metrics['tn'])}  "
              f"FP={int(metrics['fp'])}  FN={int(metrics['fn'])}")

        if save_csv_path:
            self._save_rows_to_csv(rows, save_csv_path)

        return summary

    # ── 개별 탐지 메서드 ──────────────────────────────────────

    @torch.no_grad()
    def compute_perplexity(self, text: str) -> float:
        """[1] PPL: distilgpt2 stride 슬라이딩 윈도우"""
        text = text.replace("\x00", " ").strip()
        if not text:
            return 1.0
        encodings = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = encodings["input_ids"]
        seq_len = input_ids.size(1)
        if seq_len <= 1:
            return 1.0

        nll_sum = torch.tensor(0.0, device=self.device)
        total_tokens = 0
        prev_end = 0

        for begin in range(0, seq_len, self.stride):
            end = min(begin + self.max_positions, seq_len)
            trg_len = end - prev_end
            ids_slice = input_ids[:, begin:end].to(self.device)
            target_ids = ids_slice.clone()
            target_ids[:, :-trg_len] = -100
            outputs = self.model(ids_slice, labels=target_ids)
            nll_sum += outputs.loss * trg_len
            total_tokens += trg_len
            prev_end = end
            if end == seq_len:
                break

        if total_tokens == 0:
            return 1.0
        return float(torch.exp(nll_sum / total_tokens).item())

    def _pd_score(self, text: str) -> dict:
        """[3] PD Score: 전반부/후반부 PPL 차이"""
        words = text.split()
        if len(words) < 40:
            return {"ppl_pre": 0.0, "ppl_post": 0.0, "pd": 0.0}
        mid = len(words) // 2
        ppl_pre = self.compute_perplexity(" ".join(words[:mid]))
        ppl_post = self.compute_perplexity(" ".join(words[mid:]))
        return {
            "ppl_pre": round(ppl_pre, 2),
            "ppl_post": round(ppl_post, 2),
            "pd": round(abs(ppl_pre - ppl_post), 2),
        }

    def _compute_corpus_sim(self, emb: np.ndarray) -> float:
        """[4] Corpus Similarity: 정상 corpus와 코사인 유사도 (self-sim 제외)"""
        if self._corpus_embeddings is None:
            return 0.0
        norm = np.linalg.norm(emb)
        if norm < 1e-8:
            return 0.0
        emb_norm = emb / norm
        sims = self._corpus_embeddings @ emb_norm
        if len(sims) >= 2:
            top2_idx = np.argpartition(sims, -2)[-2:]
            return float(np.min(sims[top2_idx]))
        return float(np.max(sims))

    def _rule_score(self, text: str) -> Tuple[float, List[RuleMatch]]:
        """[2] Rule: prompt injection Regex 패턴 탐지"""
        text_for_rules = re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()
        total_score = 0.0
        matched_rules = []
        for rule, regex in self.rule_patterns:
            match = regex.search(text_for_rules)
            if match:
                total_score += rule.weight
                matched_rules.append(RuleMatch(
                    name=rule.name, weight=rule.weight,
                    description=rule.description,
                    matched_text=self._safe_snippet(match.group(0), 120),
                ))
        return float(total_score), matched_rules

    # ── Rule 패턴 ─────────────────────────────────────────────

    def _compile_patterns(self, extra_patterns=None):
        patterns = self._default_patterns()
        if extra_patterns:
            for p in extra_patterns:
                patterns.append(RulePattern(
                    p["name"], p["pattern"],
                    float(p.get("weight", 1.0)), p.get("description", ""),
                ))
        return [(p, re.compile(p.pattern, flags=re.IGNORECASE | re.DOTALL))
                for p in patterns]

    def _default_patterns(self) -> List[RulePattern]:
        """
        PoisonedRAG 기반 prompt injection 탐지 패턴
        제거된 패턴 (법률 도메인 오탐):
          override_priority, tool_or_action_command, always_answer_bias
        """
        return [
            RulePattern(
                "ignore_previous_instructions",
                r"\b(ignore|disregard|forget)\b.{0,50}\b(previous|prior|above|earlier|all)\b"
                r".{0,50}\b(instruction|instructions|context|rule|rules)\b",
                2.0, "Previous instructions/context override attempt",
            ),
            RulePattern(
                "role_reassignment",
                r"\b(you are now|act as|from now on you are|assume the role of)\b"
                r".{0,80}\b(system|developer|assistant|agent|judge|reviewer)\b",
                1.5, "Role reassignment / control attempt",
            ),
            RulePattern(
                "secret_exfiltration",
                r"\b(reveal|print|show|expose|leak)\b.{0,80}"
                r"\b(system prompt|developer message|hidden prompt|secret|api key|token|credentials)\b",
                2.0, "Prompt or secret exfiltration request",
            ),
            RulePattern(
                "forced_output_format",
                r"\b(respond only with|answer only with|output exactly"
                r"|do not explain|without explanation)\b",
                1.0, "Forced output format / suppression language",
            ),
            RulePattern(
                "suppress_source_or_context",
                r"\b(do not mention|don't mention|never mention"
                r"|ignore the source|ignore the document|do not cite|don't cite)\b",
                1.0, "Attempt to hide source/context usage",
            ),
            RulePattern(
                "if_asked_answer_x",
                r"\b(if asked|when asked)\b.{0,120}\b(answer|respond|say|state|conclude)\b",
                1.0, "Conditional answer steering",
            ),
            RulePattern(
                "system_policy_reference",
                r"\b(system prompt|developer instruction|policy above"
                r"|chain of command|hidden instruction)\b",
                1.0, "Reference to system/developer-level instructions",
            ),
        ]

    # ── 유틸리티 ──────────────────────────────────────────────

    def _result_row(self, file_path, true_label, result) -> dict:
        return {
            "file_path": str(file_path),
            "true_label": true_label,
            "pred_label": 1 if result.is_malicious else 0,
            "ppl": result.ppl,
            "ppl_flag": result.ppl_flag,
            "rule_score": result.rule_score,
            "rule_flag": result.rule_flag,
            "pd_score": result.pd_score,
            "pd_flag": result.pd_flag,
            "corpus_sim": result.corpus_sim,
            "corpus_sim_flag": result.corpus_sim_flag,
            "final_score": result.final_score,
            "reason": result.reason,
        }

    def _save_rows_to_csv(self, rows, save_csv_path):
        save_csv_path = Path(save_csv_path)
        save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            return
        with save_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _load_texts_from_dir(self, root_dir) -> List[str]:
        return [fp.read_text(encoding="utf-8", errors="ignore")
                for fp in sorted(Path(root_dir).rglob("*.txt"))]

    def _preview(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        return (text if len(text) <= self.max_preview_chars
                else text[:self.max_preview_chars] + "...")

    def _safe_snippet(self, text: str, max_len: int = 120) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        return text if len(text) <= max_len else text[:max_len] + "..."

    def _auto_device(self) -> str:
        if torch.cuda.is_available(): return "cuda"
        if torch.backends.mps.is_available(): return "mps"
        return "cpu"

    def _percentile(self, values: List[float], q: float) -> float:
        if q <= 0: return float(min(values))
        if q >= 100: return float(max(values))
        xs = sorted(values)
        pos = (len(xs) - 1) * (q / 100.0)
        lo, hi = math.floor(pos), math.ceil(pos)
        if lo == hi: return float(xs[lo])
        return float(xs[lo] * (1 - (pos - lo)) + xs[hi] * (pos - lo))

    def _binary_metrics(self, y_true, y_pred) -> dict:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        total = max(len(y_true), 1)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        return {
            "accuracy": (tp + tn) / total,
            "precision": precision, "recall": recall, "f1": f1,
            "tp": float(tp), "tn": float(tn),
            "fp": float(fp), "fn": float(fn),
        }

