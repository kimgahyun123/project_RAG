# src/rule_registry.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Literal

SignalType = Literal["structure", "semantic", "verification"]

@dataclass(frozen=True)
class RuleDefinition:
    rule_id: str
    name: str
    group: str
    signal_type: SignalType
    description: str
    implementation_hint: str

RULE_GROUPS: Dict[str, str] = {
    "retrieval_structure": "Top-k 검색 구조상 조립 가능성을 확인한다. 단독 악성 판정 금지.",
    "answer_steering": "조립 context가 특정 답변/결론으로 LLM을 유도하는지 확인한다.",
    "cross_chunk_composition": "여러 청크가 결합되어 하나의 결론/논리 흐름을 만드는지 확인한다.",
    "legal_context_distortion": "법리, 판단 기준, 법적 주체, 문맥 일관성이 왜곡되는지 확인한다.",
    "evidence_imbalance": "근거 설명보다 결론 유도 청크가 과도한지 확인한다.",
    "judge_verification": "LLM Judge 또는 검증 단계에서 나온 보조 판단 신호.",
}

def R(rule_id, name, group, signal_type, desc, hint):
    return RuleDefinition(rule_id, name, group, signal_type, desc, hint)

RULES: Dict[str, RuleDefinition] = {
    "A1": R("A1","Same-source co-retrieval","retrieval_structure","structure","Top-k 안에 같은 doc_id 청크가 2개 이상 검색되는지 확인","doc_id별 count >= 2"),
    "A2": R("A2","Top-rank dominance","retrieval_structure","structure","같은 doc_id가 Rank 1~2 또는 Top-3을 장악하는지 확인","Rank1=Rank2 또는 Top3 count >= 2"),
    "A3": R("A3","Source concentration","retrieval_structure","structure","Top-k에서 특정 doc_id 비중이 과도한지 확인","count >= 3 또는 ratio >= 0.6"),
    "A4": R("A4","Answer steering","answer_steering","semantic","조립 context가 특정 답변/결론을 유도하는 표현을 포함하는지 확인","therefore, final answer, definitive conclusion 등"),
    "A5": R("A5","High-rank direct conclusion","answer_steering","semantic","상위 청크가 질의 핵심어와 직접 결론 표현을 동시에 포함하는지 확인","Rank1/2 query overlap + conclusion pattern"),
    "A6": R("A6","Repeated conclusion","cross_chunk_composition","semantic","여러 청크가 같은 결론 방향을 반복 보강하는지 확인","conclusion group 반복"),
    "A7": R("A7","Contradiction","legal_context_distortion","semantic","청크 사이에 서로 상충하는 결론/법리/판단 기준이 있는지 확인","opposing conclusion groups"),
    "A8": R("A8","Cross-chunk completion","cross_chunk_composition","semantic","개별 청크는 약하지만 조립되면 결론이 완성되는지 확인","issue/bridge/conclusion 역할 분리"),
    "A9": R("A9","LLM Judge warning","judge_verification","verification","LLM Judge가 suspicious/malicious로 판단하는지 확인","query와 Top-k만 제공"),
    "A10": R("A10","Evidence imbalance","evidence_imbalance","semantic","한쪽 결론을 지지하는 청크가 과도한지 확인","dominant direction vs opposing"),
    "A11": R("A11","Query-term over-alignment","answer_steering","semantic","청크가 질의 핵심어에 지나치게 직접 정렬되고 답변 유도성이 높은지 확인","query overlap ratio + conclusion"),
    "A12": R("A12","Position-weighted conclusion bias","answer_steering","semantic","결론 유도 청크가 Top-1/2에 위치하고 더 강한 표현을 갖는지 확인","top2 conclusion avg > lower avg"),
    "A13": R("A13","Normative language burst","answer_steering","semantic","must, should, required, solely 등 규범/단정 표현이 과도한지 확인","normative term count"),
    "A14": R("A14","Legal-standard override","legal_context_distortion","semantic","일반 법리 기준 뒤에 예외/단정 문장이 붙어 원래 기준을 뒤엎는지 확인","standard+override+strong conclusion+query overlap"),
    "A15": R("A15","Semantic redundancy with directional consistency","cross_chunk_composition","semantic","표현은 다르지만 같은 방향 결론을 반복 강화하는지 확인","same conclusion across docs/chunks"),
    "A16": R("A16","Abrupt modality shift","legal_context_distortion","semantic","may/can에서 must/required처럼 판단 강도가 상승하는지 확인","weak+strong modality with query overlap"),
    "A17": R("A17","Context-local inconsistency","legal_context_distortion","semantic","인접 청크끼리 사실/법리/판단 기준이 충돌하는지 확인","adjacent opposing groups"),
    "A18": R("A18","Retrieved-context imbalance","evidence_imbalance","semantic","정상 설명 청크보다 결론 유도 청크 비중이 높은지 확인","conclusion-heavy top-k"),
    "B1": R("B1","Non-adjacent same-doc retrieval","retrieval_structure","structure","같은 doc_id의 멀리 떨어진 chunk_index가 함께 검색되는지 확인","chunk_index gap >= 3"),
    "B2": R("B2","Retrieval-score outlier","retrieval_structure","structure","Rank1이 다른 청크보다 유난히 질의와 잘 맞는지 확인","score gap + conclusion signal"),
    "B3": R("B3","Suspicious heading marker","answer_steering","semantic","Model Direction, Definitive Conclusion 같은 제목이 있는지 확인","heading regex in first 300 chars"),
    "B4": R("B4","Role inversion / legal actor mismatch","legal_context_distortion","semantic","질의의 법적 주체와 context의 지배적 주체가 바뀌는지 확인","query actor vs conclusion chunk actor"),
    "B5": R("B5","Conclusion-before-evidence pattern","answer_steering","semantic","근거보다 결론을 먼저 제시하는지 확인","early conclusion + weak support"),
}

def _rule_sort_key(rule_id: str):
    return (0 if rule_id[0] == "A" else 1, int(rule_id[1:]))

def list_rules() -> List[RuleDefinition]:
    return [RULES[k] for k in sorted(RULES.keys(), key=_rule_sort_key)]

def get_rule(rule_id: str) -> RuleDefinition:
    return RULES[rule_id]

def get_rules_by_group(group: str) -> List[RuleDefinition]:
    return [r for r in list_rules() if r.group == group]

def summarize_rules() -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in RULES.values():
        out[r.group] = out.get(r.group, 0) + 1
    return out

def validate_rule_registry() -> None:
    if len(RULES) != 23:
        raise ValueError(f"RULES must contain 23 rules, found {len(RULES)}")
    missing = {r.group for r in RULES.values() if r.group not in RULE_GROUPS}
    if missing:
        raise ValueError(f"Unknown groups: {missing}")

if __name__ == "__main__":
    validate_rule_registry()
    print("[Rule Registry]")
    print(f"Total rules: {len(RULES)}\n")
    for g, c in summarize_rules().items():
        print(f"{g}: {c}")
    print()
    for r in list_rules():
        print(f"{r.rule_id} | {r.group} | {r.signal_type} | {r.name}")
