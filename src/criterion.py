"""
RAG 평가 스크립트 - criterion.py

Metrics:
1. Reference Recall (수학적): rag_reference vs reference (ground truth)
   - Fuzzy: 같은 조(예: 제17조)의 하위항(제17조 제2항)도 부분 일치로 카운트
2. Answer Correctness (LLM-as-a-judge): rag_answer vs answer (ground truth)
   - 1~5점 척도로 Claude가 평가
3. Faithfulness (LLM-as-a-judge): rag_answer가 ground truth에 근거한 내용만 담고 있는지
   - 1~5점 척도로 Claude가 평가 (hallucination 여부)
"""

import json
import re
from pathlib import Path

import anthropic
import httpx
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

INPUT_FILE  = str(_PROJECT_ROOT / "data" / "input"  / "all_policies_rag_answer.jsonl")
OUTPUT_FILE = str(_PROJECT_ROOT / "data" / "output" / "all_policies_eval_result.jsonl")
MODEL = "claude-sonnet-4-6"


# ─── 데이터 로드 ──────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ─── 1. Reference Recall (수학적) ────────────────────────────────────────────

def normalize_ref(ref: str) -> str:
    return re.sub(r"\s+", "", ref).strip()


def is_fuzzy_match(gt_ref: str, rag_ref: str) -> bool:
    gt_norm  = normalize_ref(gt_ref)
    rag_norm = normalize_ref(rag_ref)
    return rag_norm.startswith(gt_norm) or gt_norm.startswith(rag_norm)


def compute_recall(gt_refs: list[str], rag_refs: list[str]) -> float:
    if not gt_refs:
        return 1.0
    matched = sum(
        1 for gt in gt_refs if any(is_fuzzy_match(gt, rag) for rag in rag_refs)
    )
    return matched / len(gt_refs)


def evaluate_recall(records: list[dict]) -> list[dict]:
    results = []
    for r in records:
        gt_refs  = r.get("reference", [])
        has_rag_ref = "rag_reference" in r
        rag_refs = r.get("rag_reference", [])
        fuzzy = round(compute_recall(gt_refs, rag_refs), 4) if has_rag_ref else None
        results.append({
            "qid":           r["qid"],
            "question":      r["question"],
            "gt_reference":  gt_refs,
            "rag_reference": rag_refs,
            "recall_fuzzy":  fuzzy,
        })
    return results


# ─── 2. Answer Correctness (LLM-as-a-judge) ──────────────────────────────────

JUDGE_SYSTEM = """당신은 보험 약관 기반 QA 시스템의 답변 품질을 평가하는 전문 평가자입니다.
반드시 JSON 형식으로만 응답하십시오."""

CORRECTNESS_PROMPT = """다음 질문에 대한 두 답변을 비교하여 RAG 답변의 정확성을 평가하십시오.

[질문]
{question}

[기준 답변 (Ground Truth)]
{ground_truth}

[RAG 시스템 답변]
{rag_answer}

평가 기준:
- 사실적 정확성: RAG 답변이 기준 답변과 동일한 사실을 포함하는가
- 핵심 정보 포함: 중요 조건·예외사항 누락 없이 설명하는가
- 오류 여부: 잘못된 수치, 조항 번호, 법적 기준 등이 포함되어 있는가

점수 기준:
5 = 기준 답변과 내용이 거의 동일하며 오류 없음
4 = 핵심 내용은 맞으나 일부 세부사항 누락
3 = 대체로 맞으나 일부 부정확하거나 중요한 내용 누락
2 = 부분적으로 맞으나 오류나 누락이 많음
1 = 핵심 내용이 틀리거나 완전히 다른 내용

아래 JSON 형식으로만 응답하십시오:
{{"score": <1~5 정수>, "reason": "<한국어로 2~3문장 평가 근거>"}}"""

FAITHFULNESS_PROMPT = """다음 RAG 시스템의 답변이 기준 답변(Ground Truth)에 근거한 사실만을 담고 있는지 평가하십시오.

[질문]
{question}

[기준 답변 (Ground Truth) — 약관 원문 기반 정답]
{ground_truth}

[RAG 시스템 답변]
{rag_answer}

평가 기준 (Faithfulness):
RAG 답변의 각 주장이 기준 답변에서 뒷받침되는지 확인하십시오.
기준 답변에 없는 내용을 RAG 답변이 사실인 것처럼 추가했다면 점수가 낮습니다.

점수 기준:
5 = RAG 답변의 모든 주장이 기준 답변에 의해 뒷받침됨 (hallucination 없음)
4 = 대부분 뒷받침되며 사소한 추가 정보만 있음
3 = 일부 주장이 기준 답변에 없는 내용이나 불확실한 내용을 포함
2 = 여러 주장이 기준 답변과 다르거나 근거 없는 내용을 포함
1 = 기준 답변과 상충되거나 대부분 hallucination

아래 JSON 형식으로만 응답하십시오:
{{"score": <1~5 정수>, "reason": "<한국어로 2~3문장 평가 근거>"}}"""


def _parse_judge_response(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = re.sub(r"```\s*$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"score": None, "reason": text}


def judge_correctness(
    client: anthropic.Anthropic,
    question: str,
    ground_truth: str,
    rag_answer: str,
) -> dict:
    prompt = CORRECTNESS_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        rag_answer=rag_answer,
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_judge_response(response.content[0].text.strip())


def judge_faithfulness(
    client: anthropic.Anthropic,
    question: str,
    ground_truth: str,
    rag_answer: str,
) -> dict:
    prompt = FAITHFULNESS_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        rag_answer=rag_answer,
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_judge_response(response.content[0].text.strip())


def evaluate_correctness(records: list[dict]) -> list[dict]:
    client = anthropic.Anthropic(http_client=httpx.Client(verify=False))
    results = []
    for r in records:
        print(f"  [qid={r['qid']}] 평가 중...", end=" ", flush=True)
        c_judgment = judge_correctness(client, r["question"], r["answer"], r["rag_answer"])
        f_judgment = judge_faithfulness(client, r["question"], r["answer"], r["rag_answer"])
        print(f"correctness={c_judgment.get('score')}  faithfulness={f_judgment.get('score')}")
        results.append({
            "qid":                  r["qid"],
            "correctness_score":    c_judgment.get("score"),
            "correctness_reason":   c_judgment.get("reason", ""),
            "faithfulness_score":   f_judgment.get("score"),
            "faithfulness_reason":  f_judgment.get("reason", ""),
        })
    return results


# ─── 3. 결과 출력 및 저장 ─────────────────────────────────────────────────────

def print_results(recall_results: list[dict], correctness_results: list[dict]) -> None:
    correctness_map = {r["qid"]: r for r in correctness_results}

    print("\n" + "=" * 80)
    print("  RAG 평가 결과")
    print("=" * 80)

    total_fuzzy = 0.0
    total_score = 0.0
    total_faith = 0.0
    scored_count = 0
    faith_count  = 0

    for rc in recall_results:
        qid = rc["qid"]
        cc  = correctness_map.get(qid, {})
        score = cc.get("correctness_score")
        faith = cc.get("faithfulness_score")

        match_label = "[부분 일치]" if rc["recall_fuzzy"] == 1.0 else "[불일치]   "

        print(f"\n  qid={qid}  {match_label}")
        print(f"    GT ref  : {rc['gt_reference']}")
        print(f"    RAG ref : {rc['rag_reference']}")
        print(f"    Recall  : fuzzy={rc['recall_fuzzy']:.2f}")
        print(f"    Correctness : {score}/5  →  {cc.get('correctness_reason', '')}")
        print(f"    Faithfulness: {faith}/5  →  {cc.get('faithfulness_reason', '')}")

        total_fuzzy += rc["recall_fuzzy"]
        if isinstance(score, (int, float)):
            total_score += score
            scored_count += 1
        if isinstance(faith, (int, float)):
            total_faith += faith
            faith_count  += 1

    n = len(recall_results)
    print("\n" + "-" * 80)
    print("  종합 평균")
    print(f"    Reference Recall (Fuzzy) : {total_fuzzy / n:.3f}")
    if scored_count:
        print(f"    Answer Correctness (LLM) : {total_score / scored_count:.2f} / 5.0  (n={scored_count})")
    if faith_count:
        print(f"    Faithfulness       (LLM) : {total_faith / faith_count:.2f} / 5.0  (n={faith_count})")
    print("=" * 80)


def save_results(recall_results: list[dict], correctness_results: list[dict]) -> None:
    correctness_map = {r["qid"]: r for r in correctness_results}
    rows = []
    for rc in recall_results:
        cc = correctness_map.get(rc["qid"], {})
        rows.append({**rc, **cc})
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n✓ 평가 결과 저장: {OUTPUT_FILE}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== RAG 평가 시작 ===\n")
    records = load_jsonl(INPUT_FILE)
    print(f"총 {len(records)}개 항목 로드\n")

    print("─── 1. Reference Recall 계산 (Fuzzy) ───")
    recall_results = evaluate_recall(records)

    print("\n─── 2. Answer Correctness + Faithfulness (LLM-as-a-judge) ───")
    correctness_results = evaluate_correctness(records)

    print_results(recall_results, correctness_results)
    save_results(recall_results, correctness_results)


if __name__ == "__main__":
    main()
