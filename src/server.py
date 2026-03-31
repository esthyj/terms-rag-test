"""
RAG 평가 시각화 서버 - server.py

criterion.py의 평가 함수를 직접 임포트하여 실행하고,
결과를 웹 대시보드로 시각화합니다.

실행: python server.py
접속: http://localhost:8000
"""

import json
import sys
import threading
from pathlib import Path

# criterion.py / export_html.py 가 server.py 와 같은 디렉토리에 있으므로
# 어느 경로에서 실행해도 임포트가 되도록 sys.path 에 추가합니다.
_THIS_DIR = str(Path(__file__).parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from criterion import evaluate_recall, judge_correctness, judge_faithfulness, load_jsonl

import anthropic
import httpx
from pathlib import Path as _Path
from dotenv import load_dotenv as _load_dotenv
_BASE_DIR = _Path(__file__).parent.parent  # project root
_load_dotenv(_BASE_DIR / ".env")

_DATA_DIR        = _BASE_DIR / "data"
INPUT_FILE       = str(_DATA_DIR / "input"  / "all_policies_rag_answer.jsonl")
OUTPUT_FILE      = str(_DATA_DIR / "output" / "all_policies_eval_result.jsonl")
ADVICE_FILE      = str(_DATA_DIR / "output" / "all_policies_advice.json")


def _check_startup_conditions():
    """서버 시작 시 필수 환경 및 데이터 파일 상태를 점검합니다."""
    # 1. .env 파일 확인
    env_path = _BASE_DIR / ".env"
    if not env_path.exists():
        print("\n" + "=" * 62)
        print("  [경고] .env 파일을 찾을 수 없습니다.")
        print()
        print("  ANTHROPIC_API_KEY 설정이 필요합니다.")
        print("  .env.example 을 복사하여 .env 를 만들고,")
        print("  ANTHROPIC_API_KEY=your_api_key_here 를 채워 넣으세요.")
        print()
        print("      cp .env.example .env")
        print("      # 이후 .env 파일을 열어 API 키를 입력하세요.")
        print("=" * 62 + "\n")

    # 2. 입력 파일의 열 확인
    input_path = _Path(INPUT_FILE)
    if not input_path.exists():
        return

    with open(input_path, "r", encoding="utf-8") as _f:
        first_line = _f.readline().strip()
    if not first_line:
        return

    sample = json.loads(first_line)

    # rag_answer 열 없으면 자동 추가
    if "rag_answer" not in sample:
        print("\n" + "=" * 62)
        print(f"  [안내] '{INPUT_FILE}' 에 'rag_answer' 열이 없습니다.")
        print("  빈 문자열('')로 자동 추가하여 저장합니다.")
        print("=" * 62)
        _records = []
        with open(input_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line:
                    _obj = json.loads(_line)
                    _obj.setdefault("rag_answer", "")
                    _records.append(_obj)
        with open(input_path, "w", encoding="utf-8") as _f:
            for _rec in _records:
                _f.write(json.dumps(_rec, ensure_ascii=False) + "\n")
        print(f"  -> {len(_records)}개 행에 'rag_answer' 열 추가 완료.\n")

    # rag_reference 열 없으면 안내만 출력 (선택 항목)
    if "rag_reference" not in sample:
        print("\n" + "=" * 62)
        print(f"  [안내] '{INPUT_FILE}' 에 'rag_reference' 열이 없습니다.")
        print()
        print("  rag_reference 는 선택 항목입니다.")
        print("  RAG 시스템이 참조 조항(예: ['제17조'])을 함께 반환한다면")
        print("  해당 열을 추가하면 Reference Recall 지표도 평가됩니다.")
        print("  추가하지 않아도 Correctness·Faithfulness 평가는 정상 동작합니다.")
        print("=" * 62 + "\n")


_check_startup_conditions()


app = FastAPI()

# ─── 평가 상태 관리 ───────────────────────────────────────────────────────────

_status = {"running": False, "done": False, "error": None, "progress": "", "pct": 0}


# ─── 종합 개선 의견 생성 ──────────────────────────────────────────────────────

ADVICE_SYSTEM = """당신은 RAG(Retrieval-Augmented Generation) 시스템 성능 분석 전문가입니다.
보험 약관 QA 평가 결과를 분석하고, 구체적이고 실행 가능한 개선 의견을 제시합니다."""

ADVICE_PROMPT = """아래는 보험 약관 RAG 시스템의 평가 결과입니다.

[RAG 시스템 구조]
- 방식: Full-document RAG (청킹 없음)
- 이유: 각 약관 문서가 충분히 짧아 전체를 LLM context window에 직접 투입
- 구성: 질문 + 전체 약관 문서 -> LLM -> 답변{reference_pipeline_note}

[전체 요약 통계]
- 총 질문 수: {total}개 ({n_lacla}개 LA-CLA){recall_summary_line}
- Answer Correctness 평균: {avg_score:.2f} / 5.0
- Faithfulness 평균: {avg_faith:.2f} / 5.0
- Correctness 점수 분포: {score_dist}
- Faithfulness 점수 분포: {faith_dist}

[LA-CLA별 성능]
{lacla_stats}

[질문 유형(type)별 성능]
{type_stats}

[저성능 항목 (correctness <= 2)]
{low_score_items}

[저신뢰 항목 (faithfulness <= 2)]
{low_faith_items}
{zero_recall_section}
위 데이터를 분석하여 1~3개의 개선 의견 카드를 생성하십시오.
반드시 실제 수치와 관찰된 패턴을 근거로 작성하고, LA-CLA별/유형별 차이가 있다면 구체적으로 언급하십시오.

아래 JSON 배열 형식으로만 출력하십시오 (다른 텍스트 없이 JSON만):
[
  {{
    "type": "{allowed_types} 중 하나",
    "title": "카드 제목 (이모지 포함, 40자 이내)",
    "bullets": ["개선 의견 문장 1", "개선 의견 문장 2"]
  }}
]"""


def _build_advice_prompt(rows: list[dict], input_map: dict) -> str:
    from collections import defaultdict

    total = len(rows)
    scored = [r for r in rows if r.get("correctness_score") is not None]
    faithed = [r for r in rows if r.get("faithfulness_score") is not None]
    fuzzy_rows = [r for r in rows if r.get("recall_fuzzy") is not None]
    has_recall = len(fuzzy_rows) > 0
    avg_fuzzy = sum(r["recall_fuzzy"] for r in fuzzy_rows) / len(fuzzy_rows) if has_recall else None
    avg_score = sum(r["correctness_score"] for r in scored) / len(scored) if scored else 0
    avg_faith = sum(r["faithfulness_score"] for r in faithed) / len(faithed) if faithed else 0

    score_dist = {str(i): sum(1 for r in scored if r.get("correctness_score") == i) for i in range(1, 6)}
    faith_dist = {str(i): sum(1 for r in faithed if r.get("faithfulness_score") == i) for i in range(1, 6)}

    # LA-CLA별
    lacla_groups = defaultdict(list)
    for r in rows:
        key = f"{r.get('la', '')} / {r.get('cla', '')}"
        lacla_groups[key].append(r)
    lacla_lines = []
    for key, grp in sorted(lacla_groups.items()):
        g_scored = [r for r in grp if r.get("correctness_score") is not None]
        g_faithed = [r for r in grp if r.get("faithfulness_score") is not None]
        g_score = sum(r["correctness_score"] for r in g_scored) / len(g_scored) if g_scored else 0
        g_faith = sum(r["faithfulness_score"] for r in g_faithed) / len(g_faithed) if g_faithed else 0
        if has_recall:
            g_fuzzy_rows = [r for r in grp if r.get("recall_fuzzy") is not None]
            g_fuzzy = sum(r["recall_fuzzy"] for r in g_fuzzy_rows) / len(g_fuzzy_rows) if g_fuzzy_rows else None
            fuzzy_str = f"{g_fuzzy:.3f}" if g_fuzzy is not None else "N/A"
            lacla_lines.append(
                f"  - {key}: Recall(Fuzzy)={fuzzy_str}, Correctness={g_score:.2f}/5, Faithfulness={g_faith:.2f}/5"
            )
        else:
            lacla_lines.append(
                f"  - {key}: Correctness={g_score:.2f}/5, Faithfulness={g_faith:.2f}/5"
            )

    # 유형별
    type_groups = defaultdict(list)
    for r in rows:
        src = input_map.get(r["qid"], {})
        t = src.get("type", "기타")
        type_groups[t].append(r)
    type_lines = []
    for t, grp in sorted(type_groups.items()):
        g_scored = [r for r in grp if r.get("correctness_score") is not None]
        g_faithed = [r for r in grp if r.get("faithfulness_score") is not None]
        g_score = sum(r["correctness_score"] for r in g_scored) / len(g_scored) if g_scored else 0
        g_faith = sum(r["faithfulness_score"] for r in g_faithed) / len(g_faithed) if g_faithed else 0
        if has_recall:
            g_fuzzy_rows = [r for r in grp if r.get("recall_fuzzy") is not None]
            g_fuzzy = sum(r["recall_fuzzy"] for r in g_fuzzy_rows) / len(g_fuzzy_rows) if g_fuzzy_rows else None
            fuzzy_str = f"{g_fuzzy:.3f}" if g_fuzzy is not None else "N/A"
            type_lines.append(
                f"  - {t} ({len(grp)}개): Recall={fuzzy_str}, Correctness={g_score:.2f}/5, Faithfulness={g_faith:.2f}/5"
            )
        else:
            type_lines.append(
                f"  - {t} ({len(grp)}개): Correctness={g_score:.2f}/5, Faithfulness={g_faith:.2f}/5"
            )

    # 저성능 항목
    low_items = [r for r in rows if r.get("correctness_score") is not None and r["correctness_score"] <= 2]
    low_lines = []
    for r in low_items[:8]:
        src = input_map.get(r["qid"], {})
        q = src.get("question", "")[:45]
        t = src.get("type", "")
        low_lines.append(f"  - qid={r['qid']} [{t}] score={r['correctness_score']}: {q}...")

    # 저신뢰 항목
    low_faith_items = [r for r in rows if r.get("faithfulness_score") is not None and r["faithfulness_score"] <= 2]
    low_faith_lines = []
    for r in low_faith_items[:8]:
        src = input_map.get(r["qid"], {})
        q = src.get("question", "")[:45]
        low_faith_lines.append(f"  - qid={r['qid']} faith={r['faithfulness_score']}: {q}...")

    # Recall 관련 동적 섹션 구성
    if has_recall:
        recall_summary_line = f"\n- Reference Recall (Fuzzy) 평균: {avg_fuzzy:.3f}"
        reference_pipeline_note = " + 참조 조항"
        allowed_types = "recall 또는 correct 또는 faithfulness 또는 lacla 또는 arch 또는 prompt"

        zero_items = [r for r in rows if r.get("recall_fuzzy") is not None and r["recall_fuzzy"] == 0]
        zero_lines = []
        for r in zero_items[:8]:
            gt = r.get("gt_reference", [])
            rag = r.get("rag_reference", [])
            zero_lines.append(f"  - qid={r['qid']}: GT={gt} vs RAG={rag}")
        zero_recall_section = (
            f"[Recall 0 항목 (fuzzy recall = 0)]\n"
            f"{chr(10).join(zero_lines) or '  (없음)'}\n\n"
        )
    else:
        recall_summary_line = "\n- Reference Recall: 측정 불가 (rag_reference 미제공)"
        reference_pipeline_note = ""
        allowed_types = "correct 또는 faithfulness 또는 lacla 또는 arch 또는 prompt"
        zero_recall_section = ""

    return ADVICE_PROMPT.format(
        total=total,
        n_lacla=len(lacla_groups),
        avg_score=avg_score,
        avg_faith=avg_faith,
        score_dist=str(score_dist),
        faith_dist=str(faith_dist),
        lacla_stats="\n".join(lacla_lines) or "  (없음)",
        type_stats="\n".join(type_lines) or "  (없음)",
        low_score_items="\n".join(low_lines) or "  (없음)",
        low_faith_items="\n".join(low_faith_lines) or "  (없음)",
        recall_summary_line=recall_summary_line,
        reference_pipeline_note=reference_pipeline_note,
        allowed_types=allowed_types,
        zero_recall_section=zero_recall_section,
    )


def _generate_advice(rows: list[dict], input_map: dict, client: anthropic.Anthropic) -> list[dict]:
    import re
    prompt = _build_advice_prompt(rows, input_map)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        system=ADVICE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = re.sub(r"```\s*$", "", text).strip()

    start = text.find("[")
    end   = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    candidate = text[start:end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        try:
            import json_repair
            return json_repair.loads(candidate)
        except Exception:
            pass
        return []


def _run_eval():
    global _status
    _status.update({"running": True, "done": False, "error": None, "pct": 0})
    try:
        _status.update({"progress": "데이터 로드 중...", "pct": 5})
        records = load_jsonl(INPUT_FILE)
        total = len(records)

        _status.update({"progress": "Reference Recall 계산 중...", "pct": 10})
        recall_results = evaluate_recall(records)

        # la / cla 병합
        input_map = {r["qid"]: r for r in records}
        for rc in recall_results:
            src = input_map.get(rc["qid"], {})
            rc["la"]  = src.get("la", "")
            rc["cla"] = src.get("cla", "")

        # LLM 평가: Correctness + Faithfulness (10% → 90%)
        client = anthropic.Anthropic(http_client=httpx.Client(verify=False))
        correctness_results = []
        for i, r in enumerate(records):
            pct = 10 + int((i / total) * 80)
            _status.update({
                "progress": f"LLM 평가 중... ({i+1}/{total})",
                "pct": pct,
            })
            c_judgment = judge_correctness(
                client,
                question=r["question"],
                ground_truth=r["answer"],
                rag_answer=r["rag_answer"],
            )
            f_judgment = judge_faithfulness(
                client,
                question=r["question"],
                ground_truth=r["answer"],
                rag_answer=r["rag_answer"],
            )
            correctness_results.append({
                "qid":                 r["qid"],
                "correctness_score":   c_judgment.get("score"),
                "correctness_reason":  c_judgment.get("reason", ""),
                "faithfulness_score":  f_judgment.get("score"),
                "faithfulness_reason": f_judgment.get("reason", ""),
            })

        _status.update({"progress": "결과 저장 중...", "pct": 93})
        correctness_map = {r["qid"]: r for r in correctness_results}
        rows = [{**rc, **correctness_map.get(rc["qid"], {})} for rc in recall_results]
        _Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        _status.update({"progress": "종합 개선 의견 생성 중...", "pct": 96})
        advice = _generate_advice(rows, input_map, client)
        with open(ADVICE_FILE, "w", encoding="utf-8") as f:
            json.dump(advice, f, ensure_ascii=False, indent=2)

        _status.update({"running": False, "done": True, "progress": "완료", "pct": 100})
    except Exception as e:
        _status.update({"running": False, "done": False, "error": str(e), "progress": "", "pct": 0})


# ─── API 엔드포인트 ───────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    return JSONResponse(content=_status)


@app.post("/api/run")
def run_evaluation():
    if _status["running"]:
        return JSONResponse(content={"status": "already_running"})
    thread = threading.Thread(target=_run_eval, daemon=True)
    thread.start()
    return JSONResponse(content={"status": "started"})


@app.get("/api/advice")
def get_advice():
    if not Path(ADVICE_FILE).exists():
        return JSONResponse(content=[])
    with open(ADVICE_FILE, "r", encoding="utf-8") as f:
        return JSONResponse(content=json.load(f))


@app.post("/api/regenerate_advice")
def regenerate_advice():
    if not Path(OUTPUT_FILE).exists():
        return JSONResponse(content={"status": "error", "message": "평가 결과 파일이 없습니다."})
    try:
        rows = load_jsonl(OUTPUT_FILE)
        input_map = {r["qid"]: r for r in load_jsonl(INPUT_FILE)}
        client = anthropic.Anthropic(http_client=httpx.Client(verify=False))
        advice = _generate_advice(rows, input_map, client)
        with open(ADVICE_FILE, "w", encoding="utf-8") as f:
            json.dump(advice, f, ensure_ascii=False, indent=2)
        return JSONResponse(content={"status": "ok", "count": len(advice)})
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)})


@app.post("/api/export")
def export_html():
    from export_html import build_data, generate_html
    from datetime import datetime
    if not Path(OUTPUT_FILE).exists():
        return JSONResponse(content={"status": "error", "message": "평가 결과 파일이 없습니다."})
    results, advice   = build_data()
    generated_at      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp         = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir        = _DATA_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path       = output_dir / f"rag_eval_report_{timestamp}.html"
    output_path.write_text(generate_html(results, advice, generated_at), encoding="utf-8")
    return JSONResponse(content={"status": "ok", "filename": output_path.name})


@app.get("/api/results")
def get_results():
    if not Path(OUTPUT_FILE).exists():
        return JSONResponse(content=[])
    results = load_jsonl(OUTPUT_FILE)
    if Path(INPUT_FILE).exists():
        input_map = {r["qid"]: r for r in load_jsonl(INPUT_FILE)}
        for row in results:
            src = input_map.get(row["qid"], {})
            row["answer"]         = src.get("answer", "")
            row["rag_answer"]     = src.get("rag_answer", "")
            row["question"]   = src.get("question", row.get("question", ""))
    return JSONResponse(content=results)


# ─── HTML 대시보드 ────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG 평가 대시보드</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #f0f2f5; color: #1a1a2e; }

  header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: white; padding: 20px 32px;
    display: flex; align-items: center; justify-content: space-between;
  }
  header h1 { font-size: 1.4rem; font-weight: 600; letter-spacing: 0.02em; }
  header small { opacity: 0.6; font-size: 0.8rem; }

  .container { max-width: 1280px; margin: 0 auto; padding: 24px; }

  /* 컨트롤 */
  .controls { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
  .progress-wrap { margin-bottom: 16px; display: none; }
  .progress-wrap.visible { display: block; }
  .progress-bar-bg { background: #e9ecef; border-radius: 8px; height: 10px; overflow: hidden; }
  .progress-bar-fill { height: 100%; border-radius: 8px; background: linear-gradient(90deg, #0f3460, #0d6efd); transition: width .5s ease; }
  .progress-label { font-size: 0.8rem; color: #666; margin-top: 5px; }
  button {
    padding: 10px 22px; border: none; border-radius: 8px;
    font-size: 0.9rem; font-weight: 600; cursor: pointer; transition: all .2s;
  }
  #runBtn { background: #0f3460; color: white; }
  #runBtn:hover { background: #16213e; }
  #runBtn:disabled { background: #888; cursor: not-allowed; }
  #statusBadge { padding: 6px 14px; border-radius: 20px; font-size: 0.82rem; font-weight: 600; }
  .badge-idle    { background: #e8e8e8; color: #555; }
  .badge-running { background: #fff3cd; color: #856404; }
  .badge-done    { background: #d1e7dd; color: #0a3622; }
  .badge-error   { background: #f8d7da; color: #842029; }

  /* 탭 */
  .tabs { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 20px; border-bottom: 2px solid #e9ecef; padding-bottom: 0; }
  .tab-btn {
    padding: 6px 14px; border: none; border-radius: 8px 8px 0 0;
    font-size: 0.78rem; font-weight: 600; cursor: pointer;
    background: #e9ecef; color: #555; transition: all .15s;
    position: relative; bottom: -2px; border-bottom: 2px solid transparent;
    display: flex; flex-direction: column; align-items: center; line-height: 1.4;
  }
  .tab-btn .tab-la  { font-size: 0.78rem; font-weight: 700; }
  .tab-btn .tab-cla { font-size: 0.72rem; font-weight: 400; opacity: 0.8; }
  .tab-btn:hover { background: #d0d8e8; color: #333; }
  .tab-btn.active { background: white; color: #0f3460; border-bottom: 2px solid white; box-shadow: 0 -2px 6px rgba(0,0,0,.06); }

  /* 요약 카드 */
  .cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 20px; }
  .card { background: white; border-radius: 12px; padding: 20px 24px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
  .card .label { font-size: 0.78rem; color: #666; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 6px; }
  .card .value { font-size: 2rem; font-weight: 700; }
  .card .sub   { font-size: 0.75rem; color: #999; margin-top: 4px; }

  /* 차트 */
  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
  .chart-card { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
  .chart-card h3 { font-size: 0.88rem; color: #444; margin-bottom: 14px; }

  /* 테이블 */
  .table-wrap { background: white; border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,.08); overflow: hidden; }
  .table-wrap h3 { padding: 16px 20px; border-bottom: 1px solid #f0f0f0; font-size: 0.92rem; color: #333; }
  .table-scroll-top { overflow-x: scroll; overflow-y: hidden; height: 14px; border-bottom: 1px solid #e9ecef; background: #f8f9fa; }
  .table-scroll-top::-webkit-scrollbar { height: 10px; }
  .table-scroll-top::-webkit-scrollbar-thumb { background: #ccc; border-radius: 5px; }
  .table-scroll { overflow-x: auto; overflow-y: auto; max-height: 520px; }
  .table-scroll.no-vscroll { overflow-y: visible; max-height: none; }
  .table-scroll-top-inner { height: 1px; }
  table { width: max-content; min-width: 100%; border-collapse: collapse; font-size: 0.84rem; }
  thead th {
    background: #f8f9fa; padding: 10px 14px;
    text-align: left; font-weight: 600; color: #555;
    border-bottom: 2px solid #e9ecef; white-space: nowrap;
    position: sticky; top: 0; z-index: 2;
  }
  tbody tr:hover { background: #fafbff; }
  tbody td { padding: 10px 14px; border-bottom: 1px solid #f0f0f0; vertical-align: top; }
  tbody tr:last-child td { border-bottom: none; }

  .q-text { max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #333; }
  .ans-cell { min-width: 260px; max-width: 340px; line-height: 1.55; color: #222; white-space: pre-wrap; word-break: break-word; font-size: 0.8rem; }
  .ref-chip { display: inline-block; background: #e8eaf6; color: #3949ab; border-radius: 4px; padding: 2px 7px; font-size: 0.78rem; margin: 1px; }
  .ref-chip.rag { background: #fce4ec; color: #c62828; }

  .score { display: inline-block; width: 28px; height: 28px; border-radius: 50%; text-align: center; line-height: 28px; font-weight: 700; font-size: 0.85rem; }
  .s5 { background: #198754; color: white; }
  .s4 { background: #5cb85c; color: white; }
  .s3 { background: #f0ad4e; color: white; }
  .s2 { background: #fd7e14; color: white; }
  .s1 { background: #dc3545; color: white; }

  .rc { font-weight: 600; }
  .rc-hit  { color: #198754; }
  .rc-part { color: #f0ad4e; }
  .rc-miss { color: #dc3545; }

  .reason { min-width: 220px; max-width: 300px; color: #555; line-height: 1.4; font-size: 0.8rem; word-break: break-word; }
  .lacla-badge { display: inline-block; background: #e3f2fd; color: #1565c0; border-radius: 4px; padding: 1px 6px; font-size: 0.75rem; white-space: nowrap; }

  .empty { text-align: center; padding: 40px; color: #aaa; font-size: 0.95rem; }

  .sort-btn {
    padding: 5px 11px; border: 1px solid #d0d7de; border-radius: 6px;
    font-size: 0.76rem; font-weight: 600; cursor: pointer;
    background: #f6f8fa; color: #555; transition: all .15s;
  }
  .sort-btn:hover { background: #e9ecef; color: #333; }
  .sort-btn.active { background: #0f3460; color: white; border-color: #0f3460; }

  /* 개선 의견 */
  .advice { background: white; border-radius: 12px; padding: 24px 28px; box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-top: 24px; }
  .advice h3 { font-size: 1rem; color: #1a1a2e; margin-bottom: 18px; }
  .advice-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; }
  .advice-card { border-left: 4px solid; border-radius: 0 8px 8px 0; padding: 14px 16px; background: #fafbff; }
  .advice-card.recall        { border-color: #0d6efd; }
  .advice-card.correct       { border-color: #198754; }
  .advice-card.faithfulness  { border-color: #9c27b0; }
  .advice-card.lacla         { border-color: #e91e63; }
  .advice-card.arch          { border-color: #f0ad4e; }
  .advice-card.prompt        { border-color: #00897b; }
  .advice-card h4 { font-size: 0.85rem; font-weight: 700; margin-bottom: 8px; color: #333; }
  .advice-card ul { padding-left: 16px; margin: 0; }
  .advice-card li { font-size: 0.81rem; color: #555; line-height: 1.7; }
</style>
</head>
<body>

<header>
  <div>
    <h1>RAG 평가 대시보드</h1>
    <small>전체 정책 · 7개 LA-CLA</small>
  </div>
  <small id="lastUpdated"></small>
</header>

<div class="container">

  <!-- 컨트롤 -->
  <div class="controls">
    <button id="runBtn" onclick="runEval()">▶ 평가 실행</button>
    <button id="exportBtn" onclick="exportHtml()" style="background:#198754;color:white;">📄 리포트 저장</button>
    <span id="statusBadge" class="badge-idle">대기 중</span>
    <span id="progressMsg" style="font-size:0.83rem;color:#666;"></span>
  </div>
  <div class="progress-wrap" id="progressWrap">
    <div class="progress-bar-bg">
      <div class="progress-bar-fill" id="progressFill" style="width:0%"></div>
    </div>
    <div class="progress-label" id="progressLabel">0%</div>
  </div>

  <!-- 탭 -->
  <div class="tabs" id="tabBar">
    <button class="tab-btn active" onclick="selectTab('ALL')" data-key="ALL">전체</button>
  </div>

  <!-- 요약 카드 -->
  <div class="cards">
    <div class="card">
      <div class="label">Recall (Fuzzy)</div>
      <div class="value" id="avgFuzzy">—</div>
      <div class="sub">부분 일치 (하위항 포함) 기준</div>
    </div>
    <div class="card">
      <div class="label">Correctness (LLM)</div>
      <div class="value" id="avgScore">—</div>
      <div class="sub">1~5점 · Claude 평가</div>
    </div>
    <div class="card">
      <div class="label">Faithfulness (LLM)</div>
      <div class="value" id="avgFaith">—</div>
      <div class="sub">1~5점 · Hallucination 여부</div>
    </div>
  </div>

  <!-- 차트 -->
  <div class="charts">
    <div class="chart-card">
      <h3>항목별 Reference Recall (Fuzzy)</h3>
      <canvas id="recallChart" height="200"></canvas>
    </div>
    <div class="chart-card">
      <h3>항목별 Correctness / Faithfulness</h3>
      <canvas id="scoreChart" height="200"></canvas>
    </div>
  </div>

  <!-- 상세 테이블 -->
  <div class="table-wrap">
    <div style="display:flex;align-items:center;justify-content:space-between;padding:14px 20px;border-bottom:1px solid #f0f0f0;">
      <h3 id="tableTitle" style="margin:0">항목별 상세 결과</h3>
      <div style="display:flex;gap:6px;align-items:center;">
        <span style="font-size:0.78rem;color:#888;">정렬:</span>
        <button class="sort-btn active" data-sort="correctness" onclick="setSort('correctness')">Correctness ↑</button>
        <button class="sort-btn" data-sort="qid" onclick="setSort('qid')">기본 (qid)</button>
      </div>
    </div>
    <div class="table-scroll-top" id="tableScrollTop">
      <div class="table-scroll-top-inner" id="tableScrollTopInner"></div>
    </div>
    <div class="table-scroll" id="tableScroll">
      <div id="tableBody"></div>
    </div>
  </div>

  <!-- 종합 개선 의견 -->
  <div class="advice" id="adviceSection" style="display:none">
    <h3>💡 종합 개선 의견 <small id="adviceSubtitle" style="font-size:0.75rem;font-weight:400;color:#888;"></small></h3>
    <div class="advice-grid" id="adviceGrid"></div>
  </div>

</div>

<script>
let recallChart, scoreChart;
let allData = [];
let currentTab = 'ALL';
let currentSort = 'correctness';

// ── 정렬 ──────────────────────────────────────────────────────────────────────
function setSort(key) {
  currentSort = key;
  document.querySelectorAll('.sort-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.sort === key);
  });
  selectTab(currentTab);
}

function sortData(data) {
  const d = [...data];
  if (currentSort === 'correctness') {
    d.sort((a, b) => (a.correctness_score ?? 99) - (b.correctness_score ?? 99));
  } else if (currentSort === 'faithfulness') {
    d.sort((a, b) => (a.faithfulness_score ?? 99) - (b.faithfulness_score ?? 99));
  } else if (currentSort === 'recall') {
    d.sort((a, b) => (a.recall_fuzzy ?? 99) - (b.recall_fuzzy ?? 99));
  }
  // 'qid'는 원래 순서 유지
  return d;
}

// ── 탭 생성 ───────────────────────────────────────────────────────────────────
function buildTabs(data) {
  const bar = document.getElementById('tabBar');
  // 기존 탭 (전체 제외) 제거
  Array.from(bar.querySelectorAll('[data-key]:not([data-key="ALL"])')).forEach(el => el.remove());

  const seen = new Set();
  data.forEach(r => {
    const key = r.la + '_' + r.cla;
    if (!seen.has(key)) {
      seen.add(key);
      const btn = document.createElement('button');
      btn.className = 'tab-btn';
      btn.dataset.key = key;
      btn.innerHTML = `<span class="tab-la">${r.la}</span><span class="tab-cla">${r.cla}</span>`;
      btn.onclick = () => selectTab(key);
      bar.appendChild(btn);
    }
  });
}

function selectTab(key) {
  currentTab = key;
  document.querySelectorAll('.tab-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.key === key);
  });

  const isAll = key === 'ALL';
  const filtered = isAll ? allData : allData.filter(r => (r.la + '_' + r.cla) === key);

  const title = isAll
    ? '항목별 상세 결과 (전체 ' + allData.length + '개)'
    : '항목별 상세 결과 — ' + key.replace('_', ' · ') + ' (' + filtered.length + '개)';
  document.getElementById('tableTitle').textContent = title;

  // 세로 스크롤: 전체 탭에서만 활성화
  const tableScroll = document.getElementById('tableScroll');
  tableScroll.classList.toggle('no-vscroll', !isAll);

  // 종합 개선 의견: 전체 탭에서만 표시
  const adviceSection = document.getElementById('adviceSection');
  if (adviceSection.dataset.hasContent === 'true') {
    adviceSection.style.display = isAll ? 'block' : 'none';
  }

  renderView(sortData(filtered), !isAll);
}

// ── 차트 초기화 ───────────────────────────────────────────────────────────────
function initCharts() {
  const recallCtx = document.getElementById('recallChart').getContext('2d');
  recallChart = new Chart(recallCtx, {
    type: 'bar',
    data: { labels: [], datasets: [
      { label: 'Fuzzy', data: [], backgroundColor: 'rgba(13,110,253,0.6)', borderRadius: 4 },
    ]},
    options: {
      scales: { y: { min: 0, max: 1, ticks: { stepSize: 0.5 } } },
      plugins: { legend: { display: false } },
      animation: { duration: 600 }
    }
  });

  const scoreCtx = document.getElementById('scoreChart').getContext('2d');
  scoreChart = new Chart(scoreCtx, {
    type: 'bar',
    data: { labels: [], datasets: [
      { label: 'Correctness', data: [], backgroundColor: 'rgba(25,135,84,0.7)', borderRadius: 4 },
      { label: 'Faithfulness', data: [], backgroundColor: 'rgba(153,102,255,0.6)', borderRadius: 4 },
    ]},
    options: {
      scales: { y: { min: 0, max: 5, ticks: { stepSize: 1 } } },
      plugins: { legend: { position: 'top' } },
      animation: { duration: 600 }
    }
  });
}

// ── 뷰 렌더링 (카드 + 차트 + 테이블) ─────────────────────────────────────────
function scoreClass(v) {
  return v >= 5 ? 's5' : v >= 4 ? 's4' : v >= 3 ? 's3' : v >= 2 ? 's2' : 's1';
}

function renderView(data, showLaCla) {
  if (!data.length) {
    document.getElementById('tableBody').innerHTML = '<div class="empty">결과가 없습니다.</div>';
    document.getElementById('avgFuzzy').textContent = '—';
    document.getElementById('avgScore').textContent = '—';
    document.getElementById('avgFaith').textContent = '—';
    return;
  }

  // 요약 카드
  const fuzzyData = data.filter(r => r.recall_fuzzy != null);
  const avgFuzzy  = fuzzyData.length ? fuzzyData.reduce((s, r) => s + r.recall_fuzzy, 0) / fuzzyData.length : null;
  const scored    = data.filter(r => r.correctness_score != null);
  const faithed   = data.filter(r => r.faithfulness_score != null);
  const avgScore  = scored.length  ? scored.reduce((s, r) => s + r.correctness_score, 0)  / scored.length  : null;
  const avgFaith  = faithed.length ? faithed.reduce((s, r) => s + r.faithfulness_score, 0) / faithed.length : null;

  document.getElementById('avgFuzzy').textContent = avgFuzzy != null ? avgFuzzy.toFixed(3) : '—';
  document.getElementById('avgScore').textContent = avgScore != null ? avgScore.toFixed(2) + ' / 5' : '—';
  document.getElementById('avgFaith').textContent = avgFaith != null ? avgFaith.toFixed(2) + ' / 5' : '—';

  // 차트
  const labels = data.map(r => 'qid ' + r.qid);
  recallChart.data.labels = labels;
  recallChart.data.datasets[0].data = data.map(r => r.recall_fuzzy ?? null);
  recallChart.update();

  scoreChart.data.labels = labels;
  scoreChart.data.datasets[0].data = data.map(r => r.correctness_score ?? 0);
  scoreChart.data.datasets[1].data = data.map(r => r.faithfulness_score ?? 0);
  scoreChart.update();

  // 테이블
  const escHtml = s => (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const laClaTh = showLaCla ? '' : '<th>LA / CLA</th>';

  const rows = data.map(r => {
    const rf  = r.recall_fuzzy;
    const sc  = r.correctness_score;
    const fa  = r.faithfulness_score;
    const rcCls  = rf == null ? 'rc-miss' : rf === 1 ? 'rc-hit' : rf > 0 ? 'rc-part' : 'rc-miss';
    const rfHtml = rf != null ? rf.toFixed(2) : '—';
    const scHtml = sc != null ? `<span class="score ${scoreClass(sc)}">${sc}</span>` : '—';
    const faHtml = fa != null ? `<span class="score ${scoreClass(fa)}">${fa}</span>` : '—';
    const gtChips  = (r.gt_reference  || []).map(x => `<span class="ref-chip">${x}</span>`).join('');
    const ragChips = (r.rag_reference || []).map(x => `<span class="ref-chip rag">${x}</span>`).join('');
    const laClaTd  = showLaCla ? '' : `<td><span class="lacla-badge">${r.la}<br>${r.cla}</span></td>`;

    return `<tr>
      <td style="font-weight:600;color:#0f3460;white-space:nowrap">${r.qid}</td>
      <td class="q-text" title="${(r.question||'').replace(/"/g,'&quot;')}">${r.question || ''}</td>
      <td class="rc ${rcCls}">${rfHtml}</td>
      <td>${scHtml}</td>
      <td class="reason">${escHtml(r.correctness_reason || '')}</td>
      <td>${faHtml}</td>
      <td class="reason">${escHtml(r.faithfulness_reason || '')}</td>
      <td class="ans-cell">${escHtml(r.answer)}</td>
      <td class="ans-cell">${escHtml(r.rag_answer)}</td>
      <td>${gtChips}</td>
      <td>${ragChips}</td>
      ${laClaTd}
    </tr>`;
  }).join('');

  document.getElementById('tableBody').innerHTML = `
    <table>
      <thead><tr>
        <th>qid</th><th>질문</th>
        <th>Recall<br>Fuzzy</th>
        <th>Correctness</th><th>Correctness 사유</th>
        <th>Faithfulness</th><th>Faithfulness 사유</th>
        <th>모범 답변</th><th>RAG 답변</th>
        <th>GT Reference</th><th>RAG Reference</th>
        ${laClaTh}
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;

  document.getElementById('lastUpdated').textContent =
    '마지막 업데이트: ' + new Date().toLocaleTimeString('ko-KR');
}

// ── 개선 의견 렌더링 ──────────────────────────────────────────────────────────
const TYPE_CLASS = { recall: 'recall', correct: 'correct', faithfulness: 'faithfulness', lacla: 'lacla', arch: 'arch', prompt: 'prompt' };

async function loadAdvice() {
  const res   = await fetch('/api/advice');
  const cards = await res.json();
  const section = document.getElementById('adviceSection');
  const grid    = document.getElementById('adviceGrid');
  const sub     = document.getElementById('adviceSubtitle');

  if (!cards.length) { section.style.display = 'none'; return; }

  sub.textContent = '전체 평가 데이터 기반으로 Claude가 생성한 의견';
  grid.innerHTML = cards.map(c => {
    const cls = TYPE_CLASS[c.type] || 'arch';
    const bullets = (c.bullets || []).map(b => `<li>${b}</li>`).join('');
    return `<div class="advice-card ${cls}"><h4>${c.title}</h4><ul>${bullets}</ul></div>`;
  }).join('');
  section.dataset.hasContent = 'true';
  section.style.display = currentTab === 'ALL' ? 'block' : 'none';
}

// ── 결과 로드 ─────────────────────────────────────────────────────────────────
async function loadResults() {
  const res  = await fetch('/api/results');
  allData = await res.json();
  if (allData.length) {
    buildTabs(allData);
    selectTab(currentTab);
  } else {
    document.getElementById('tableBody').innerHTML =
      '<div class="empty">결과가 없습니다. 평가를 실행하세요.</div>';
  }
  await loadAdvice();
}

// ── 리포트 저장 ───────────────────────────────────────────────────────────────
async function exportHtml() {
  const btn  = document.getElementById('exportBtn');
  const prog = document.getElementById('progressMsg');
  btn.disabled = true;
  btn.textContent = '📄 저장 중...';
  prog.textContent = 'HTML 리포트 생성 중...';
  try {
    const res  = await fetch('/api/export', { method: 'POST' });
    const data = await res.json();
    if (data.status === 'ok') {
      prog.textContent = '✓ 저장 완료: ' + data.filename;
      setTimeout(() => { prog.textContent = ''; }, 4000);
    } else {
      prog.textContent = '오류: ' + data.message;
    }
  } finally {
    btn.disabled = false;
    btn.textContent = '📄 리포트 저장';
  }
}

// ── 평가 실행 ─────────────────────────────────────────────────────────────────
async function runEval() {
  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  await fetch('/api/run', { method: 'POST' });
  pollStatus();
}

function pollStatus() {
  const badge = document.getElementById('statusBadge');
  const prog  = document.getElementById('progressMsg');
  const btn   = document.getElementById('runBtn');
  const progressWrap  = document.getElementById('progressWrap');
  const progressFill  = document.getElementById('progressFill');
  const progressLabel = document.getElementById('progressLabel');

  const timer = setInterval(async () => {
    const res  = await fetch('/api/status');
    const stat = await res.json();

    if (stat.running) {
      badge.className = 'badge-running';
      badge.textContent = '실행 중';
      prog.textContent = stat.progress || '';
      progressWrap.classList.add('visible');
      progressFill.style.width = (stat.pct || 0) + '%';
      progressLabel.textContent = (stat.pct || 0) + '%';
    } else if (stat.done) {
      clearInterval(timer);
      badge.className = 'badge-done';
      badge.textContent = '완료';
      prog.textContent = '';
      btn.disabled = false;
      progressFill.style.width = '100%';
      progressLabel.textContent = '100%';
      setTimeout(() => progressWrap.classList.remove('visible'), 2000);
      await loadResults();
    } else if (stat.error) {
      clearInterval(timer);
      badge.className = 'badge-error';
      badge.textContent = '오류';
      prog.textContent = stat.error;
      btn.disabled = false;
      progressWrap.classList.remove('visible');
    }
  }, 1500);
}

// ── 상단 미러 스크롤바 ─────────────────────────────────────────────────────────
function initMirrorScroll() {
  const top   = document.getElementById('tableScrollTop');
  const inner = document.getElementById('tableScrollTopInner');
  const bot   = document.getElementById('tableScroll');

  function syncInnerWidth() { inner.style.width = bot.scrollWidth + 'px'; }
  syncInnerWidth();
  new ResizeObserver(syncInnerWidth).observe(bot);

  let syncing = false;
  top.addEventListener('scroll', () => { if (!syncing) { syncing = true; bot.scrollLeft = top.scrollLeft; syncing = false; } });
  bot.addEventListener('scroll', () => { if (!syncing) { syncing = true; top.scrollLeft = bot.scrollLeft; syncing = false; } });
}

// ── 초기화 ────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  initCharts();
  initMirrorScroll();
  await loadResults();

  const res  = await fetch('/api/status');
  const stat = await res.json();
  if (stat.running) {
    document.getElementById('runBtn').disabled = true;
    pollStatus();
  }
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


# ─── 실행 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== RAG 평가 대시보드 ===")
    print("http://localhost:8000 에서 확인하세요.\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
