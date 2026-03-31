"""
RAG 평가 결과를 standalone HTML 파일로 내보내기

실행: python export_html.py
출력: rag_eval_report_YYYYMMDD_HHMMSS.html  (서버 없이 브라우저에서 바로 열 수 있음)
"""

import json
from datetime import datetime
from pathlib import Path

RESULT_FILE  = Path(__file__).parent / "all_policies_eval_result.jsonl"
INPUT_FILE   = Path(__file__).parent / "all_policies_rag_answer.jsonl"
ADVICE_FILE  = Path(__file__).parent / "all_policies_advice.json"
PREPROCESSED = Path(__file__).parent / "preprocessed_files"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_context_lengths() -> dict:
    lengths = {}
    for md in PREPROCESSED.glob("*.md"):
        lengths[md.stem] = len(md.read_text(encoding="utf-8"))
    return lengths


def build_data() -> tuple[list[dict], list[dict]]:
    results   = load_jsonl(RESULT_FILE)
    inputs    = load_jsonl(INPUT_FILE)
    advice    = json.loads(ADVICE_FILE.read_text(encoding="utf-8")) if ADVICE_FILE.exists() else []
    ctx_lens  = load_context_lengths()

    input_map = {r["qid"]: r for r in inputs}
    for row in results:
        src = input_map.get(row["qid"], {})
        row["answer"]         = src.get("answer", "")
        row["rag_answer"]     = src.get("rag_answer", "")
        row["question"]       = src.get("question", row.get("question", ""))
        la_cla = f"{row.get('la', '')}_{row.get('cla', '')}"
        row["context_length"] = ctx_lens.get(la_cla, 0)

    return results, advice


def generate_html(results: list[dict], advice: list[dict], generated_at: str) -> str:
    results_js = json.dumps(results, ensure_ascii=False)
    advice_js  = json.dumps(advice,  ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG 평가 리포트 · {generated_at}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #f0f2f5; color: #1a1a2e; }}

  header {{
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: white; padding: 20px 32px;
    display: flex; align-items: center; justify-content: space-between;
  }}
  header h1 {{ font-size: 1.4rem; font-weight: 600; letter-spacing: 0.02em; }}
  header small {{ opacity: 0.6; font-size: 0.8rem; }}

  .container {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}

  .export-banner {{
    background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px;
    padding: 10px 16px; margin-bottom: 20px; font-size: 0.83rem; color: #856404;
  }}

  .tabs {{ display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 20px; border-bottom: 2px solid #e9ecef; }}
  .tab-btn {{
    padding: 6px 14px; border: none; border-radius: 8px 8px 0 0;
    font-size: 0.78rem; font-weight: 600; cursor: pointer;
    background: #e9ecef; color: #555; transition: all .15s;
    position: relative; bottom: -2px; border-bottom: 2px solid transparent;
    display: flex; flex-direction: column; align-items: center; line-height: 1.4;
  }}
  .tab-btn .tab-la  {{ font-size: 0.78rem; font-weight: 700; }}
  .tab-btn .tab-cla {{ font-size: 0.72rem; font-weight: 400; opacity: 0.8; }}
  .tab-btn .tab-ctx {{ font-size: 0.68rem; font-weight: 400; opacity: 0.6; }}
  .tab-btn:hover {{ background: #d0d8e8; color: #333; }}
  .tab-btn.active {{ background: white; color: #0f3460; border-bottom: 2px solid white; box-shadow: 0 -2px 6px rgba(0,0,0,.06); }}

  .cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 20px; }}
  .card {{ background: white; border-radius: 12px; padding: 20px 24px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  .card .label {{ font-size: 0.78rem; color: #666; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 6px; }}
  .card .value {{ font-size: 2rem; font-weight: 700; }}
  .card .sub   {{ font-size: 0.75rem; color: #999; margin-top: 4px; }}

  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }}
  .chart-card {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  .chart-card h3 {{ font-size: 0.88rem; color: #444; margin-bottom: 14px; }}

  .table-wrap {{ background: white; border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,.08); overflow: hidden; }}
  .table-scroll-top {{ overflow-x: scroll; overflow-y: hidden; height: 14px; border-bottom: 1px solid #e9ecef; background: #f8f9fa; }}
  .table-scroll-top::-webkit-scrollbar {{ height: 10px; }}
  .table-scroll-top::-webkit-scrollbar-thumb {{ background: #ccc; border-radius: 5px; }}
  .table-scroll {{ overflow-x: auto; overflow-y: auto; max-height: 520px; }}
  .table-scroll.no-vscroll {{ overflow-y: visible; max-height: none; }}
  .table-scroll-top-inner {{ height: 1px; }}
  table {{ width: max-content; min-width: 100%; border-collapse: collapse; font-size: 0.84rem; }}
  thead th {{
    background: #f8f9fa; padding: 10px 14px;
    text-align: left; font-weight: 600; color: #555;
    border-bottom: 2px solid #e9ecef; white-space: nowrap;
    position: sticky; top: 0; z-index: 2;
  }}
  tbody tr:hover {{ background: #fafbff; }}
  tbody td {{ padding: 10px 14px; border-bottom: 1px solid #f0f0f0; vertical-align: top; }}
  tbody tr:last-child td {{ border-bottom: none; }}

  .q-text {{ max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #333; }}
  .ans-cell {{ min-width: 260px; max-width: 340px; line-height: 1.55; color: #222; white-space: pre-wrap; word-break: break-word; font-size: 0.8rem; }}
  .ref-chip {{ display: inline-block; background: #e8eaf6; color: #3949ab; border-radius: 4px; padding: 2px 7px; font-size: 0.78rem; margin: 1px; }}
  .ref-chip.rag {{ background: #fce4ec; color: #c62828; }}

  .score {{ display: inline-block; width: 28px; height: 28px; border-radius: 50%; text-align: center; line-height: 28px; font-weight: 700; font-size: 0.85rem; }}
  .s5 {{ background: #198754; color: white; }}
  .s4 {{ background: #5cb85c; color: white; }}
  .s3 {{ background: #f0ad4e; color: white; }}
  .s2 {{ background: #fd7e14; color: white; }}
  .s1 {{ background: #dc3545; color: white; }}

  .rc {{ font-weight: 600; }}
  .rc-hit  {{ color: #198754; }}
  .rc-part {{ color: #f0ad4e; }}
  .rc-miss {{ color: #dc3545; }}

  .reason {{ min-width: 220px; max-width: 300px; color: #555; line-height: 1.4; font-size: 0.8rem; word-break: break-word; }}
  .lacla-badge {{ display: inline-block; background: #e3f2fd; color: #1565c0; border-radius: 4px; padding: 1px 6px; font-size: 0.75rem; white-space: nowrap; }}
  .empty {{ text-align: center; padding: 40px; color: #aaa; font-size: 0.95rem; }}

  .sort-btn {{
    padding: 5px 11px; border: 1px solid #d0d7de; border-radius: 6px;
    font-size: 0.76rem; font-weight: 600; cursor: pointer;
    background: #f6f8fa; color: #555; transition: all .15s;
  }}
  .sort-btn:hover {{ background: #e9ecef; color: #333; }}
  .sort-btn.active {{ background: #0f3460; color: white; border-color: #0f3460; }}

  .advice {{ background: white; border-radius: 12px; padding: 24px 28px; box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-top: 24px; }}
  .advice h3 {{ font-size: 1rem; color: #1a1a2e; margin-bottom: 18px; }}
  .advice-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; }}
  .advice-card {{ border-left: 4px solid; border-radius: 0 8px 8px 0; padding: 14px 16px; background: #fafbff; }}
  .advice-card.recall       {{ border-color: #0d6efd; }}
  .advice-card.correct      {{ border-color: #198754; }}
  .advice-card.faithfulness {{ border-color: #9c27b0; }}
  .advice-card.lacla        {{ border-color: #e91e63; }}
  .advice-card.arch         {{ border-color: #f0ad4e; }}
  .advice-card.prompt       {{ border-color: #00897b; }}
  .advice-card h4 {{ font-size: 0.85rem; font-weight: 700; margin-bottom: 8px; color: #333; }}
  .advice-card ul {{ padding-left: 16px; margin: 0; }}
  .advice-card li {{ font-size: 0.81rem; color: #555; line-height: 1.7; }}
</style>
</head>
<body>

<header>
  <div>
    <h1>RAG 평가 리포트</h1>
    <small>생성일시: {generated_at}</small>
  </div>
</header>

<div class="container">

  <div class="export-banner">
    📄 이 파일은 서버 없이 브라우저에서 바로 열 수 있는 standalone 리포트입니다. (인터넷 연결 필요: Chart.js CDN)
  </div>

  <div class="tabs" id="tabBar">
    <button class="tab-btn active" onclick="selectTab('ALL')" data-key="ALL">전체</button>
  </div>

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

  <div class="advice" id="adviceSection" style="display:none">
    <h3>💡 종합 개선 의견 <small style="font-size:0.75rem;font-weight:400;color:#888;">전체 평가 데이터 기반 Claude 생성</small></h3>
    <div class="advice-grid" id="adviceGrid"></div>
  </div>

</div>

<script>
const EMBEDDED_DATA   = {results_js};
const EMBEDDED_ADVICE = {advice_js};

let recallChart, scoreChart;
let allData     = EMBEDDED_DATA;
let currentTab  = 'ALL';
let currentSort = 'correctness';

function setSort(key) {{
  currentSort = key;
  document.querySelectorAll('.sort-btn').forEach(b => b.classList.toggle('active', b.dataset.sort === key));
  selectTab(currentTab);
}}

function sortData(data) {{
  const d = [...data];
  if (currentSort === 'correctness') d.sort((a, b) => (a.correctness_score ?? 99) - (b.correctness_score ?? 99));
  return d;
}}

function buildTabs(data) {{
  const bar  = document.getElementById('tabBar');
  Array.from(bar.querySelectorAll('[data-key]:not([data-key="ALL"])')).forEach(el => el.remove());
  const seen = new Set();
  data.forEach(r => {{
    const key = r.la + '_' + r.cla;
    if (!seen.has(key)) {{
      seen.add(key);
      const btn = document.createElement('button');
      btn.className   = 'tab-btn';
      btn.dataset.key = key;
      const ctxLen = (r.context_length || 0).toLocaleString() + ' 자';
      btn.innerHTML   = `<span class="tab-la">${{r.la}}</span><span class="tab-cla">${{r.cla}}</span><span class="tab-ctx">${{ctxLen}}</span>`;
      btn.onclick     = () => selectTab(key);
      bar.appendChild(btn);
    }}
  }});
}}

function selectTab(key) {{
  currentTab = key;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.key === key));
  const isAll    = key === 'ALL';
  const filtered = isAll ? allData : allData.filter(r => (r.la + '_' + r.cla) === key);
  const title    = isAll
    ? '항목별 상세 결과 (전체 ' + allData.length + '개)'
    : '항목별 상세 결과 — ' + key.replace('_', ' · ') + ' (' + filtered.length + '개)';
  document.getElementById('tableTitle').textContent = title;
  document.getElementById('tableScroll').classList.toggle('no-vscroll', !isAll);
  const adv = document.getElementById('adviceSection');
  if (adv.dataset.hasContent === 'true') adv.style.display = isAll ? 'block' : 'none';
  renderView(sortData(filtered), !isAll);
}}

function initCharts() {{
  recallChart = new Chart(document.getElementById('recallChart').getContext('2d'), {{
    type: 'bar',
    data: {{ labels: [], datasets: [{{ label: 'Fuzzy', data: [], backgroundColor: 'rgba(13,110,253,0.6)', borderRadius: 4 }}] }},
    options: {{ scales: {{ y: {{ min: 0, max: 1, ticks: {{ stepSize: 0.5 }} }} }}, plugins: {{ legend: {{ display: false }} }}, animation: {{ duration: 600 }} }}
  }});
  scoreChart = new Chart(document.getElementById('scoreChart').getContext('2d'), {{
    type: 'bar',
    data: {{ labels: [], datasets: [
      {{ label: 'Correctness',  data: [], backgroundColor: 'rgba(25,135,84,0.7)',  borderRadius: 4 }},
      {{ label: 'Faithfulness', data: [], backgroundColor: 'rgba(153,102,255,0.6)', borderRadius: 4 }},
    ]}},
    options: {{ scales: {{ y: {{ min: 0, max: 5, ticks: {{ stepSize: 1 }} }} }}, plugins: {{ legend: {{ position: 'top' }} }}, animation: {{ duration: 600 }} }}
  }});
}}

function scoreClass(v) {{ return v>=5?'s5':v>=4?'s4':v>=3?'s3':v>=2?'s2':'s1'; }}

function renderView(data, showLaCla) {{
  if (!data.length) {{
    document.getElementById('tableBody').innerHTML = '<div class="empty">결과가 없습니다.</div>';
    ['avgFuzzy','avgScore','avgFaith'].forEach(id => document.getElementById(id).textContent = '—');
    return;
  }}
  const fuzzyData = data.filter(r => r.recall_fuzzy != null);
  const avgFuzzy  = fuzzyData.length ? fuzzyData.reduce((s,r) => s+r.recall_fuzzy, 0)/fuzzyData.length : null;
  const scored    = data.filter(r => r.correctness_score  != null);
  const faithed   = data.filter(r => r.faithfulness_score != null);
  const avgScore  = scored.length  ? scored.reduce((s,r)  => s+r.correctness_score,  0)/scored.length  : null;
  const avgFaith  = faithed.length ? faithed.reduce((s,r) => s+r.faithfulness_score, 0)/faithed.length : null;

  document.getElementById('avgFuzzy').textContent = avgFuzzy != null ? avgFuzzy.toFixed(3) : '—';
  document.getElementById('avgScore').textContent = avgScore != null ? avgScore.toFixed(2)+' / 5' : '—';
  document.getElementById('avgFaith').textContent = avgFaith != null ? avgFaith.toFixed(2)+' / 5' : '—';

  const labels = data.map(r => 'qid '+r.qid);
  recallChart.data.labels = labels;
  recallChart.data.datasets[0].data = data.map(r => r.recall_fuzzy ?? null);
  recallChart.update();
  scoreChart.data.labels = labels;
  scoreChart.data.datasets[0].data = data.map(r => r.correctness_score  ?? 0);
  scoreChart.data.datasets[1].data = data.map(r => r.faithfulness_score ?? 0);
  scoreChart.update();

  const escHtml  = s => (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const laClaTh  = showLaCla ? '' : '<th>LA / CLA</th>';
  const rows = data.map(r => {{
    const rf = r.recall_fuzzy;
    const sc = r.correctness_score;
    const fa = r.faithfulness_score;
    const rcCls  = rf==null?'rc-miss':rf===1?'rc-hit':rf>0?'rc-part':'rc-miss';
    const rfHtml = rf != null ? rf.toFixed(2) : '—';
    const scHtml = sc != null ? `<span class="score ${{scoreClass(sc)}}">${{sc}}</span>` : '—';
    const faHtml = fa != null ? `<span class="score ${{scoreClass(fa)}}">${{fa}}</span>` : '—';
    const gtChips  = (r.gt_reference  ||[]).map(x=>`<span class="ref-chip">${{x}}</span>`).join('');
    const ragChips = (r.rag_reference ||[]).map(x=>`<span class="ref-chip rag">${{x}}</span>`).join('');
    const laClaTd  = showLaCla ? '' : `<td><span class="lacla-badge">${{r.la}}<br>${{r.cla}}</span></td>`;
    return `<tr>
      <td style="font-weight:600;color:#0f3460;white-space:nowrap">${{r.qid}}</td>
      <td class="q-text" title="${{(r.question||'').replace(/"/g,'&quot;')}}">${{r.question||''}}</td>
      <td class="rc ${{rcCls}}">${{rfHtml}}</td>
      <td>${{scHtml}}</td><td class="reason">${{escHtml(r.correctness_reason||'')}}</td>
      <td>${{faHtml}}</td><td class="reason">${{escHtml(r.faithfulness_reason||'')}}</td>
      <td class="ans-cell">${{escHtml(r.answer)}}</td>
      <td class="ans-cell">${{escHtml(r.rag_answer)}}</td>
      <td>${{gtChips}}</td><td>${{ragChips}}</td>
      ${{laClaTd}}
    </tr>`;
  }}).join('');

  document.getElementById('tableBody').innerHTML = `
    <table><thead><tr>
      <th>qid</th><th>질문</th>
      <th>Recall<br>Fuzzy</th>
      <th>Correctness</th><th>Correctness 사유</th>
      <th>Faithfulness</th><th>Faithfulness 사유</th>
      <th>모범 답변</th><th>RAG 답변</th>
      <th>GT Reference</th><th>RAG Reference</th>
      ${{laClaTh}}
    </tr></thead><tbody>${{rows}}</tbody></table>`;
}}

function renderAdvice() {{
  const section = document.getElementById('adviceSection');
  const grid    = document.getElementById('adviceGrid');
  if (!EMBEDDED_ADVICE.length) {{ section.style.display='none'; return; }}
  const TYPE_CLASS = {{ recall:'recall', correct:'correct', faithfulness:'faithfulness', lacla:'lacla', arch:'arch', prompt:'prompt' }};
  grid.innerHTML = EMBEDDED_ADVICE.map(c => {{
    const cls     = TYPE_CLASS[c.type] || 'arch';
    const bullets = (c.bullets||[]).map(b=>`<li>${{b}}</li>`).join('');
    return `<div class="advice-card ${{cls}}"><h4>${{c.title}}</h4><ul>${{bullets}}</ul></div>`;
  }}).join('');
  section.dataset.hasContent = 'true';
  section.style.display = currentTab === 'ALL' ? 'block' : 'none';
}}

function initMirrorScroll() {{
  const top   = document.getElementById('tableScrollTop');
  const inner = document.getElementById('tableScrollTopInner');
  const bot   = document.getElementById('tableScroll');
  function sync() {{ inner.style.width = bot.scrollWidth + 'px'; }}
  sync();
  new ResizeObserver(sync).observe(bot);
  let syncing = false;
  top.addEventListener('scroll', () => {{ if(!syncing){{ syncing=true; bot.scrollLeft=top.scrollLeft; syncing=false; }} }});
  bot.addEventListener('scroll', () => {{ if(!syncing){{ syncing=true; top.scrollLeft=bot.scrollLeft; syncing=false; }} }});
}}

window.addEventListener('DOMContentLoaded', () => {{
  initCharts();
  initMirrorScroll();
  if (allData.length) {{
    buildTabs(allData);
    selectTab('ALL');
  }}
  renderAdvice();
}});
</script>
</body>
</html>"""


def main():
    print("=== RAG 평가 결과 HTML 내보내기 ===\n")

    results, advice = build_data()
    if not results:
        print("✗ 평가 결과 파일이 없습니다. 먼저 평가를 실행하세요.")
        return

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file  = Path(__file__).parent / f"rag_eval_report_{timestamp}.html"

    html = generate_html(results, advice, generated_at)
    output_file.write_text(html, encoding="utf-8")

    print(f"✓ 저장 완료: {output_file}")
    print(f"  총 {len(results)}개 평가 결과 / 개선 의견 {len(advice)}개 카드 포함")
    print(f"\n브라우저에서 파일을 직접 열거나 공유하세요.")


if __name__ == "__main__":
    main()
