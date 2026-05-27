#!/usr/bin/env python3
"""Generate an evaluation.html report from RAGAS evaluation results.

Usage:
    python scripts/generate_report.py --input data/evaluation/report.json --output docs/evaluation.html
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _score_color(score: float) -> str:
    """Map score (0–1) to a traffic-light color."""
    if score >= 0.8:
        return "#2e7d32"  # green
    elif score >= 0.6:
        return "#f57f17"  # amber
    else:
        return "#c62828"  # red


def _score_bar(score: float, width: int = 120) -> str:
    """Render a horizontal CSS bar for a score."""
    pct = max(0, min(1, score)) * 100
    color = _score_color(score)
    return (
        f'<div style="display:inline-block;width:{width}px;height:12px;'
        f'background:#e0e0e0;border-radius:6px;vertical-align:middle;margin-right:8px;">'
        f'<div style="width:{pct:.0f}%;height:100%;background:{color};'
        f'border-radius:6px;"></div></div>'
    )


METRIC_LABELS_ZH = {
    "context_precision": "上下文精确度",
    "context_recall": "上下文召回率",
    "faithfulness": "忠实度",
    "answer_relevancy": "回答相关性",
    "answer_correctness": "回答正确性",
}

INTENT_LABELS_ZH = {
    "howto": "做法查询",
    "recommendation": "推荐",
    "ingredient": "原料查询",
    "factual": "事实查询",
}


def render_report(data: dict) -> str:
    """Render full HTML report from evaluation data dict."""
    agg = data.get("aggregate", {})
    per_intent = data.get("per_intent", {})
    meta = data.get("metadata", {})
    samples = data.get("samples", [])

    metric_names = sorted(agg.keys()) if agg else [
        "context_precision", "context_recall", "faithfulness",
        "answer_relevancy", "answer_correctness",
    ]

    # ---- Overall score cards ----
    cards_html = ""
    for name in metric_names:
        score = agg.get(name, 0)
        label = METRIC_LABELS_ZH.get(name, name)
        cards_html += f"""
        <div class="score-card">
          <div class="score-value" style="color:{_score_color(score)}">{score:.3f}</div>
          <div class="score-label">{label}</div>
          <div class="score-bar">{_score_bar(score, 100)}</div>
        </div>"""

    # ---- Per-intent table ----
    intent_rows = ""
    for intent in ["recommendation", "howto", "ingredient", "factual"]:
        scores = per_intent.get(intent, {})
        if not scores:
            continue
        label = INTENT_LABELS_ZH.get(intent, intent)
        cells = f'<td class="intent-name">{label}</td>'
        for m in metric_names:
            val = scores.get(m)
            if val is not None:
                cells += f'<td><span style="color:{_score_color(val)};font-weight:600;">{val:.3f}</span></td>'
            else:
                cells += '<td class="na">—</td>'
        intent_rows += f"<tr>{cells}</tr>"

    per_intent_html = ""
    if intent_rows:
        header_cells = "<th>意图</th>" + "".join(
            f"<th>{METRIC_LABELS_ZH.get(m, m)}</th>" for m in metric_names
        )
        per_intent_html = f"""
        <div class="section">
          <h2>按意图分组</h2>
          <table class="intent-table">
            <thead><tr>{header_cells}</tr></thead>
            <tbody>{intent_rows}</tbody>
          </table>
        </div>"""

    # ---- Per-sample table ----
    sample_rows = ""
    for i, s in enumerate(samples):
        q = s["query"]
        intent = INTENT_LABELS_ZH.get(s.get("intent", ""), s.get("intent", ""))
        scores = s.get("scores", {})
        cells = (
            f'<td class="idx">{i + 1}</td>'
            f'<td class="qcol">{q}</td>'
            f'<td class="intent">{intent}</td>'
        )
        for m in metric_names:
            val = scores.get(m)
            if val is not None:
                cells += f'<td><span style="color:{_score_color(val)};font-weight:600;">{val:.3f}</span></td>'
            else:
                cells += '<td class="na">—</td>'
        sample_rows += f"<tr>{cells}</tr>"

    header_cells = "<th>#</th><th>查询</th><th>意图</th>" + "".join(
        f"<th>{METRIC_LABELS_ZH.get(m, m)}</th>" for m in metric_names
    )

    # ---- Answer details (collapsible) ----
    answer_rows = ""
    for i, s in enumerate(samples):
        answer = s.get("answer", "")
        if not answer:
            continue
        # Truncate for display
        display = answer[:200] + "…" if len(answer) > 200 else answer
        answer_rows += f"""
        <details class="answer-detail">
          <summary><strong>#{i + 1}</strong> {s['query']}</summary>
          <div class="answer-text">{answer}</div>
        </details>"""

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Evaluation — RAGAS 评估报告</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    background: #f8f9fa;
    font-family: 'Noto Sans SC', 'Segoe UI', system-ui, sans-serif;
    margin: 0; padding: 20px;
    color: #333;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ color: #1a1a2e; text-align: center; font-size: 24px; margin: 0 0 4px 0; }}
  .subtitle {{ text-align: center; color: #666; font-size: 13px; margin: 0 0 24px 0; }}

  /* Summary cards */
  .cards {{
    display: flex; flex-wrap: wrap; gap: 16px; justify-content: center;
    margin-bottom: 24px;
  }}
  .score-card {{
    background: #fff; border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,.07);
    padding: 16px 20px; min-width: 160px; text-align: center;
  }}
  .score-value {{ font-size: 32px; font-weight: 700; margin-bottom: 2px; }}
  .score-label {{ font-size: 12px; color: #888; margin-bottom: 8px; }}
  .score-bar {{ display: flex; justify-content: center; }}

  /* Sections */
  .section {{
    background: #fff; border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,.07);
    padding: 20px 24px; margin-bottom: 18px;
  }}
  .section h2 {{
    font-size: 16px; color: #1a1a2e; margin: 0 0 12px 0;
    padding-bottom: 8px; border-bottom: 2px solid #e8f4fd;
  }}

  /* Tables */
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  thead th {{
    background: #e8f4fd; color: #1565c0; font-weight: 600;
    padding: 8px 10px; text-align: left; white-space: nowrap;
    position: sticky; top: 0;
  }}
  tbody td {{
    padding: 7px 10px; border-bottom: 1px solid #f0f0f0;
  }}
  tr:hover td {{ background: #fafafa; }}
  .intent-table {{ max-width: 800px; }}
  .intent-name {{ font-weight: 600; }}
  .qcol {{ max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .idx {{ color: #999; font-size: 11px; text-align: center; width: 30px; }}
  .intent {{ color: #666; font-size: 12px; }}
  .na {{ color: #bbb; }}

  /* Answer details */
  .answer-detail {{
    margin: 4px 0; padding: 6px 0;
    border-bottom: 1px solid #f0f0f0;
  }}
  .answer-detail summary {{
    cursor: pointer; font-size: 13px; color: #333;
    padding: 4px 0;
  }}
  .answer-detail summary:hover {{ color: #1565c0; }}
  .answer-text {{
    background: #fafafa; border-radius: 6px; padding: 12px 16px;
    margin: 8px 0; font-size: 13px; line-height: 1.7;
    white-space: pre-wrap; word-break: break-word;
  }}

  /* Meta footer */
  .meta {{
    background: #e8f4fd; border-radius: 8px; padding: 12px 16px;
    font-size: 13px; color: #333; line-height: 1.7;
    margin-top: 8px;
  }}
  .meta code {{ background: #d0e8f7; padding: 1px 5px; border-radius: 3px; font-size: 12px; }}
  .meta strong {{ color: #1a73e8; }}

  /* Mermaid box */
  .mermaid-box {{
    background: #fff; border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,.07);
    padding: 20px; margin-bottom: 18px; overflow-x: auto;
  }}

  @media (max-width: 768px) {{
    .cards {{ flex-direction: column; align-items: center; }}
    .score-card {{ min-width: 140px; }}
  }}
</style>
</head>
<body>
<div class="container">

<h1>RAGAS 评估报告</h1>
<p class="subtitle">
  {meta.get('llm_model', 'N/A')} ·
  {meta.get('num_samples', len(samples))} 样本 ·
  指标: {', '.join(METRIC_LABELS_ZH.get(m, m) for m in metric_names)}
</p>

<div class="section">
  <h2>整体指标</h2>
  <div class="cards">
    {cards_html}
  </div>
</div>

{per_intent_html}

<div class="section">
  <h2>逐样本评分</h2>
  <div style="overflow-x:auto;">
  <table>
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{sample_rows}</tbody>
  </table>
  </div>
</div>

<div class="section">
  <h2>回答详情</h2>
  {answer_rows if answer_rows else '<p style="color:#999;">无回答数据</p>'}
</div>

<div class="meta">
  <strong>评估配置</strong><br>
  模型: <code>{meta.get('llm_model', 'N/A')}</code> ·
  最大 token: <code>{meta.get('max_tokens', '4096')}</code> ·
  样本数: <code>{meta.get('num_samples', len(samples))}</code><br>
  指标: {', '.join(f'<code>{m}</code>' for m in metric_names)} ·
  生成时间: <code>{meta.get('generated_at', 'N/A')}</code>
</div>

</div>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation.html from RAGAS report JSON."
    )
    parser.add_argument(
        "--input", "-i", type=str, default="data/evaluation/report.json",
        help="Path to report.json",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="docs/evaluation.html",
        help="Output HTML path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    html = render_report(data)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
