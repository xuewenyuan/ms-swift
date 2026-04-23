#!/usr/bin/env python3
"""Render a static HTML dashboard from VLA cluster mining reports."""

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple


PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
ACTION_COLORS = {
    "downsample_frequent": "#4C78A8",
    "keep_tail": "#F58518",
    "review_first": "#E45756",
    "keep_hard_or_review": "#B279A2",
    "balanced_keep": "#54A24B",
}
CANDIDATE_COLORS = {
    "low_cluster_closeness": "#E45756",
    "minority_in_cluster": "#F58518",
    "rare_scene_decision": "#B279A2",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a static HTML dashboard from VLA mining reports.")
    parser.add_argument("--mining-dir", required=True, help="Directory produced by analyze_cluster_mining.py.")
    parser.add_argument("--output", required=True, help="Output HTML path.")
    parser.add_argument("--topn", type=int, default=30, help="Top categories shown in summary charts. Default: 30.")
    parser.add_argument(
        "--review-limit",
        type=int,
        default=5000,
        help="Maximum review candidate rows embedded in the dashboard. Default: 5000.")
    parser.add_argument(
        "--table-limit",
        type=int,
        default=200,
        help="Maximum rows shown in HTML tables. Full CSV/JSONL remains on disk. Default: 200.")
    parser.add_argument(
        "--plotly-js",
        default=PLOTLY_CDN,
        help="Plotly.js URL or local browser path used by the HTML. Default: Plotly CDN.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mining_dir = Path(args.mining_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    data = load_dashboard_data(mining_dir, args.topn, args.review_limit, args.table_limit)
    output.write_text(render_html(data, args.plotly_js), encoding="utf-8")
    print(f"wrote dashboard -> {output}")


def load_dashboard_data(mining_dir: Path, topn: int, review_limit: int, table_limit: int) -> Dict[str, Any]:
    summary = read_json_if_exists(mining_dir / "mining_summary.json")
    sampling = read_csv_dicts(mining_dir / "sampling_buckets.csv")
    cluster_decisions = read_csv_dicts(mining_dir / "cluster_decision_distribution.csv")
    scene_decisions = read_csv_dicts(mining_dir / "scene_decision_counts.csv")
    label_clusters = read_csv_dicts(mining_dir / "label_cluster_distribution.csv")
    review_candidates = read_jsonl_limit(mining_dir / "review_candidates.jsonl", review_limit)
    conditional_jobs = read_jsonl_limit(mining_dir / "conditional_cluster_jobs.jsonl", table_limit)

    return {
        "summary": summary,
        "cards": build_cards(summary, sampling, cluster_decisions, review_candidates),
        "overview": build_overview_data(scene_decisions, label_clusters, review_candidates, topn),
        "sampling": build_sampling_data(sampling, topn),
        "clusters": build_cluster_data(cluster_decisions, topn),
        "scene_decision": build_scene_decision_heatmap(scene_decisions, topn),
        "label_dispersion": build_label_dispersion(label_clusters, topn),
        "review": build_review_data(review_candidates, table_limit),
        "conditional_jobs": conditional_jobs[:table_limit],
        "tables": {
            "sampling_buckets": sampling[:table_limit],
            "review_candidates": review_candidates[:table_limit],
        },
    }


def build_cards(summary: Dict[str, Any], sampling: Sequence[Dict[str, str]],
                cluster_decisions: Sequence[Dict[str, str]], review_candidates: Sequence[Dict[str, Any]]) -> List[Dict[str,
                                                                                                                        Any]]:
    cluster_ids = {row.get("cluster_id") for row in cluster_decisions if row.get("cluster_id") not in (None, "")}
    labels = {row.get("label") for row in sampling if row.get("label") not in (None, "")}
    primary_scenes = {row.get("primary_scene") for row in sampling if row.get("primary_scene") not in (None, "")}
    return [
        {"name": "Rows", "value": summary.get("num_rows", sum(to_int(row.get("count")) for row in sampling))},
        {"name": "Primary Scenes", "value": summary.get("num_primary_scenes", len(primary_scenes))},
        {"name": "Decision Labels", "value": summary.get("num_labels", len(labels))},
        {"name": "Clusters", "value": summary.get("num_clusters", len(cluster_ids))},
        {"name": "Sampling Buckets", "value": summary.get("num_sampling_buckets", len(sampling))},
        {"name": "Review Candidates", "value": sum(1 for _ in review_candidates)},
    ]


def build_overview_data(scene_decisions: Sequence[Dict[str, str]], label_clusters: Sequence[Dict[str, str]],
                        review_candidates: Sequence[Dict[str, Any]], topn: int) -> Dict[str, Any]:
    primary_counts: Counter = Counter()
    for row in scene_decisions:
        if row.get("scene_level") == "primary":
            primary_counts[row.get("scene", "UNKNOWN")] += to_int(row.get("count"))

    label_counts: Counter = Counter()
    for row in label_clusters:
        label_counts[row.get("label", "UNKNOWN")] += to_int(row.get("count"))

    candidate_counts = Counter(row.get("candidate_type", "UNKNOWN") for row in review_candidates)
    return {
        "primary_scene_bar": counter_chart(primary_counts, topn),
        "label_bar": counter_chart(label_counts, topn),
        "candidate_bar": counter_chart(candidate_counts, topn),
    }


def build_sampling_data(sampling: Sequence[Dict[str, str]], topn: int) -> Dict[str, Any]:
    primary_counts: Counter = Counter()
    label_before: Counter = Counter()
    label_after: Counter = Counter()
    action_counts: Counter = Counter()
    scatter = []
    for row in sampling:
        count = to_int(row.get("count"))
        target = to_int(row.get("suggested_target_count"))
        label = row.get("label", "UNKNOWN")
        primary = row.get("primary_scene", "UNKNOWN")
        action = row.get("suggested_action", "UNKNOWN")
        primary_counts[primary] += count
        label_before[label] += count
        label_after[label] += target
        action_counts[action] += 1
        scatter.append({
            "x": count,
            "y": target,
            "action": action,
            "primary_scene": primary,
            "label": label,
            "cluster_id": row.get("cluster_id"),
            "cluster_purity": to_float(row.get("cluster_purity")),
            "text": f"{primary}<br>{label}<br>cluster={row.get('cluster_id')}<br>action={action}",
        })

    treemap = build_sampling_treemap(sampling, topn)
    labels = [key for key, _ in label_before.most_common(topn)]
    return {
        "treemap": treemap,
        "action_bar": counter_chart(action_counts, topn),
        "before_after": {
            "labels": labels,
            "before": [label_before[label] for label in labels],
            "after": [label_after[label] for label in labels],
        },
        "scatter": downsample_rows(scatter, limit=5000),
        "primary_bar": counter_chart(primary_counts, topn),
    }


def build_sampling_treemap(sampling: Sequence[Dict[str, str]], topn: int) -> Dict[str, List[Any]]:
    primary_totals: Counter = Counter()
    label_totals: Counter = Counter()
    bucket_rows = sorted(sampling, key=lambda row: to_int(row.get("count")), reverse=True)[:max(topn * 20, topn)]
    for row in bucket_rows:
        count = to_int(row.get("count"))
        primary = row.get("primary_scene", "UNKNOWN")
        label = row.get("label", "UNKNOWN")
        primary_totals[primary] += count
        label_totals[(primary, label)] += count

    ids: List[str] = []
    labels: List[str] = []
    parents: List[str] = []
    values: List[int] = []
    colors: List[str] = []
    text: List[str] = []

    for primary, count in primary_totals.items():
        node_id = f"scene::{primary}"
        ids.append(node_id)
        labels.append(primary)
        parents.append("")
        values.append(count)
        colors.append("#BAB0AC")
        text.append(f"{primary}<br>count={count}")
    for (primary, label), count in label_totals.items():
        node_id = f"label::{primary}::{label}"
        ids.append(node_id)
        labels.append(label)
        parents.append(f"scene::{primary}")
        values.append(count)
        colors.append("#9ecae9")
        text.append(f"{primary}<br>{label}<br>count={count}")
    for row in bucket_rows:
        primary = row.get("primary_scene", "UNKNOWN")
        label = row.get("label", "UNKNOWN")
        cluster_id = row.get("cluster_id", "UNKNOWN")
        action = row.get("suggested_action", "UNKNOWN")
        count = to_int(row.get("count"))
        ids.append(f"bucket::{primary}::{label}::{cluster_id}")
        labels.append(f"c{cluster_id}")
        parents.append(f"label::{primary}::{label}")
        values.append(count)
        colors.append(ACTION_COLORS.get(action, "#72B7B2"))
        text.append(f"{primary}<br>{label}<br>cluster={cluster_id}<br>count={count}<br>action={action}")
    return {"ids": ids, "labels": labels, "parents": parents, "values": values, "colors": colors, "text": text}


def build_cluster_data(cluster_decisions: Sequence[Dict[str, str]], topn: int) -> Dict[str, Any]:
    cluster_size: Counter = Counter()
    dominant_rows = []
    for row in cluster_decisions:
        cluster_id = row.get("cluster_id", "UNKNOWN")
        cluster_size[cluster_id] += to_int(row.get("count"))
        if str(row.get("is_dominant", "")).lower() in {"true", "1", "yes"}:
            dominant_rows.append(row)

    scatter = []
    for row in dominant_rows:
        cluster_id = row.get("cluster_id", "UNKNOWN")
        scatter.append({
            "x": cluster_size[cluster_id],
            "y": to_float(row.get("cluster_purity")),
            "cluster_id": cluster_id,
            "label": row.get("label", "UNKNOWN"),
            "entropy": to_float(row.get("cluster_entropy")),
            "text": f"cluster={cluster_id}<br>{row.get('label')}<br>size={cluster_size[cluster_id]}"
                    f"<br>purity={to_float(row.get('cluster_purity')):.4f}",
        })

    low_purity = sorted(scatter, key=lambda row: row["y"])[:topn]
    largest = sorted(scatter, key=lambda row: row["x"], reverse=True)[:topn]
    return {"scatter": scatter, "low_purity": low_purity, "largest": largest}


def build_scene_decision_heatmap(scene_decisions: Sequence[Dict[str, str]], topn: int) -> Dict[str, Any]:
    scene_counts: Counter = Counter()
    label_counts: Counter = Counter()
    values: DefaultDict[Tuple[str, str], int] = defaultdict(int)
    for row in scene_decisions:
        if row.get("scene_level") != "primary":
            continue
        scene = row.get("scene", "UNKNOWN")
        label = row.get("label", "UNKNOWN")
        count = to_int(row.get("count"))
        scene_counts[scene] += count
        label_counts[label] += count
        values[(scene, label)] += count
    scenes = [key for key, _ in scene_counts.most_common(topn)]
    labels = [key for key, _ in label_counts.most_common(topn)]
    z = []
    text = []
    for scene in scenes:
        row_values = []
        row_text = []
        for label in labels:
            count = values[(scene, label)]
            row_values.append(math.log1p(count))
            row_text.append(f"{scene}<br>{label}<br>count={count}")
        z.append(row_values)
        text.append(row_text)
    return {"x": labels, "y": scenes, "z": z, "text": text}


def build_label_dispersion(label_clusters: Sequence[Dict[str, str]], topn: int) -> Dict[str, Any]:
    entropy_by_label: Dict[str, float] = {}
    cluster_count_by_label: Counter = Counter()
    count_by_label: Counter = Counter()
    for row in label_clusters:
        label = row.get("label", "UNKNOWN")
        entropy_by_label[label] = max(entropy_by_label.get(label, 0.0), to_float(row.get("label_cluster_entropy")))
        cluster_count_by_label[label] += 1
        count_by_label[label] += to_int(row.get("count"))
    rows = sorted(entropy_by_label, key=lambda label: entropy_by_label[label], reverse=True)[:topn]
    return {
        "labels": rows,
        "entropy": [entropy_by_label[label] for label in rows],
        "clusters": [cluster_count_by_label[label] for label in rows],
        "counts": [count_by_label[label] for label in rows],
    }


def build_review_data(review_candidates: Sequence[Dict[str, Any]], table_limit: int) -> Dict[str, Any]:
    rows = downsample_rows(list(review_candidates), limit=5000)
    return {
        "scatter": [{
            "x": to_float(row.get("closeness")),
            "y": to_float(row.get("cluster_label_ratio")),
            "candidate_type": row.get("candidate_type", "UNKNOWN"),
            "text": candidate_hover(row),
        } for row in rows],
        "table": list(review_candidates[:table_limit]),
    }


def candidate_hover(row: Dict[str, Any]) -> str:
    parts = [
        f"type={row.get('candidate_type')}",
        f"sample_id={row.get('sample_id')}",
        f"sample_token={row.get('sample_token')}",
        f"timestamp={row.get('timestamp')}",
        f"scene={row.get('primary_scene')} / {row.get('secondary_scene')}",
        f"label={row.get('label')}",
        f"dominant={row.get('dominant_label')}",
        f"cluster={row.get('cluster_id')}",
        f"reason={row.get('reason')}",
    ]
    return "<br>".join(html.escape(str(part)) for part in parts)


def counter_chart(counter: Counter, topn: int) -> Dict[str, List[Any]]:
    items = counter.most_common(topn)
    return {"x": [key for key, _ in items], "y": [value for _, value in items]}


def render_html(data: Dict[str, Any], plotly_js: str) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VLA Cluster Mining Dashboard</title>
  <script src="{html.escape(plotly_js)}"></script>
  <style>
    :root {{
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #5f6b7a;
      --line: #d9dee7;
      --accent: #31688e;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    header {{
      padding: 24px 32px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 26px;
      font-weight: 700;
    }}
    h2 {{
      margin: 0 0 16px;
      font-size: 19px;
    }}
    .subtle {{
      color: var(--muted);
      font-size: 13px;
    }}
    main {{
      padding: 20px 32px 40px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .card .name {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .card .value {{
      font-size: 24px;
      font-weight: 700;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin: 16px 0;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 16px;
    }}
    .plot {{
      width: 100%;
      min-height: 390px;
    }}
    .plot.tall {{
      min-height: 560px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 12px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 7px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #eef2f7;
      z-index: 1;
    }}
    .table-wrap {{
      overflow: auto;
      max-height: 520px;
      border: 1px solid var(--line);
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>VLA Cluster Mining Dashboard</h1>
    <div class="subtle">Sampling balance, cluster quality, scene-decision coverage, and review candidates.</div>
  </header>
  <main>
    <div id="cards" class="cards"></div>

    <section>
      <h2>Overview</h2>
      <div class="grid">
        <div id="primarySceneBar" class="plot"></div>
        <div id="labelBar" class="plot"></div>
        <div id="candidateBar" class="plot"></div>
      </div>
    </section>

    <section>
      <h2>Sampling Balance</h2>
      <div id="samplingTreemap" class="plot tall"></div>
      <div class="grid">
        <div id="beforeAfterBar" class="plot"></div>
        <div id="samplingScatter" class="plot"></div>
        <div id="actionBar" class="plot"></div>
      </div>
    </section>

    <section>
      <h2>Cluster Quality</h2>
      <div class="grid">
        <div id="clusterScatter" class="plot"></div>
        <div id="labelDispersion" class="plot"></div>
      </div>
    </section>

    <section>
      <h2>Scene x Decision</h2>
      <div id="sceneDecisionHeatmap" class="plot tall"></div>
    </section>

    <section>
      <h2>Review Candidates</h2>
      <div id="reviewScatter" class="plot"></div>
      <div id="reviewTable" class="table-wrap"></div>
    </section>

    <section>
      <h2>Conditional Clustering Jobs</h2>
      <div id="jobsTable" class="table-wrap"></div>
    </section>
  </main>

  <script>
    const DATA = {payload};
    const layoutBase = {{
      paper_bgcolor: "white",
      plot_bgcolor: "white",
      margin: {{l: 56, r: 24, t: 42, b: 90}},
      font: {{family: "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", size: 12}},
    }};

    function fmt(v) {{
      if (v === null || v === undefined || v === "") return "";
      if (typeof v === "number") return v.toLocaleString();
      return String(v);
    }}

    function renderCards() {{
      const root = document.getElementById("cards");
      root.innerHTML = DATA.cards.map(card => `
        <div class="card"><div class="name">${{escapeHtml(card.name)}}</div><div class="value">${{fmt(card.value)}}</div></div>
      `).join("");
    }}

    function bar(div, title, chart, opts={{}}) {{
      Plotly.newPlot(div, [{{type: "bar", x: chart.x, y: chart.y, marker: {{color: opts.color || "#4C78A8"}}}}],
        {{...layoutBase, title, xaxis: {{automargin: true}}, yaxis: {{automargin: true, title: opts.ytitle || "count"}}}},
        {{responsive: true}});
    }}

    function renderOverview() {{
      bar("primarySceneBar", "Top Primary Scenes", DATA.overview.primary_scene_bar, {{color: "#4C78A8"}});
      bar("labelBar", "Top Decision Labels", DATA.overview.label_bar, {{color: "#54A24B"}});
      bar("candidateBar", "Review Candidate Types", DATA.overview.candidate_bar, {{color: "#E45756"}});
    }}

    function renderSampling() {{
      const t = DATA.sampling.treemap;
      Plotly.newPlot("samplingTreemap", [{{
        type: "treemap",
        ids: t.ids,
        labels: t.labels,
        parents: t.parents,
        values: t.values,
        marker: {{colors: t.colors}},
        text: t.text,
        hovertemplate: "%{{text}}<extra></extra>",
        branchvalues: "total",
      }}], {{...layoutBase, title: "Primary Scene -> Decision Label -> Cluster"}}, {{responsive: true}});

      const ba = DATA.sampling.before_after;
      Plotly.newPlot("beforeAfterBar", [
        {{type: "bar", name: "raw count", x: ba.labels, y: ba.before, marker: {{color: "#4C78A8"}}}},
        {{type: "bar", name: "suggested target", x: ba.labels, y: ba.after, marker: {{color: "#F58518"}}}},
      ], {{...layoutBase, barmode: "group", title: "Raw vs Suggested Target by Label", xaxis: {{automargin: true}}}},
      {{responsive: true}});

      const sc = DATA.sampling.scatter;
      Plotly.newPlot("samplingScatter", [{{
        type: "scattergl",
        mode: "markers",
        x: sc.map(d => d.x),
        y: sc.map(d => d.y),
        text: sc.map(d => d.text),
        hovertemplate: "%{{text}}<br>count=%{{x}}<br>target=%{{y}}<extra></extra>",
        marker: {{size: 7, opacity: 0.75, color: sc.map(d => actionColor(d.action))}},
      }}], {{...layoutBase, title: "Bucket Downsampling Shape", xaxis: {{title: "raw count", type: "log"}},
             yaxis: {{title: "suggested target count", type: "log"}}}}, {{responsive: true}});

      bar("actionBar", "Sampling Actions", DATA.sampling.action_bar, {{color: "#B279A2"}});
    }}

    function renderClusters() {{
      const sc = DATA.clusters.scatter;
      Plotly.newPlot("clusterScatter", [{{
        type: "scattergl",
        mode: "markers",
        x: sc.map(d => d.x),
        y: sc.map(d => d.y),
        text: sc.map(d => d.text),
        hovertemplate: "%{{text}}<extra></extra>",
        marker: {{size: 8, opacity: 0.78, color: sc.map(d => d.y), colorscale: "Viridis", colorbar: {{title: "purity"}}}},
      }}], {{...layoutBase, title: "Cluster Size vs Purity", xaxis: {{title: "cluster size", type: "log"}},
             yaxis: {{title: "purity", range: [0, 1.02]}}}}, {{responsive: true}});

      const ld = DATA.label_dispersion;
      Plotly.newPlot("labelDispersion", [{{
        type: "bar",
        x: ld.entropy,
        y: ld.labels,
        orientation: "h",
        marker: {{color: "#E45756"}},
        customdata: ld.clusters.map((v, i) => [v, ld.counts[i]]),
        hovertemplate: "%{{y}}<br>entropy=%{{x:.4f}}<br>clusters=%{{customdata[0]}}<br>count=%{{customdata[1]}}<extra></extra>",
      }}], {{...layoutBase, title: "Most Dispersed Labels Across Clusters",
             xaxis: {{title: "label-cluster entropy"}}, yaxis: {{automargin: true}}}}, {{responsive: true}});
    }}

    function renderSceneDecision() {{
      const h = DATA.scene_decision;
      Plotly.newPlot("sceneDecisionHeatmap", [{{
        type: "heatmap",
        x: h.x,
        y: h.y,
        z: h.z,
        text: h.text,
        hovertemplate: "%{{text}}<br>log1p(count)=%{{z:.3f}}<extra></extra>",
        colorscale: "YlGnBu",
      }}], {{...layoutBase, title: "Primary Scene x Decision Label (log count)",
             xaxis: {{automargin: true}}, yaxis: {{automargin: true}}}}, {{responsive: true}});
    }}

    function renderReview() {{
      const rows = DATA.review.scatter;
      const groups = groupBy(rows, d => d.candidate_type || "UNKNOWN");
      const traces = Object.entries(groups).map(([name, items]) => ({{
        type: "scattergl",
        mode: "markers",
        name,
        x: items.map(d => d.x),
        y: items.map(d => d.y),
        text: items.map(d => d.text),
        hovertemplate: "%{{text}}<br>closeness=%{{x:.4f}}<br>label_ratio=%{{y:.4f}}<extra></extra>",
        marker: {{size: 7, opacity: 0.74, color: candidateColor(name)}},
      }}));
      Plotly.newPlot("reviewScatter", traces,
        {{...layoutBase, title: "Review Candidates: Closeness vs Cluster Label Ratio",
          xaxis: {{title: "centroid closeness"}}, yaxis: {{title: "label ratio in cluster", range: [0, 1.02]}}}},
        {{responsive: true}});
      renderTable("reviewTable", DATA.review.table, [
        "candidate_type", "sample_id", "sample_token", "timestamp", "primary_scene", "secondary_scene",
        "cluster_id", "label", "dominant_label", "closeness", "cluster_label_ratio", "reason"
      ]);
      renderTable("jobsTable", DATA.conditional_jobs, [
        "primary_scene", "label", "count", "suggested_num_clusters", "reason"
      ]);
    }}

    function renderTable(div, rows, columns) {{
      const root = document.getElementById(div);
      if (!rows || rows.length === 0) {{
        root.innerHTML = "<div class='subtle' style='padding:12px'>No rows.</div>";
        return;
      }}
      root.innerHTML = `<table><thead><tr>${{columns.map(c => `<th>${{escapeHtml(c)}}</th>`).join("")}}</tr></thead>
        <tbody>${{rows.map(row => `<tr>${{columns.map(c => `<td>${{escapeHtml(fmt(row[c]))}}</td>`).join("")}}</tr>`).join("")}}</tbody></table>`;
    }}

    function actionColor(action) {{
      const colors = {json.dumps(ACTION_COLORS)};
      return colors[action] || "#72B7B2";
    }}
    function candidateColor(type) {{
      const colors = {json.dumps(CANDIDATE_COLORS)};
      return colors[type] || "#4C78A8";
    }}
    function groupBy(rows, fn) {{
      const out = {{}};
      for (const row of rows) {{
        const key = fn(row);
        if (!out[key]) out[key] = [];
        out[key].push(row);
      }}
      return out;
    }}
    function escapeHtml(value) {{
      return String(value ?? "").replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[c]));
    }}

    renderCards();
    renderOverview();
    renderSampling();
    renderClusters();
    renderSceneDecision();
    renderReview();
  </script>
</body>
</html>
"""


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl_limit(path: Path, limit: int) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def downsample_rows(rows: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if len(rows) <= limit:
        return list(rows)
    step = max(1, len(rows) // limit)
    return list(rows[::step][:limit])


def to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
