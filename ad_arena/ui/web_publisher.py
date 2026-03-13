"""Static HTML leaderboard generator for GitHub Pages.

Reads ``results/leaderboard.json`` produced by :class:`ResultsStore` and
writes a self-contained ``index.html`` to ``docs/leaderboard/`` with:
- Aggregated metrics across all scenarios
- Per-scenario tabs with individual results
- Campaign Efficiency and Strategy Optimization suite toggles
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class WebPublisher:
    """Generates a static HTML leaderboard page from stored results."""

    def __init__(
        self,
        results_dir: Path = Path("results"),
        output_dir: Path = Path("docs/leaderboard"),
    ) -> None:
        self.results_dir = results_dir
        self.output_dir = output_dir

    def generate(self) -> None:
        """Read ``leaderboard.json`` and produce ``index.html``."""
        leaderboard_path = self.results_dir / "leaderboard.json"
        if not leaderboard_path.exists():
            logger.error("leaderboard.json not found at %s", leaderboard_path)
            return

        raw = leaderboard_path.read_text(encoding="utf-8")
        entries: list[dict] = json.loads(raw)

        scenarios = sorted({e["scenario_name"] for e in entries})

        # Build per-scenario rows for both suites
        scenario_ce_tables = {}
        scenario_so_tables = {}
        for sc in scenarios:
            sc_entries = [e for e in entries if e["scenario_name"] == sc]
            scenario_ce_tables[sc] = self._render_table_body(
                self._build_rows(sc_entries, score_key="campaign_efficiency_score")
            )
            scenario_so_tables[sc] = self._render_table_body(
                self._build_rows(sc_entries, score_key="strategy_optimization_score")
            )

        # Build aggregated rows (average across scenarios per model)
        agg_ce_rows = self._build_aggregated_rows(entries, score_key="campaign_efficiency_score")
        agg_so_rows = self._build_aggregated_rows(entries, score_key="strategy_optimization_score")
        agg_ce_body = self._render_table_body(agg_ce_rows)
        agg_so_body = self._render_table_body(agg_so_rows)

        # Build scenario tabs HTML
        scenario_tabs_html = self._render_scenario_tabs(scenarios)
        scenario_panels_html = self._render_scenario_panels(
            scenarios, scenario_ce_tables, scenario_so_tables
        )

        html = _HTML_TEMPLATE.format(
            agg_ce_body=agg_ce_body,
            agg_so_body=agg_so_body,
            scenario_tabs=scenario_tabs_html,
            scenario_panels=scenario_panels_html,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / "index.html"
        out_path.write_text(html, encoding="utf-8")
        logger.info("Leaderboard written to %s", out_path)

    @staticmethod
    def _build_rows(
        entries: list[dict],
        score_key: str = "campaign_efficiency_score",
    ) -> list[dict]:
        """Derive display-ready rows from entries, ranked by score_key."""
        rows: list[dict] = []
        for e in entries:
            total_spend = e.get("total_spend", 0)
            total_revenue = e.get("total_revenue", 0)
            total_impressions = e.get("total_impressions", 0)
            total_clicks = e.get("total_clicks", 0)
            total_conversions = e.get("total_conversions", 0)
            learning_days = e.get("learning_days", 0)
            optimizing_days = e.get("optimizing_days", 0)
            total_classifiable = learning_days + optimizing_days

            roas = (total_revenue / total_spend) if total_spend else 0.0
            cpa = (total_spend / total_conversions) if total_conversions else 0.0
            ctr = (total_clicks / total_impressions) if total_impressions else 0.0
            cvr = (total_conversions / total_clicks) if total_clicks else 0.0
            ld_ratio = learning_days / total_classifiable if total_classifiable else 0.0
            profit = e.get(score_key, e.get("total_profit", 0.0))

            rows.append({
                "model_name": e.get("model_name", ""),
                "model_type": e.get("model_type", "baseline"),
                "profit": profit,
                "roas": roas,
                "cpa": cpa,
                "ctr": ctr,
                "cvr": cvr,
                "conversions": total_conversions,
                "learning_rate": e.get("learning_rate", 0.0),
                "learning_days_ratio": ld_ratio,
                "api_calls": e.get("llm_api_calls", 0),
            })

        rows.sort(key=lambda r: r["profit"], reverse=True)
        for idx, row in enumerate(rows, start=1):
            row["rank"] = idx
        return rows

    @staticmethod
    def _build_aggregated_rows(
        entries: list[dict],
        score_key: str = "campaign_efficiency_score",
    ) -> list[dict]:
        """Aggregate metrics across scenarios per model, then rank."""
        by_model: dict[str, list[dict]] = defaultdict(list)
        for e in entries:
            by_model[e.get("model_name", "")].append(e)

        rows: list[dict] = []
        for model_name, model_entries in by_model.items():
            n = len(model_entries)
            total_spend = sum(e.get("total_spend", 0) for e in model_entries)
            total_revenue = sum(e.get("total_revenue", 0) for e in model_entries)
            total_impressions = sum(e.get("total_impressions", 0) for e in model_entries)
            total_clicks = sum(e.get("total_clicks", 0) for e in model_entries)
            total_conversions = sum(e.get("total_conversions", 0) for e in model_entries)
            total_learning = sum(e.get("learning_days", 0) for e in model_entries)
            total_optimizing = sum(e.get("optimizing_days", 0) for e in model_entries)
            total_classifiable = total_learning + total_optimizing

            avg_score = sum(e.get(score_key, 0) for e in model_entries) / n
            roas = (total_revenue / total_spend) if total_spend else 0.0
            cpa = (total_spend / total_conversions) if total_conversions else 0.0
            ctr = (total_clicks / total_impressions) if total_impressions else 0.0
            cvr = (total_conversions / total_clicks) if total_clicks else 0.0
            avg_lr = sum(e.get("learning_rate", 0) for e in model_entries) / n
            ld_ratio = total_learning / total_classifiable if total_classifiable else 0.0
            total_api = sum(e.get("llm_api_calls", 0) for e in model_entries)

            rows.append({
                "model_name": model_name,
                "model_type": model_entries[0].get("model_type", "baseline"),
                "profit": avg_score,
                "roas": roas,
                "cpa": cpa,
                "ctr": ctr,
                "cvr": cvr,
                "conversions": total_conversions,
                "learning_rate": avg_lr,
                "learning_days_ratio": ld_ratio,
                "api_calls": total_api,
            })

        rows.sort(key=lambda r: r["profit"], reverse=True)
        for idx, row in enumerate(rows, start=1):
            row["rank"] = idx
        return rows

    @staticmethod
    def _render_table_body(rows: list[dict]) -> str:
        """Return ``<tr>`` elements for every row."""
        lines: list[str] = []
        for r in rows:
            type_label = "LLM" if r["model_type"] == "llm" else "Baseline"
            type_cls = "type-llm" if r["model_type"] == "llm" else "type-baseline"
            lines.append(
                f"<tr>"
                f"<td>{r['rank']}</td>"
                f"<td>{_esc(r['model_name'])}</td>"
                f'<td><span class="badge {type_cls}">{type_label}</span></td>'
                f"<td>{r['profit']:,.2f}</td>"
                f"<td>{r['roas']:.2f}</td>"
                f"<td>{r['cpa']:,.2f}</td>"
                f"<td>{r['ctr']:.4f}</td>"
                f"<td>{r['cvr']:.4f}</td>"
                f"<td>{r['conversions']}</td>"
                f"<td>{r['learning_rate']:.2f}</td>"
                f"<td>{r['learning_days_ratio']:.2f}</td>"
                f"<td>{r['api_calls']}</td>"
                f"</tr>"
            )
        return "\n".join(lines)

    @staticmethod
    def _render_scenario_tabs(scenarios: list[str]) -> str:
        """Render the scenario tab buttons."""
        tabs = []
        for sc in scenarios:
            safe_id = sc.replace(" ", "-").replace("/", "-")
            tabs.append(
                f'<button class="scenario-tab-btn" onclick="switchScenario(\'{safe_id}\')">'
                f"{_esc(sc)}</button>"
            )
        return "\n".join(tabs)

    @staticmethod
    def _render_scenario_panels(
        scenarios: list[str],
        ce_tables: dict[str, str],
        so_tables: dict[str, str],
    ) -> str:
        """Render per-scenario table panels."""
        panels = []
        for sc in scenarios:
            safe_id = sc.replace(" ", "-").replace("/", "-")
            panels.append(f'''
<div id="scenario-{safe_id}" class="scenario-panel">
  <h3>{_esc(sc)}</h3>
  <div class="suite-tables">
    <div class="suite-ce-panel suite-panel active">
      <div class="table-wrap"><table>{_THEAD}
        <tbody>{ce_tables[sc]}</tbody>
      </table></div>
    </div>
    <div class="suite-so-panel suite-panel">
      <div class="table-wrap"><table>{_THEAD}
        <tbody>{so_tables[sc]}</tbody>
      </table></div>
    </div>
  </div>
</div>''')
        return "\n".join(panels)


def _esc(text: str) -> str:
    """Minimal HTML-entity escaping."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


_THEAD = """\
<thead>
<tr>
  <th>Rank</th><th>Model</th><th>Type</th><th>Profit</th>
  <th>ROAS</th><th>CPA</th><th>CTR</th><th>CVR</th>
  <th>Conv</th><th>Learning Rate</th><th>Learning Days %</th><th>API Calls</th>
</tr>
</thead>"""


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ad Arena — LLM Leaderboard</title>
<style>
  :root {{
    --bg: #f8f9fa; --card: #fff; --border: #dee2e6;
    --text: #212529; --muted: #6c757d; --accent: #0d6efd;
    --llm: #6f42c1; --baseline: #198754;
  }}
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
         background:var(--bg); color:var(--text); line-height:1.6; }}
  .container {{ max-width:1200px; margin:0 auto; padding:2rem 1rem; }}
  h1 {{ font-size:1.75rem; margin-bottom:.25rem; }}
  h3 {{ font-size:1.1rem; margin-bottom:.75rem; color:var(--text); }}
  .subtitle {{ color:var(--muted); margin-bottom:1.5rem; }}

  /* Suite toggle (CE / SO) */
  .suite-toggle {{ display:flex; gap:0; margin-bottom:1rem; }}
  .suite-btn {{
    padding:.5rem 1.25rem; border:1px solid var(--border); background:var(--card);
    cursor:pointer; font-size:.9rem; font-weight:600; color:var(--muted);
    border-bottom:2px solid transparent;
  }}
  .suite-btn:first-child {{ border-radius:6px 0 0 0; }}
  .suite-btn:last-child {{ border-radius:0 6px 0 0; }}
  .suite-btn.active {{ color:var(--accent); border-bottom-color:var(--accent); background:#fff; }}

  /* Scenario tabs */
  .scenario-tabs {{ display:flex; gap:.25rem; margin-bottom:1.25rem; flex-wrap:wrap; }}
  .scenario-tab-btn {{
    padding:.4rem 1rem; border:1px solid var(--border); background:var(--card);
    cursor:pointer; font-size:.85rem; font-weight:500; color:var(--muted);
    border-radius:6px; transition: all .15s;
  }}
  .scenario-tab-btn:hover {{ background:#e9ecef; }}
  .scenario-tab-btn.active {{ color:#fff; background:var(--accent); border-color:var(--accent); }}

  .scenario-panel {{ display:none; }}
  .scenario-panel.active {{ display:block; }}
  .suite-panel {{ display:none; }}
  .suite-panel.active {{ display:block; }}

  .table-wrap {{ overflow-x:auto; background:var(--card); border:1px solid var(--border); border-radius:8px; }}
  table {{ width:100%; border-collapse:collapse; font-size:.875rem; }}
  thead th {{ position:sticky; top:0; background:var(--card); padding:.6rem .75rem;
              text-align:left; border-bottom:2px solid var(--border); white-space:nowrap; }}
  tbody td {{ padding:.5rem .75rem; border-bottom:1px solid var(--border); white-space:nowrap; }}
  tbody tr:hover {{ background:#f1f3f5; }}
  .badge {{ display:inline-block; padding:.15rem .5rem; border-radius:12px;
            font-size:.75rem; font-weight:600; text-transform:uppercase; letter-spacing:.03em; }}
  .type-llm {{ background:#ede7f6; color:var(--llm); }}
  .type-baseline {{ background:#e8f5e9; color:var(--baseline); }}

  .section {{ margin-top:2.5rem; }}
  .section h2 {{ font-size:1.25rem; margin-bottom:.75rem; }}
  .section p, .section li {{ color:var(--muted); font-size:.9rem; }}
  .section ul {{ padding-left:1.25rem; }}
  footer {{ margin-top:3rem; text-align:center; color:var(--muted); font-size:.8rem; }}
</style>
</head>
<body>
<div class="container">
  <h1>Ad Arena &mdash; LLM Leaderboard</h1>
  <p class="subtitle">Comparing LLM and baseline bidding strategies across standardized ad-auction scenarios.</p>

  <!-- Suite toggle -->
  <div class="suite-toggle">
    <button class="suite-btn active" onclick="switchSuite('ce')">Campaign Efficiency</button>
    <button class="suite-btn" onclick="switchSuite('so')">Strategy Optimization</button>
  </div>

  <!-- Scenario tabs -->
  <div class="scenario-tabs">
    <button class="scenario-tab-btn active" onclick="switchScenario('aggregated')">All Scenarios (Aggregated)</button>
    {scenario_tabs}
  </div>

  <!-- Aggregated panel -->
  <div id="scenario-aggregated" class="scenario-panel active">
    <h3>Aggregated across all scenarios</h3>
    <div class="suite-tables">
      <div class="suite-ce-panel suite-panel active">
        <div class="table-wrap"><table>{thead}
          <tbody>{agg_ce_body}</tbody>
        </table></div>
      </div>
      <div class="suite-so-panel suite-panel">
        <div class="table-wrap"><table>{thead}
          <tbody>{agg_so_body}</tbody>
        </table></div>
      </div>
    </div>
  </div>

  <!-- Per-scenario panels -->
  {scenario_panels}

  <div class="section" id="how-it-works">
    <h2>How It Works</h2>
    <p>
      The Ad Arena benchmark evaluates how well LLMs can learn to run
      search-ad campaigns from data alone. Each model acts as a bidder in a
      simulated ad auction, setting daily keyword bids, budgets, and ad copy
      over a 30-day episode.
    </p>
    <h3>Evaluation Suites</h3>
    <ul>
      <li><strong>Campaign Efficiency</strong> &mdash; Total profit (revenue
        minus spend) across all 30 days.</li>
      <li><strong>Strategy Optimization</strong> &mdash; Profit from only the
        final 7 days (days 24&ndash;30).</li>
    </ul>
  </div>

  <footer>
    Generated by Ad Arena &middot; <a href="https://github.com/ad-arena">GitHub</a>
  </footer>
</div>

<script>
var currentSuite = 'ce';

function switchSuite(suite) {{
  currentSuite = suite;
  document.querySelectorAll('.suite-btn').forEach(function(b) {{ b.classList.remove('active'); }});
  document.querySelectorAll('.suite-btn').forEach(function(b) {{
    if ((suite === 'ce' && b.textContent.includes('Campaign')) ||
        (suite === 'so' && b.textContent.includes('Strategy'))) {{
      b.classList.add('active');
    }}
  }});
  // Toggle suite panels in all scenario panels
  document.querySelectorAll('.suite-ce-panel').forEach(function(p) {{
    p.classList.toggle('active', suite === 'ce');
  }});
  document.querySelectorAll('.suite-so-panel').forEach(function(p) {{
    p.classList.toggle('active', suite === 'so');
  }});
}}

function switchScenario(id) {{
  document.querySelectorAll('.scenario-panel').forEach(function(p) {{ p.classList.remove('active'); }});
  document.querySelectorAll('.scenario-tab-btn').forEach(function(b) {{ b.classList.remove('active'); }});
  document.getElementById('scenario-' + id).classList.add('active');
  // Highlight the clicked tab
  var btns = document.querySelectorAll('.scenario-tab-btn');
  for (var i = 0; i < btns.length; i++) {{
    if ((id === 'aggregated' && i === 0) ||
        btns[i].getAttribute('onclick').indexOf("'" + id + "'") !== -1) {{
      btns[i].classList.add('active');
    }}
  }}
  // Re-apply suite toggle
  switchSuite(currentSuite);
}}
</script>
</body>
</html>
""".replace("{thead}", _THEAD)
