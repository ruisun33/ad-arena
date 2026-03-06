"""Static HTML leaderboard generator for GitHub Pages.

Reads ``results/leaderboard.json`` produced by :class:`ResultsStore` and
writes a self-contained ``index.html`` to ``docs/leaderboard/`` with two
tabs: Campaign Efficiency and Strategy Optimization.
"""

from __future__ import annotations

import json
import logging
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> None:
        """Read ``leaderboard.json`` and produce ``index.html`` with dual suite tabs."""
        leaderboard_path = self.results_dir / "leaderboard.json"
        if not leaderboard_path.exists():
            logger.error("leaderboard.json not found at %s", leaderboard_path)
            return

        raw = leaderboard_path.read_text(encoding="utf-8")
        entries: list[dict] = json.loads(raw)

        # Build rows for each suite, ranked by the respective score
        ce_rows = self._build_rows(entries, score_key="campaign_efficiency_score")
        so_rows = self._build_rows(entries, score_key="strategy_optimization_score")

        scenarios = sorted({e["scenario_name"] for e in entries})
        scenario_options = self._render_scenario_options(scenarios)

        ce_table_body = self._render_table_body(ce_rows)
        so_table_body = self._render_table_body(so_rows)

        html = _HTML_TEMPLATE.format(
            ce_table_body=ce_table_body,
            so_table_body=so_table_body,
            scenario_options=scenario_options,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / "index.html"
        out_path.write_text(html, encoding="utf-8")
        logger.info("Leaderboard written to %s", out_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_rows(
        entries: list[dict],
        score_key: str = "campaign_efficiency_score",
    ) -> list[dict]:
        """Derive display-ready rows with computed metrics, ranked by *score_key*."""
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
            learning_days_ratio = (
                learning_days / total_classifiable
                if total_classifiable
                else 0.0
            )

            # Use the selected score key for the profit column
            profit = e.get(score_key, e.get("total_profit", 0.0))

            rows.append({
                "model_name": e.get("model_name", ""),
                "scenario_name": e.get("scenario_name", ""),
                "model_type": e.get("model_type", "baseline"),
                "profit": profit,
                "roas": roas,
                "cpa": cpa,
                "ctr": ctr,
                "cvr": cvr,
                "conversions": total_conversions,
                "learning_rate": e.get("learning_rate", 0.0),
                "learning_days_ratio": learning_days_ratio,
                "api_calls": e.get("llm_api_calls", 0),
            })

        # Sort descending by profit (the suite-specific score), assign rank
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
                f'<tr data-scenario="{_esc(r["scenario_name"])}">'
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
    def _render_scenario_options(scenarios: list[str]) -> str:
        options = ['<option value="all">All Scenarios</option>']
        for s in scenarios:
            options.append(f'<option value="{_esc(s)}">{_esc(s)}</option>')
        return "\n".join(options)


def _esc(text: str) -> str:
    """Minimal HTML-entity escaping."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ------------------------------------------------------------------
# HTML template — single self-contained page with dual suite tabs
# ------------------------------------------------------------------

_THEAD = """\
<thead>
<tr>
  <th data-col="0" data-type="num">Rank</th>
  <th data-col="1" data-type="str">Model</th>
  <th data-col="2" data-type="str">Type</th>
  <th data-col="3" data-type="num">Profit</th>
  <th data-col="4" data-type="num">ROAS</th>
  <th data-col="5" data-type="num">CPA</th>
  <th data-col="6" data-type="num">CTR</th>
  <th data-col="7" data-type="num">CVR</th>
  <th data-col="8" data-type="num">Conversions</th>
  <th data-col="9" data-type="num">Learning Rate</th>
  <th data-col="10" data-type="num">Learning Days %</th>
  <th data-col="11" data-type="num">API Calls</th>
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
  .subtitle {{ color:var(--muted); margin-bottom:1.5rem; }}
  .tabs {{ display:flex; gap:0; margin-bottom:1rem; }}
  .tab-btn {{
    padding:.5rem 1.25rem; border:1px solid var(--border); background:var(--card);
    cursor:pointer; font-size:.9rem; font-weight:600; color:var(--muted);
    border-bottom:2px solid transparent;
  }}
  .tab-btn:first-child {{ border-radius:6px 0 0 0; }}
  .tab-btn:last-child {{ border-radius:0 6px 0 0; }}
  .tab-btn.active {{ color:var(--accent); border-bottom-color:var(--accent); background:#fff; }}
  .tab-panel {{ display:none; }}
  .tab-panel.active {{ display:block; }}
  .controls {{ display:flex; align-items:center; gap:.75rem; margin-bottom:1rem; }}
  .controls label {{ font-weight:600; font-size:.9rem; }}
  .controls select {{ padding:.35rem .6rem; border:1px solid var(--border); border-radius:4px;
                      font-size:.9rem; background:var(--card); }}
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

  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('ce')">Campaign Efficiency</button>
    <button class="tab-btn" onclick="switchTab('so')">Strategy Optimization</button>
  </div>

  <div class="controls">
    <label for="scenario-filter">Scenario:</label>
    <select id="scenario-filter" onchange="filterScenario(this.value)">
      {scenario_options}
    </select>
  </div>

  <div id="tab-ce" class="tab-panel active">
    <div class="table-wrap">
      <table id="table-ce">""" + _THEAD + """
        <tbody>{ce_table_body}</tbody>
      </table>
    </div>
  </div>

  <div id="tab-so" class="tab-panel">
    <div class="table-wrap">
      <table id="table-so">""" + _THEAD + """
        <tbody>{so_table_body}</tbody>
      </table>
    </div>
  </div>

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
        minus spend) across all 30 days. Rewards models that balance
        exploration cost against optimization gain efficiently from day one.</li>
      <li><strong>Strategy Optimization</strong> &mdash; Profit from only the
        final 7 days (days 24&ndash;30). Rewards models that converge on the
        best strategy, regardless of how much they spent learning in the
        first 23 days.</li>
    </ul>
    <h3>Benchmark Mechanics</h3>
    <ul>
      <li><strong>Scenarios</strong> &mdash; Each scenario defines a unique
        market: keywords with different economics, competitor behavior, and
        budget constraints.</li>
      <li><strong>Deterministic Seeds</strong> &mdash; A root seed drives all
        randomness. Same seed + same strategy = identical results.</li>
      <li><strong>Adaptation Metrics</strong> &mdash; The harness observes
        daily strategy changes and computes <em>learning rate</em>,
        <em>strategy volatility</em>, and <em>convergence day</em>.</li>
      <li><strong>Baselines</strong> &mdash; Algorithmic baselines run
        alongside every LLM to provide reference performance levels.</li>
    </ul>
  </div>

  <footer>
    Generated by Ad Arena &middot; <a href="https://github.com/ad-arena">GitHub</a>
  </footer>
</div>

<script>
function switchTab(id) {{
  document.querySelectorAll('.tab-panel').forEach(function(p) {{ p.classList.remove('active'); }});
  document.querySelectorAll('.tab-btn').forEach(function(b) {{ b.classList.remove('active'); }});
  document.getElementById('tab-' + id).classList.add('active');
  // highlight the clicked button
  var btns = document.querySelectorAll('.tab-btn');
  for (var i = 0; i < btns.length; i++) {{
    if ((id === 'ce' && i === 0) || (id === 'so' && i === 1)) btns[i].classList.add('active');
  }}
  // re-apply scenario filter to the newly active tab
  filterScenario(document.getElementById('scenario-filter').value);
}}

function filterScenario(value) {{
  var active = document.querySelector('.tab-panel.active');
  if (!active) return;
  var rows = active.querySelectorAll('tbody tr');
  for (var i = 0; i < rows.length; i++) {{
    if (value === 'all' || rows[i].getAttribute('data-scenario') === value) {{
      rows[i].style.display = '';
    }} else {{
      rows[i].style.display = 'none';
    }}
  }}
}}
</script>
</body>
</html>
"""
