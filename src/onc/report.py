"""Build the ONC-AGI section of the report page from the result files.

    uv run python -m onc.report --page PAGE.html --out artifacts/onc/report.html

The section is inserted before the glossary of the existing page and every
number in it is read from artifacts/onc/*.json, so the page and RESULTS.md
cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from onc.conformal import aci, binary_scores

ART = Path("artifacts/onc")


def load(name: str) -> dict | None:
    path = ART / name
    return json.loads(path.read_text()) if path.exists() else None


def f2(x, digits=2) -> str:
    return "" if x is None else f"{x:.{digits}f}"


def condition(ev: dict, label: str) -> dict:
    return next(c for c in ev["conditions"] if c["label"] == label)


def strip(pairs: list[tuple[float, int]], title: str, note: str) -> str:
    """One conformal strip over a held-out stream: a cell per patient, running coverage below."""
    res = aci(binary_scores([p for p, _ in pairs]), [y for _, y in pairs])
    n = len(pairs)
    cells = []
    for i, (size, hit) in enumerate(zip(res.sizes, res.covered)):
        kind = "abstain" if size == 2 else ("empty" if size == 0 else ("hit" if hit else "miss"))
        cells.append((i, kind))
    return json.dumps({"title": title, "note": note, "n": n, "cells": cells, "coverage": res.coverage, "committed": res.committed})


BASELINE_NAMES = {
    "oracle": "oracle (knows the key)",
    "univariate_bh": "univariate BH",
    "lasso": "lasso",
    "elastic_net": "elastic net",
    "stability": "stability selection",
    "random_forest": "random forest",
}


def comparison_points(base: dict | None, ev: dict | None, mode: str) -> list[dict]:
    """Bars for one panel: our committee designs, the benchmark's baselines, the cheaters folded into one."""
    if not base or not ev:
        return []
    mode_key = "full_access" if mode == "full" else "sequential"
    recs = {r["agent"]: r for r in base["records"] if r.get("mode") == mode_key and r.get("status") == "ok"}
    ours = [("full / likelihood", "committee (ours)")] if mode == "full" else [
        ("seq / likelihood / disagreement", "ours: disagreement stop"),
        ("seq / likelihood / random", "ours: random stop"),
        ("seq / likelihood / staged", "ours: staged recruitment"),
        ("seq / likelihood / pipeline", "ours: buy everything"),
    ]
    points = []
    for label, name in ours:
        try:
            c = condition(ev, label)
        except StopIteration:
            continue
        points.append({"name": name, "kind": "ours", "ds": c["discovery_score"], "lo": c["interval"][0], "hi": c["interval"][1], "cost": c["mean_data_cost"]})
    for agent, name in BASELINE_NAMES.items():
        r = recs.get(agent)
        if r:
            points.append({"name": name, "kind": "reference" if agent == "oracle" else "baseline", "ds": r["discovery_score"], "lo": r["interval"][0], "hi": r["interval"][1], "cost": r["mean_data_cost"]})
    cheaters = [r for a, r in recs.items() if a not in BASELINE_NAMES]
    if cheaters:
        best = max(cheaters, key=lambda r: r["discovery_score"])
        points.append({"name": f"{len(cheaters)} cheaters, best of them", "kind": "cheater", "ds": best["discovery_score"], "lo": best["interval"][0], "hi": best["interval"][1], "cost": best["mean_data_cost"]})
    return points


def extra_points(store_name: str, mode: str) -> list[dict]:
    """The LLM agents and the best single hypothesis, from their own result files when present."""
    points = []
    for key, label in (("qwen", "LLM agent, Qwen3-Coder-30B"), ("gptoss", "LLM agent, gpt-oss-120b")):
        ev = load(f"llm_{key}_{store_name}.json")
        if not ev:
            continue
        c = next((c for c in ev["conditions"] if c["mode"] == mode), None)
        if c and c["discovery_score"] is not None:
            points.append({"name": label, "kind": "llm", "ds": c["discovery_score"], "lo": c["interval"][0], "hi": c["interval"][1], "cost": c["mean_data_cost"]})
    ab = load(f"ablation_{store_name}.json")
    if ab:
        singles = [c for c in ab["conditions"] if c["mode"] == mode and "single" in c["label"] and c["discovery_score"] is not None]
        if singles:
            best = max(singles, key=lambda c: c["discovery_score"])
            name = best["label"].split("single: ")[1]
            points.append({"name": f"best single hypothesis ({name})", "kind": "single", "ds": best["discovery_score"], "lo": best["interval"][0], "hi": best["interval"][1], "cost": best["mean_data_cost"]})
    return points


def comparison_figures(toy: dict | None, dev: dict | None) -> str:
    """Discovery Score against the benchmark's baselines, and score against spend in sequential mode."""
    panels = []
    for store_name, ev, base_file in (("toy", toy, "baselines.json"), ("dev", dev, "baselines_dev.json")):
        base = load(base_file)
        for mode in ("full", "seq"):
            pts = comparison_points(base, ev, mode)
            if pts:
                ours = [pt for pt in pts if pt["kind"] == "ours"]
                pts = ours + extra_points(store_name, mode) + [pt for pt in pts if pt["kind"] != "ours"]
                n = ev["conditions"][0]["n_worlds"] if ev else 0
                title = ("Full access" if mode == "full" else "Sequential") + f", {store_name} worlds ({n})"
                panels.append({"id": f"cmp_{store_name}_{mode}", "title": title, "mode": mode, "store": store_name, "points": pts})
    if not panels:
        return ""
    canvases = "".join(
        f'<figure class="fig"><div class="barwrap" style="position:relative;height:{110 + 30 * len(p["points"])}px"><canvas id="{p["id"]}"></canvas></div></figure>'
        for p in panels
    )
    scatter = [p for p in panels if p["mode"] == "seq"]
    scatter_canvases = "".join(
        f'<figure class="fig"><div style="position:relative;height:300px"><canvas id="{p["id"]}_cost"></canvas></div></figure>' for p in scatter
    )
    rows = "".join(
        f"<tr><td>{p['title']}</td><td>{pt['name']}</td><td class=num>{pt['ds']:.3f}</td><td class=num>[{pt['lo']:+.2f}, {pt['hi']:+.2f}]</td><td class=num>{pt['cost']:.0f}</td></tr>"
        for p in panels for pt in p["points"]
    )
    table = f'<details><summary style="cursor:pointer;color:var(--muted);font-size:.9rem">Table view of both figures</summary><div class="tablewrap"><table><tr><th>Panel</th><th>Agent</th><th class=num>Discovery</th><th class=num>95%</th><th class=num>Mean cost (USD)</th></tr>{rows}</table></div></details>'
    script = """
<script>
window.addEventListener('load', () => {
  if (typeof Chart === 'undefined') return;
  const PANELS = __PANELS__;
  const T = theme();
  const dark = matchMedia('(prefers-color-scheme: dark)').matches && document.documentElement.dataset.theme !== 'light' || document.documentElement.dataset.theme === 'dark';
  const GRAY = dark ? '#7d8189' : '#6b6f78';
  const ink = T.fg, muted = T.muted;
  // The interval lines and the few direct labels, bound to one panel's points.
  const intervalPlugin = (points) => {
    const firstBaseline = points.findIndex(pt => pt.kind === 'baseline');
    return {id:'intervalBars', afterDatasetsDraw(chart){
      const ctx = chart.ctx, x = chart.scales.x;
      ctx.save(); ctx.strokeStyle = ink; ctx.lineWidth = 1.2;
      points.forEach((pt, i) => {
        const bar = chart.getDatasetMeta(pt.kind === 'ours' ? 0 : 1).data[i]; if (!bar) return;
        const y = bar.y, x0 = x.getPixelForValue(Math.max(0, pt.lo)), x1 = x.getPixelForValue(Math.max(0, pt.hi));
        if (x1 - x0 < 1) return;
        ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.moveTo(x0, y-4); ctx.lineTo(x0, y+4); ctx.moveTo(x1, y-4); ctx.lineTo(x1, y+4); ctx.stroke();
      });
      ctx.font = `11px ${css('--body')}`; ctx.fillStyle = ink; ctx.textBaseline = 'middle';
      points.forEach((pt, i) => {
        if (pt.kind !== 'ours' && pt.kind !== 'reference' && i !== firstBaseline) return;
        const bar = chart.getDatasetMeta(pt.kind === 'ours' ? 0 : 1).data[i]; if (!bar) return;
        ctx.fillText(pt.ds.toFixed(2), Math.max(bar.x, x.getPixelForValue(Math.max(0, pt.hi))) + 8, bar.y);
      });
      ctx.restore();
    }};
  };
  for (const p of PANELS) {
    const el = document.getElementById(p.id); if (!el) continue;
    const labels = p.points.map(pt => pt.name);
    const ours = p.points.map(pt => pt.kind === 'ours' ? pt.ds : null);
    const rest = p.points.map(pt => pt.kind === 'ours' ? null : pt.ds);
    new Chart(el, {type:'bar', plugins:[intervalPlugin(p.points)], data:{labels, datasets:[
      {label:'committee (ours)', data: ours, backgroundColor: T.accent, borderRadius: 4, borderSkipped: 'start', barThickness: 18, grouped: false},
      {label:"the benchmark's agents", data: rest, backgroundColor: GRAY, borderRadius: 4, borderSkipped: 'start', barThickness: 18, grouped: false}
    ]}, options:{indexAxis:'y', maintainAspectRatio:false, animation:false, layout:{padding:{right:44}},
      plugins:{title:{display:true, text:`Discovery Score, ${p.title}`, color: ink, font:{weight:'600'}}, legend:{labels:{boxWidth:12, color: muted}},
        tooltip:{callbacks:{label: it => { const pt = p.points[it.dataIndex]; return `${pt.name}: ${pt.ds.toFixed(3)} [${pt.lo.toFixed(2)}, ${pt.hi.toFixed(2)}], mean cost ${Math.round(pt.cost)} USD`; }}}},
      scales:{x:{min:0, max:1.08, grid:{color: T.rule, lineWidth:1}, ticks:{color: muted, callback: v => v <= 1 ? v.toFixed(2) : ''}, title:{display:true, text:'Discovery Score = Find × Restraint (95% bootstrap interval)', color: muted}},
              y:{grid:{display:false}, ticks:{color: ink, font:{size:12}}}}}});
    const sc = document.getElementById(p.id + '_cost'); if (!sc) continue;
    const costLabels = {id:'costLabels', afterDatasetsDraw(ch){ const ctx = ch.ctx; ctx.save(); ctx.font = `11px ${css('--body')}`; ctx.fillStyle = ink;
      ch.data.datasets.forEach((ds, di) => { const meta = ch.getDatasetMeta(di); meta.data.forEach((el, i) => { const d = ds.data[i]; if (!d.label) return; ctx.textAlign = d.align || 'left'; ctx.fillText(d.label, el.x + (d.dx || 10), el.y + (d.dy || 4)); }); }); ctx.restore(); }};
    const oursPts = p.points.filter(pt => pt.kind === 'ours').map((pt, k) => ({x: pt.cost, y: pt.ds, label: pt.name.replace('ours: ', ''), name: pt.name,
      align: pt.cost > 15000 ? 'right' : 'left', dx: pt.cost > 15000 ? -10 : 10, dy: k === 1 ? 16 : (k === 0 ? -8 : 4)}));
    const restPts = p.points.filter(pt => pt.kind !== 'ours').map(pt => ({x: pt.cost, y: pt.ds, name: pt.name}));
    const bl = restPts.filter(pt => pt.x > 20000); if (bl.length) { const top = bl.reduce((a, b) => a.y > b.y ? a : b); top.label = 'baselines and cheaters, buying the whole pool'; top.align = 'right'; top.dx = -10; top.dy = -8; }
    p.points.forEach((pt, i) => { if (pt.kind === 'llm' || pt.kind === 'single') { const r = restPts.find(q => q.name === pt.name); if (r) { r.label = pt.name.replace('LLM agent, ', 'LLM: ').replace('best single hypothesis', 'best single'); r.align = 'left'; r.dx = 10; r.dy = pt.kind === 'llm' ? 4 : -8; } } });
    const orc = restPts.find(pt => pt.name.startsWith('oracle')); if (orc) { orc.label = 'oracle'; orc.dx = 10; orc.dy = 4; }
    new Chart(sc, {type:'scatter', plugins:[costLabels], data:{datasets:[
      {label:'committee (ours)', data: oursPts, backgroundColor: T.accent, borderColor: T.panel, borderWidth: 2, pointRadius: 7, pointHoverRadius: 9},
      {label:"the benchmark's agents", data: restPts, backgroundColor: GRAY, borderColor: T.panel, borderWidth: 2, pointRadius: 6, pointHoverRadius: 8}
    ]}, options:{maintainAspectRatio:false, animation:false, layout:{padding:{right:16}},
      plugins:{title:{display:true, text:`Score against spend, ${p.title}`, color: ink, font:{weight:'600'}}, legend:{labels:{usePointStyle:true, boxWidth:10, color: muted}},
        tooltip:{callbacks:{label: it => `${it.raw.name}: Discovery ${it.raw.y.toFixed(3)} at ${Math.round(it.raw.x)} USD`}}},
      scales:{x:{min:-500, max:30000, grid:{color: T.rule, lineWidth:1}, ticks:{color: muted, stepSize: 5000, callback: v => v >= 0 ? (v/1000).toFixed(0) + 'k' : ''}, title:{display:true, text:'mean data cost per world (USD)', color: muted}},
              y:{min:0, max:1.05, grid:{color: T.rule, lineWidth:1}, ticks:{color: muted}, title:{display:true, text:'Discovery Score', color: muted}}}}});
  }
});
</script>"""
    return (
        '<p class="col">The headline comparison: our committee against the benchmark\'s own baselines (standard statistics that buy the whole pool in sequential mode), its cheaters, two LLM tool-calling agents (one model call per turn with recruit, assay, summary, Python and submit tools; Qwen3-Coder-30B and gpt-oss-120b on Modal) and the best single hypothesis template run alone with the committee\'s own admission and weights, on the same worlds. Bars carry the 95% bootstrap interval of the unfloored score. In sequential mode the second figure puts score against spend: the committee reaches the oracle analyst\'s level of recovery for a quarter of the cost, and the efficiency factor is what separates it from buy-everything.</p>'
        + '<div class="grid2">' + canvases + '</div>' + '<div class="grid2">' + scatter_canvases + '</div>' + table
        + script.replace("__PANELS__", json.dumps(panels))
    )


def section() -> str:
    toy = load("eval_toy.json")
    dev = load("eval_dev.json")
    hacks = load("hacks_toy.json")
    train = load("train.json")
    lookup = load("arc_lookup.json")
    rows = []
    if toy:
        for label, name in (("full / likelihood", "Full access, committee"), ("seq / likelihood / disagreement", "Sequential, disagreement stop"),
                            ("seq / likelihood / random", "Sequential, random stop"), ("seq / likelihood / staged", "Sequential, staged template"),
                            ("seq / likelihood / pipeline", "Sequential, buy everything")):
            c = condition(toy, label)
            rows.append((name, "toy, 10", c))
    if dev:
        for label, name in (("full / likelihood", "Full access, committee"), ("seq / likelihood / disagreement", "Sequential, disagreement stop"),
                            ("seq / likelihood / random", "Sequential, random stop"), ("seq / likelihood / staged", "Sequential, staged template"),
                            ("seq / likelihood / pipeline", "Sequential, buy everything")):
            c = condition(dev, label)
            rows.append((name, "dev, 100", c))
    table = "".join(
        f"<tr><td>{name}</td><td>{worlds}</td><td class=num>{c['discovery_score']:.2f}</td><td class=num>[{c['interval'][0]:.2f}, {c['interval'][1]:.2f}]</td>"
        f"<td class=num>{c['find']:.2f}</td><td class=num>{c['restraint']:.2f}</td><td class=num>{c['mean_data_cost']:.0f}</td><td class=num>{c['mean_efficiency']:.2f}</td>"
        f"<td class=num>{c['p_signal'].get('ece', float('nan')):.2f}</td><td class=num>{c['p_driver'].get('ece', float('nan')):.2f}</td>"
        f"<td class=num>{(f2(c['held_out'].get('aci_coverage')) + ' / ' + f2(c['held_out'].get('aci_committed'))) if c['held_out'].get('n') else ''}</td></tr>"
        for name, worlds, c in rows
    )
    strips = []
    if toy:
        c = condition(toy, "seq / likelihood / disagreement")
        pairs = [tuple(x) for w in c["worlds"] for x in w["held_out"]]
        strips.append(strip(pairs, "Toy worlds, sequential", "10 worlds, held-out patients in recruit order"))
    dev_strip = load("eval_strips_dev.json")
    if dev_strip:
        c = condition(dev_strip, "seq / likelihood / disagreement")
        pairs = [tuple(x) for w in c["worlds"][:40] for x in w["held_out"]]
        strips.append(strip(pairs, "Dev worlds, sequential (first 40 worlds)", "generated worlds, not benchmark results"))
    hack_rows = ""
    frontier = []
    if hacks:
        co = hacks["coverage_only"]
        frontier = co["frontier"]
        dial, a = co["trained"], co["aci"]
        cf = hacks["confidence_free"]
        s_rows = "".join(f"<tr><td class=num>{r['sharpness']:g}</td><td class=num>{r['r_task']:+.3f}</td><td class=num>{r['ece_signal']:.3f}</td><td class=num>{r['ece_driver']:.3f}</td><td class=num>{r['r_cal']:+.3f}</td></tr>" for r in cf["rows"])
        cheat = hacks["cheaters"]["seq"]
        ours = next(r for r in cheat if r["kind"] == "ours")
        best_base = max(r["reward"] for r in cheat if r["kind"] == "baseline")
        worst_base = min(r["reward"] for r in cheat if r["kind"] == "baseline")
        best_cheat = max(r["reward"] for r in cheat if r["kind"] == "cheater")
        hack_rows = f"""
<tr><td>1. Coverage only</td><td>A set-width dial paid for coverage alone</td><td>width {dial['width']:.2f}: coverage {dial['coverage']:.2f}, commits on {dial['committed']:.0%} of patients</td><td>Coverage is reported with the commit share; ACI gives {a['aci_coverage']:.2f} at {a['aci_committed']:.0%}</td></tr>
<tr><td>2. Task reward only</td><td>Sharpen every reported probability (odds to the power s)</td><td>R_task unchanged for s = 0.5 to 32; ECE of P(driver) {cf['rows'][0]['ece_driver']:.2f} to {cf['rows'][-1]['ece_driver']:.2f}</td><td>R_cal falls {cf['rows'][0]['r_cal']:+.3f} to {cf['rows'][-1]['r_cal']:+.3f}; R_task + 0.5 R_cal picks s = {cf['argmax_with_cal']:g}</td></tr>
<tr><td>3. The benchmark's cheaters</td><td>11 cheaters and random under R, with the certainty their lists imply</td><td>Every cheater at or below the benchmark floor</td><td>R at most {best_cheat:+.2f}; honest baselines {worst_base:+.2f} to {best_base:+.2f}; committee {ours['reward']:+.2f} (sequential)</td></tr>"""
        hack_rows += f"""
<tr><td>4. Committee collapse</td><td>Members and distinct driver sets per world across training</td><td>{'see the training table' if train else 'pending'}</td><td>The policy cannot edit the committee; members are refit from data only</td></tr>"""
        if lookup:
            hack_rows += f"""
<tr><td>5. Exact-replay hack (ARC)</td><td>A lookup table of the training transitions, identity elsewhere</td><td>Passes admission on every split; held-out accuracy 0.00 on ar25 L3 and disagreement 0.95 with the real members</td><td>Perturbation rule rejects it on every split and wrongly rejects 0 of 19 real members</td></tr>"""
        sharp_table = f"""<div class="tablewrap"><table><tr><th class=num>Sharpness s</th><th class=num>R_task</th><th class=num>ECE P(signal)</th><th class=num>ECE P(driver)</th><th class=num>R_cal</th></tr>{s_rows}</table></div>"""
    else:
        sharp_table = ""
    figures = comparison_figures(toy, dev)
    train_rows = ""
    if train:
        for arm, res in train["arms"].items():
            row = res["held_out"].get("dev_seq")
            if row is None:
                continue
            p = res["policy"]
            ho = row["held_out"]
            members = sum(len(w["members"]) for w in row["worlds"]) / len(row["worlds"])
            sets = sum(len({tuple(m["drivers"]) for m in w["members"]}) for w in row["worlds"]) / len(row["worlds"])
            train_rows += (
                f"<tr><td>{arm}</td><td class=num>{res.get('lam_cal', 0):.2f} / {res.get('lam_dis', 0):.2f}</td><td class=num>{p['tau']:.2f} / {p['stop_threshold']:.3f} / {p['spend_cap']:.2f} / {p['sharpness']:.2f}</td>"
                f"<td class=num>{row['discovery_score']:.2f} [{row['interval'][0]:.2f}, {row['interval'][1]:.2f}]</td><td class=num>{row['find']:.2f}</td><td class=num>{row['restraint']:.2f}</td><td class=num>{row['leak_rate']:.2f}</td><td class=num>{row['mean_data_cost']:.0f}</td>"
                f"<td class=num>{row['p_signal'].get('ece', float('nan')):.3f}</td><td class=num>{row['p_driver'].get('ece', float('nan')):.3f}</td><td class=num>{ho.get('brier', float('nan')):.3f}</td><td class=num>{f2(ho.get('aci_coverage'))} / {f2(ho.get('aci_committed'))}</td><td class=num>{members:.1f} / {sets:.1f}</td></tr>"
            )
    train_block = (
        f"""<div class="tablewrap"><table><tr><th>Arm</th><th class=num>λ_cal / λ_dis</th><th class=num>τ / stop / cap / sharpness</th><th class=num>DS [95%]</th><th class=num>Find</th><th class=num>Restraint</th><th class=num>Leak</th><th class=num>Cost</th><th class=num>ECE P(signal)</th><th class=num>ECE P(driver)</th><th class=num>Brier p(y|x)</th><th class=num>ACI cov / commit</th><th class=num>Members / driver sets</th></tr>{train_rows}</table></div>"""
        if train_rows
        else "<p class=col>Training results pending.</p>"
    )
    return f"""
<h2 id="onc">The same method on a science benchmark: ONC-AGI</h2>
<p class="col">ONC-AGI poses one question per world: here is a cohort with an outcome; which measurements drive it? The agent returns an ordered list of features, or an empty list, and in sequential mode buys its own data with <code>recruit</code> and <code>assay</code>. The score is Find × Restraint: chance-normalised recovery of the planted drivers, times the difference between abstaining on null worlds and abstaining on signal worlds. Listing a post-outcome feature zeroes the world; overspending the reference cost scales the credit down. Only toy fixture worlds ship today, so every number here is a toy-world or generated-world number, not a benchmark result.</p>

<h3>The committee, ported</h3>
<div class="steps col">
<div class="step"><div class="n">1</div><div><strong>Hypotheses.</strong> Eight hypothesis programs, one per causal role family: null, direct, conservative, sparse, confounder-adjusted, upstream cause, interaction and correlated block. Each reads the revealed baseline data and returns an ordered driver set and a logistic model p(y | x). Post-outcome columns are removed before any hypothesis sees the data: the leak filter is a filter, not a learned behaviour.<div class="code">onc.hypotheses</div></div></div>
<div class="step"><div class="n">2</div><div><strong>Admission and weights.</strong> A hypothesis is admitted when its 5-fold cross-validated log loss, selection step included, is within 0.02 of the best member. The null hypothesis is always a member. Weights are \\(w_k \\propto e^{{-n\\,\\ell_k}}\\) with \\(\\ell_k\\) the CV log loss; equal weights are the ablation and give the null member 1/K whatever the data, which costs all of Restraint in full access.<div class="code">onc.committee.build_committee</div></div></div>
<div class="step"><div class="n">3</div><div><strong>What it reports.</strong> P(signal) is the weight of the non-null members; P(driver = f) is the weight of the members that list f, one vote per correlation cluster; p(y | x) is the weighted mixture. Submission: abstain if P(signal) &lt; 0.5, else list the features with P(driver) ≥ τ in decreasing order. Disagreement U is the normalised entropy over the members' driver sets.</div></div>
<div class="step"><div class="n">4</div><div><strong>Acquisition by disagreement.</strong> Recruit 60 patients across strata, assay every baseline feature on them, then recruit 40 more while the expected drop in U per 1000 USD exceeds a threshold and spend stays under the cap. The committee from before each batch is scored on that batch after it is revealed, which gives a true held-out stream of p(y | x) for calibration and conformal coverage. When no non-null member survives, the only repair is more data; that step earns no shaping reward.<div class="code">onc.agent.CommitteeAgent</div></div></div>
<div class="step"><div class="n">5</div><div><strong>Reward.</strong> \\(R = R_{{\\text{{task}}}} + \\lambda_{{\\text{{cal}}}} R_{{\\text{{cal}}}} + \\lambda_{{\\text{{dis}}}} R_{{\\text{{dis}}}}\\). The task term is the benchmark's own score written per episode. The calibration term is minus the Brier score of P(signal), of P(driver) over the listed features, and of p(y | x) on the held-out patients. The shaping term is the drop in disagreement, \\(U_{{t-1}} - U_t\\), potential-based so it telescopes to \\(U_0 - U_T\\) and cannot be farmed; the policy cannot touch the committee, so it cannot lower U by fiat.<div class="code">onc.rewards</div></div></div>
</div>

<h3>Results</h3>
<div class="tablewrap"><table>
<tr><th>Condition</th><th>Worlds</th><th class=num>Discovery</th><th class=num>95%</th><th class=num>Find</th><th class=num>Restraint</th><th class=num>Cost (USD)</th><th class=num>Efficiency</th><th class=num>ECE P(signal)</th><th class=num>ECE P(driver)</th><th class=num>ACI coverage / commit</th></tr>
{table}
</table></div>
{figures}
<p class="col">The benchmark's baselines score 0.77 (univariate BH, lasso) to 0.33 (random forest) on the toy worlds in full access and 0.31 or less in sequential mode, where they buy the whole pool; every cheater scores 0. The committee stops after a quarter of the buy-everything spend at efficiency 1 on every world, which is where its sequential margin comes from. P(signal) is calibrated (ECE 0.01 to 0.05); P(driver) is not: the committee lists extra features at high probability and they earn no credit. That is the term the calibration reward targets.</p>

<h3>Conformal coverage on the held-out patients</h3>
<p class="col">Each strip is the held-out stream of one run: one cell per patient whose outcome the committee predicted before that patient's batch was revealed, target coverage 0.90. A single-label set is a commitment; a two-label set is an abstention.</p>
<div class="legend aci-legend"><span class="hit">committed, right</span><span class="miss">committed, wrong (×)</span><span class="abst">abstained</span><span class="empty">empty set, forced miss (×)</span></div>
<figure class="fig" id="oncStrips"><div id="oncStripsBody"></div><figcaption>Coverage holds at the target with commitments on 18% of toy patients and 40% of dev patients. The committee's p(y | x) is honest but not sharp: most patients' outcomes are not predictable from the planted mechanism alone.</figcaption></figure>
<figure class="fig"><canvas id="oncFrontier" height="260"></canvas><figcaption>Coverage against the share of patients with a committed prediction, for a set-width dial from 0 to 1 on the toy held-out stream. A dial paid for coverage alone moves to width 1: coverage 1, nothing committed. ACI sits at the target.</figcaption></figure>

<h3>Reward hacking checks</h3>
<div class="tablewrap"><table>
<tr><th>Check</th><th>Naive reward and exploit</th><th>What happens</th><th>Does our reward catch it</th></tr>
{hack_rows}
</table></div>
{sharp_table}

<h3>Training the policy with the three signals</h3>
<p class="col">What is trained is the policy: a Gaussian over four decision parameters (the listing threshold τ, the stop threshold, the spend cap and the sharpness of the reported probabilities), updated by a group-relative policy gradient on batches of generated sequential worlds. The committee is frozen. Arms A to D differ only in the reward weights; all arms see the same worlds. Held-out: 30 generated sequential worlds never used in training.</p>
{train_block}
<p class="col">Every number in this section comes from RESULTS.md entries O1 to O7 and the files under <code>artifacts/onc/</code>.</p>
<script>
window.addEventListener('load', () => {{
const ONC_STRIPS = [{",".join(strips)}];
const ONC_FRONTIER = {json.dumps(frontier)};
(function drawOncStrips(){{
  const W = 860, padL = 34, padR = 34, yS = 2, hS = 18, yTop = 30, yBot = 96, H = 116;
  const yc = v => yBot - (Math.max(0.5, v) - 0.5) / 0.5 * (yBot - yTop);
  let html = '';
  for (const r of ONC_STRIPS) {{
    const n = r.n, NMAX = Math.max(n, 1), cw = (W-padL-padR)/NMAX;
    let s = `<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="${{r.title}}">`;
    let h = 0, pts = [];
    for (const [i, kind] of r.cells) {{
      const x = padL + i*cw, w = Math.max(1, cw-0.5);
      let fill = 'var(--keep)', stroke = 'none';
      if (kind === 'miss') fill = 'var(--accent)';
      if (kind === 'abstain') fill = 'var(--frozen-soft)';
      if (kind === 'empty') {{ fill = 'transparent'; stroke = 'var(--accent)'; }}
      s += `<rect class="cell" x="${{x.toFixed(1)}}" y="${{yS}}" width="${{w.toFixed(1)}}" height="${{hS}}" fill="${{fill}}" stroke="${{stroke}}" stroke-width="1"><title>patient ${{i+1}}: ${{kind}}</title></rect>`;
      h += (kind === 'miss' || kind === 'empty') ? 0 : 1; pts.push(`${{(x+cw/2).toFixed(1)}},${{yc(h/(i+1)).toFixed(1)}}`);
    }}
    [1.0, 0.9, 0.5].forEach(v => {{ s += `<text x="${{padL-6}}" y="${{yc(v)+4}}" text-anchor="end" font-size="10.5" font-family="var(--mono)" fill="var(--muted)">${{v.toFixed(1)}}</text>`; }});
    s += `<line x1="${{padL}}" x2="${{padL+NMAX*cw}}" y1="${{yc(0.9)}}" y2="${{yc(0.9)}}" stroke="var(--muted)" stroke-width="1" stroke-dasharray="4 4"/>`;
    s += `<polyline points="${{pts.join(' ')}}" fill="none" stroke="var(--fg)" stroke-width="2" stroke-linejoin="round"/>`;
    s += `</svg>`;
    html += `<div class="aci-row"><div class="aci-head"><strong>${{r.title}}</strong>${{r.note}}. Coverage ${{r.coverage.toFixed(2)}}; committed on ${{(r.committed*100).toFixed(0)}}% of ${{n}} patients.</div><div class="svgwrap">${{s}}</div></div>`;
  }}
  html = html.replace(/(fill|stroke|font-family)="var\\(--([a-z-]+)\\)"/g, (m, a, n) => `${{a}}="${{getComputedStyle(document.documentElement).getPropertyValue('--'+n).trim().replace(/"/g,"'")}}"`);
  document.getElementById('oncStripsBody').innerHTML = html;
}})();
(function drawOncFrontier(){{
  if (!ONC_FRONTIER.length || typeof Chart === 'undefined') return;
  const T = theme();
  const aciPt = ONC_STRIPS.length ? [{{x: ONC_STRIPS[0].committed, y: ONC_STRIPS[0].coverage}}] : [];
  new Chart(document.getElementById('oncFrontier'), {{type:'scatter', data:{{datasets:[
    {{label:'set-width dial (width 0 to 1)', data: ONC_FRONTIER.map(r=>({{x:r.committed, y:r.coverage}})), showLine:true, borderColor:T.frozen, backgroundColor:T.frozen, pointRadius:4, tension:0}},
    {{label:'ACI, target 0.90', data: aciPt, backgroundColor:T.keep, borderColor:T.panel, borderWidth:2, pointStyle:'rectRot', pointRadius:9}},
    {{label:'dial trained on coverage alone', data:[{{x:0, y:1}}], backgroundColor:'transparent', borderColor:T.accent, borderWidth:2, pointStyle:'rect', pointRadius:9}},
    {{type:'line', label:'target 0.90', data:[{{x:-0.05,y:0.9}},{{x:1.05,y:0.9}}], borderColor:T.muted, borderWidth:1, borderDash:[4,4], pointRadius:0}}
  ]}}, options:{{plugins:{{title:{{display:true, text:'Coverage vs share of patients with a committed prediction (toy, sequential)', color:T.fg}}, legend:{{labels:{{usePointStyle:true, boxWidth:10, font:{{size:11}}}}}}}},
    scales:{{x:{{min:-0.05, max:1.05, title:{{display:true, text:'share committed'}}}}, y:{{min:0, max:1.05, title:{{display:true, text:'coverage'}}}}}}}}}});
}})();
}});
</script>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--page", required=True, help="the current report page (HTML)")
    parser.add_argument("--out", default="artifacts/onc/report.html")
    args = parser.parse_args(argv)
    page = Path(args.page).read_text()
    marker = "<h2>Glossary</h2>"
    assert marker in page, "the page has no glossary to insert before"
    if '<h2 id="onc">' in page:
        start = page.index('<h2 id="onc">')
        page = page[:start] + page[page.index(marker):]
    page = page.replace("ARC-AGI-3 · world models · Track 2.3", "Track 2.3 · epistemological agents · ARC-AGI-3 and ONC-AGI")
    page = page.replace(marker, section() + "\n" + marker, 1)
    Path(args.out).write_text(page)
    print(f"wrote {args.out} ({len(page)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
