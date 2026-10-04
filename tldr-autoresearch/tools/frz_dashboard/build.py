#!/usr/bin/env python3
"""Build the frozen-v5 dashboard (insights vs none): dashboard_frz/index.html.

The frz arms (./ctl_frz.sh) keep the h2h arm layout under data/h2h/<arm>/ (ledger.jsonl -> rl/ledger.jsonl,
repo/, runner_state.json, ...), so this is tools/h2h_dashboard/build.py (same data collection, same template)
with the frz arm labels, page title and output dir. Arms come from H2H_CONFIG, i.e. data/frz/arms.json
(tools/frz_dashboard/watch.sh sets it). Stdlib only, read-only on the run.

    H2H_CONFIG=data/frz/arms.json /usr/bin/python3 tools/frz_dashboard/build.py [--out FILE] [--data H2H_DIR] [--status]
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "h2h_dashboard_build", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "h2h_dashboard", "build.py"))
h2h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h2h)

h2h.DESCRIPTION = __doc__.split("\n")[0]
h2h.OUT_DIR = "dashboard_frz"
h2h.ARM_TEXT = {
    "x1": ("v5 + insights", "frozen v5 in the training harness, TL;DR insights on"),
    "x2": ("v5, no insights", "frozen v5 in the training harness, insights off"),
}
h2h.TEMPLATE_SUBS = (
    ("<title>Base vs v5 head-to-head</title>", "<title>Frozen v5: insights vs none</title>"),
    ("<h1>Base model vs RLTL;DR v5</h1>", "<h1>Frozen v5 in the training harness: insights vs none</h1>"),
)

if __name__ == "__main__":
    h2h.main()
