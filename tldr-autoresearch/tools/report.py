#!/usr/bin/env python3
"""Progress report of the RLTL;DR autoresearch run (reads data/ only; safe to run any time).

    python3 tools/report.py [--data DIR] [--groups N]      (or ./ctl.sh report, which passes config's data dir)
"""
import argparse
import json
import os

# default data dir: $RLTLDR_ROOT/data, else <repo>/data next to tools/
ROOT = os.environ.get("RLTLDR_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def jl(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--groups", type=int, default=10)
    a = ap.parse_args()
    D = a.data
    st = json.load(open(f"{D}/driver_state.json")) if os.path.exists(f"{D}/driver_state.json") else {}
    ts = json.load(open(f"{D}/trainer_state.json")) if os.path.exists(f"{D}/trainer_state.json") else {}
    led = jl(f"{D}/ledger.jsonl")
    dm, tm = jl(f"{D}/metrics_driver.jsonl"), jl(f"{D}/metrics_trainer.jsonl")

    print("== overview")
    if st:
        p = st["parent"]
        print(f"group {st['group']} (attempt {st['k']}/8 in progress)  attempts={st['n_attempts']}  kept={st['n_kept']}  "
              f"void={st.get('n_void', 0)}")
        print(f"best val_bpb {p['val_bpb']:.6f}  commit {p['commit'][:7]}")
    print(f"policy version (trainer): {ts.get('version', 0)}   training runs in ledger: "
          f"{sum(1 for e in led if e.get('status') != 'refused')}")

    if os.path.exists(f"{D}/results.tsv"):
        rows = open(f"{D}/results.tsv").read().strip().split("\n")[1:]
        kept = [r for r in rows if "\tkeep\t" in r]
        print("\n== kept experiments (best-so-far trajectory)")
        for r in kept:
            c, v, mem, _, desc = r.split("\t", 4)
            print(f"  {c}  {float(v):.6f}  {desc[:110]}")
        print(f"  ({len(rows)} experiments total, {len(kept)} kept)")

    print(f"\n== last {a.groups} groups (deconfounded metrics, paper App. D)")
    print("  grp  succ  first  no-ins  w/-ins  ins-adv  crash  policy  best")
    for m in dm[-a.groups:]:
        f = lambda x: "  -  " if x is None else f"{x:5.2f}"
        print(f"  {m['group']:3d}  {m['n_success']}/{m['n']}  {str(m['first_attempt_success'])[:5]:5s}  "
              f"{f(m['success_rate_no_insight'])}  {f(m['success_rate_with_insight'])}  {f(m.get('insight_advantage'))}  "
              f"{m['n_crash']:5d}  {str(m.get('policy_versions')):7s} {m['best_val_bpb']:.6f}")

    print(f"\n== insights of the current group")
    for i in st.get("insights", []):
        print(f"  [{i['idx']}] {i['hint']}")

    print(f"\n== last {a.groups} trainer updates")
    for m in tm[-a.groups:]:
        u = m.get("update") or {}
        d = m.get("data") or {}
        print(f"  group {m['group']}: updated={m.get('updated')} v={m.get('version')} grpo_tok={d.get('n_grpo_tokens')} "
              f"sft_tok={d.get('n_sft_tokens')} steps={u.get('steps')} t={m.get('t_update', 0):.0f}s "
              f"clip/epoch={u.get('clipfrac_by_epoch')} mismatch={u.get('mismatch_signed')} "
              f"is_clamped={u.get('is_w_clamped_frac')} gnorm={[round(g, 2) for g in u.get('grad_norm', [])]}")


if __name__ == "__main__":
    main()
