"""Tests for the task-agnostic harness: ar.py (both frameworks) and research.py (hypothesis).

They run on the toy task, plus metric extraction on a real CIFAR-100 harness output, so the same
framework code is exercised against two different task specs.

    python3 -m unittest discover tests
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAKE = os.path.join(ROOT, "arena", "make_workspace.py")
TOY = os.path.join(ROOT, "tasks", "toy")

CIFAR_OUTPUT = """trial 1/3: accuracy=75.56% training=12.302s inference=0.117s
{
  "requested_trials": 3,
  "complete": true,
  "mean_accuracy": 0.7562333333333333,
  "accuracy_std": 0.0021,
  "mean_training_time": 12.731000000000002,
  "training_time_std": 0.04,
  "qualified": true
}
Results: /x/results/arena/20261003T134755Z-5bffb574
"""


def load_ar():
    import importlib.util

    spec = importlib.util.spec_from_file_location("ar_under_test", os.path.join(ROOT, "framework", "core", "ar.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Workspace(unittest.TestCase):
    framework = "hypothesis"

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ws = os.path.join(self.tmp, "ws")
        subprocess.run([sys.executable, MAKE, self.ws, "--framework", self.framework, "--task", TOY,
                        "--repo", os.path.join(TOY, "repo")], check=True, capture_output=True)  # fmt: skip
        self.git("config", "user.email", "t@t")
        self.git("config", "user.name", "t")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def git(self, *a):
        return subprocess.run(["git", *a], cwd=self.ws, check=True, capture_output=True, text=True).stdout

    def cli(self, tool, *a):
        return subprocess.run([sys.executable, tool, *a], cwd=self.ws, capture_output=True, text=True)

    def ok(self, tool, *a):
        p = self.cli(tool, *a)
        self.assertEqual(p.returncode, 0, f"{tool} {a}:\n{p.stdout}\n{p.stderr}")
        return p.stdout

    def fails(self, tool, *a):
        p = self.cli(tool, *a)
        self.assertNotEqual(p.returncode, 0, f"{tool} {a} unexpectedly succeeded:\n{p.stdout}")
        return p.stderr

    def edit(self, lr=0.1, width=8, epochs=5, msg="exp"):
        with open(os.path.join(self.ws, "model", "params.py"), "w") as f:
            f.write(f"LR = {lr}\nWIDTH = {width}\nEPOCHS = {epochs}\n")
        self.git("commit", "-qam", msg)
        return self.git("rev-parse", "--short=7", "HEAD").strip()

    def results(self):
        lines = open(os.path.join(self.ws, "results.tsv")).read().splitlines()
        h = lines[0].split("\t")
        return [dict(zip(h, line.split("\t"))) for line in lines[1:]]


class TestRunner(Workspace):
    framework = "baseline"

    def test_workspace_has_only_the_baseline_framework(self):
        names = sorted(os.listdir(self.ws))
        self.assertIn("program.md", names)
        self.assertNotIn("research.py", names)
        self.assertEqual(self.git("status", "--porcelain"), "")

    def test_greedy_loop_on_the_toy_task(self):
        self.ok("ar.py", "init")
        out = self.ok("ar.py", "run", "--description", "baseline")
        self.assertIn("loss=1.15", out)
        self.assertIn("no branch tip yet", self.ok("ar.py", "log", "--status", "keep"))
        tip = self.git("rev-parse", "--short=7", "HEAD").strip()
        # better loss but still infeasible: improves (infeasible_order = loss)
        self.edit(lr=0.03, width=8, epochs=5)
        self.ok("ar.py", "run")
        self.assertIn("no run meets the constraints yet", self.ok("ar.py", "log", "--status", "keep"))
        # feasible now (loss <= 0.6) beats infeasible regardless of time
        self.edit(lr=0.03, width=16, epochs=10)
        self.ok("ar.py", "run")
        self.assertIn("first run that meets the constraints", self.ok("ar.py", "log", "--status", "keep"))
        # faster but infeasible: never an improvement
        self.edit(lr=0.03, width=4, epochs=2)
        self.ok("ar.py", "run")
        self.assertIn("misses a constraint", self.fails("ar.py", "log", "--status", "keep"))
        out = self.ok("ar.py", "log", "--status", "discard")
        self.assertIn("git reset --hard", out)
        rows = self.results()
        self.assertEqual([r["status"] for r in rows], ["keep", "keep", "keep", "discard"])
        self.assertEqual([r["feasible"] for r in rows], ["no", "no", "yes", "no"])
        self.assertNotEqual(tip, rows[-1]["commit"])

    def test_crash_and_guards(self):
        self.ok("ar.py", "init")
        self.ok("ar.py", "run")
        self.ok("ar.py", "log", "--status", "keep")
        self.edit(width=128)  # simulated OOM
        out = self.ok("ar.py", "run")
        self.assertIn("no metrics found", out)
        p = self.cli("ar.py", "log", "--status", "discard")
        self.assertIn("recording a crash", p.stderr)
        self.assertEqual(self.results()[-1]["status"], "crash")
        # uncommitted edits, and committed edits outside the editable paths, are refused
        with open(os.path.join(self.ws, "model", "params.py"), "a") as f:
            f.write("# dirty\n")
        self.assertIn("commit the experiment first", self.fails("ar.py", "start"))
        self.git("checkout", "--", "model")
        with open(os.path.join(self.ws, "train.py"), "a") as f:
            f.write("# harness change\n")
        self.git("commit", "-qam", "touch harness")
        self.assertIn("outside the editable paths", self.fails("ar.py", "start"))

    def test_wait_timeout_and_abort(self):
        self.ok("ar.py", "init")
        with open(os.path.join(self.ws, "model", "params.py"), "a") as f:
            f.write("import time\ntime.sleep(5)\n")
        self.git("commit", "-qam", "slow")
        self.ok("ar.py", "start")
        p = self.cli("ar.py", "wait", "--max", "0.5", "--poll", "0.1")
        self.assertEqual(p.returncode, 3, p.stdout)
        self.ok("ar.py", "abort", "--reason", "too slow")
        p = self.cli("ar.py", "log", "--status", "keep")  # no metrics: recorded as a crash, never kept
        self.assertIn("recording a crash", p.stderr)
        self.assertEqual(self.results()[-1]["status"], "crash")
        self.assertIn("[aborted: too slow]", self.results()[-1]["description"])


class TestTaskSpecs(unittest.TestCase):
    def test_cifar_spec_reads_the_benchmark_output(self):
        ar = load_ar()
        t = json.load(open(os.path.join(ROOT, "tasks", "cifar100", "task.json")))
        m = ar.extract_metrics(t, CIFAR_OUTPUT)
        self.assertEqual(m["complete"], True)
        self.assertAlmostEqual(m["accuracy"], 0.75623, places=4)
        self.assertAlmostEqual(m["time"], 12.731, places=3)
        self.assertTrue(ar.feasible(t, m))
        self.assertFalse(ar.feasible(t, {**m, "accuracy": 0.7529}))
        self.assertFalse(ar.feasible(t, {**m, "complete": False}))
        crashed = ar.extract_metrics(t, CIFAR_OUTPUT.replace('"mean_accuracy": 0.75623', '"mean_accuracy": null')
                                     .replace("0.7562333333333333", "null"))  # fmt: skip
        self.assertFalse(ar.valid(t, crashed))
        # among feasible runs, lower time wins; an infeasible faster run never does
        self.assertTrue(ar.improves(t, {**m, "time": 12.0}, m)[0])
        self.assertFalse(ar.improves(t, {**m, "time": 13.0}, m)[0])
        self.assertFalse(ar.improves(t, {**m, "time": 5.0, "accuracy": 0.70}, m)[0])
        self.assertTrue(ar.improves(t, m, {**m, "accuracy": 0.74})[0])

    def test_framework_files_contain_no_task_knowledge(self):
        words = ["cifar", "accuracy", "val_bpb", "nanochat", "torch", "gpu", "epoch", "75%", "0.753", "seeds"]
        for rel in ["framework/core/ar.py", "framework/baseline/program.md",
                    "framework/hypothesis/program.md", "framework/hypothesis/research.py"]:  # fmt: skip
            text = open(os.path.join(ROOT, rel)).read().lower()
            hits = [w for w in words if w in text]
            self.assertEqual(hits, [], f"{rel} mentions {hits}")

    def test_programs_share_everything_but_the_method(self):
        base = open(os.path.join(ROOT, "framework/baseline/program.md")).read()
        hyp = open(os.path.join(ROOT, "framework/hypothesis/program.md")).read()
        for title in ["## The task", "## Running experiments"]:
            a = base[base.index(title):].split("\n## ")[0]
            b = hyp[hyp.index(title):].split("\n## ")[0]
            self.assertEqual(a, b, title)
        self.assertEqual(base[base.index("**NEVER STOP**"):], hyp[hyp.index("**NEVER STOP**"):])


class TestHypothesisFramework(Workspace):
    framework = "hypothesis"

    def prereg(self, exp, lo, hi):
        path = os.path.join(self.ws, "notebook", f"{exp}.md")
        text = open(path).read()
        text = text.replace("- Prediction:\n", f"- Prediction: loss [{lo}, {hi}]\n", 1)
        for k, v in [("- Supports if:", "- Supports if: loss drops"), ("- Refutes if:", "- Refutes if: loss rises"),
                     ("- If it lands in the predicted range:", "- If it lands in the predicted range: go on"),
                     ("- If it moves the opposite way:", "- If it moves the opposite way: revert"),
                     ("- If it crashes:", "- If it crashes: fix")]:  # fmt: skip
            text = text.replace(k, v, 1)
        open(path, "w").write(text)

    def test_full_cycle(self):
        self.ok("ar.py", "init")
        self.ok("research.py", "init")
        self.ok("research.py", "hypo", "add", "--statement", "lr is too high")
        self.ok("research.py", "hypo", "add", "--parent", "H1", "--statement", "lower lr helps")
        self.ok("research.py", "launch", "--idea", "-", "--description", "baseline")
        self.ok("ar.py", "wait")
        self.ok("research.py", "log", "--status", "keep", "--verdict", "-")
        self.assertIn("world model", self.fails("research.py", "launch", "--idea", "-", "--description", "x"))
        self.ok("research.py", "snapshot")
        iid = self.ok("research.py", "idea", "add", "--hypothesis", "H1.1", "--description", "lr 0.03",
                      "--expected", "loss down").strip()  # fmt: skip
        self.ok("research.py", "idea", "next")
        self.edit(lr=0.03)
        # drafted while the previous run trained, sealed at launch: blind by construction
        draft = os.path.join(self.ws, "drafts", f"{iid}.prereg.md")
        open(draft, "w").write(
            "## Pre-registration\n- Prediction: loss [0.5, 1.0]\n- Supports if: loss drops\n"
            "- Refutes if: loss rises\n\n## Contingencies\n- If it lands in the predicted range: go on\n"
            "- If it moves the opposite way: revert\n- If it crashes: fix\n")
        self.assertIn("sealed at launch", self.ok("research.py", "launch", "--idea", iid, "--prereg", draft))
        self.ok("ar.py", "wait")
        out = self.ok("research.py", "log", "--status", "keep", "--verdict", "supports")
        self.assertIn("loss:hit", out)
        self.assertIn("blind", out)
        hypo = [ln for ln in open(os.path.join(self.ws, "hypothesis.tsv")) if ln.startswith("H1.1\t")][0]
        self.assertIn("E001+", hypo)
        self.assertEqual(self.results()[-1]["tag"], iid)
        self.assertEqual(self.git("status", "--porcelain"), "")  # research state stays untracked

    def test_crash_requeues_and_misses_need_postmortems(self):
        self.ok("ar.py", "init")
        self.ok("research.py", "init")
        self.ok("research.py", "hypo", "add", "--statement", "wider is better")
        self.ok("research.py", "launch", "--idea", "-", "--description", "baseline")
        self.ok("ar.py", "wait")
        self.ok("research.py", "log", "--status", "keep", "--verdict", "-")
        self.ok("research.py", "snapshot")
        iid = self.ok("research.py", "idea", "add", "--hypothesis", "H1", "--description", "width 128",
                      "--expected", "x").strip()  # fmt: skip
        self.ok("research.py", "idea", "next")
        self.edit(width=128)
        self.ok("research.py", "launch", "--idea", iid)
        self.ok("ar.py", "wait")
        self.ok("research.py", "log", "--status", "crash", "--verdict", "refutes")
        ideas = open(os.path.join(self.ws, "ideas.tsv")).read()
        self.assertIn("\tqueued\t", ideas)
        self.git("reset", "-q", "--hard", "HEAD~1")
        self.ok("research.py", "snapshot")
        iid2 = self.ok("research.py", "idea", "add", "--hypothesis", "H1", "--description", "width 16",
                       "--expected", "x").strip()  # fmt: skip
        self.ok("research.py", "idea", "set", iid, "--status", "dropped")
        self.ok("research.py", "idea", "next")
        self.edit(width=16)
        self.ok("research.py", "launch", "--idea", iid2)
        self.prereg("E002", 5.0, 6.0)  # a range the outcome will miss
        self.ok("research.py", "prereg")
        self.ok("ar.py", "wait")
        self.assertIn("loss:miss", self.ok("research.py", "log", "--status", "keep", "--verdict", "supports"))
        self.ok("research.py", "snapshot")
        status = self.ok("research.py", "status")
        self.assertIn("post-mortems: E001, E002", status)  # the crash and the missed prediction


if __name__ == "__main__":
    unittest.main()
