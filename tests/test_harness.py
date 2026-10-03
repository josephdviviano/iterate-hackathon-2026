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
        for rel in ["framework/core/ar.py", "framework/baseline/program.md", "framework/hypothesis/program.md",
                    "framework/hypothesis/research.py", "framework/hypothesis/lit.py",
                    "framework/hypothesis-lit/program.md", "framework/hypothesis-dr/program.md"]:  # fmt: skip
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


class HypothesisBase(Workspace):
    framework = "hypothesis"

    def setUp(self):
        super().setUp()
        self.llm_log = os.path.join(self.tmp, "llm.jsonl")
        os.environ.update(RESEARCH_LLM_CMD=os.path.join(ROOT, "tests", "fake_llm.py"), FAKE_LLM_LOG=self.llm_log)
        os.chmod(os.environ["RESEARCH_LLM_CMD"], 0o755)
        os.environ.pop("FAKE_LLM_REFUTE", None)
        fixture = os.path.join(self.tmp, "lit_fixture.json")
        json.dump({"papers": [
            {"title": f"Paper {k}", "abstract": f"Abstract {k} about learning rates.", "year": "2024",
             "arxiv": "", "url": f"https://example.org/{k}", "source": "fixture"} for k in range(25)],
            "texts": {"Paper 0": "Intro. Lower learning rates converge faster in short schedules on small networks. More."}},
            open(fixture, "w"))
        os.environ["RESEARCH_LIT_FIXTURE"] = fixture

    def tearDown(self):
        st = os.path.join(self.ws, ".ar", "lit_serve.json")
        if os.path.exists(st):
            try:
                os.kill(json.load(open(st))["pid"], 9)
            except (ProcessLookupError, KeyError):
                pass
        super().tearDown()

    def calls(self):
        return [json.loads(line) for line in open(self.llm_log)] if os.path.exists(self.llm_log) else []

    def start(self):
        self.ok("ar.py", "init")
        self.ok("research.py", "init")
        self.ok("research.py", "launch", "--idea", "-", "--description", "baseline")
        self.ok("ar.py", "wait")
        self.ok("research.py", "log", "--status", "keep", "--verdict", "-")
        self.ok("research.py", "snapshot")
        self.ok("research.py", "meta", "wait", "--max", "60")

    def results_of(self, name):
        lines = open(os.path.join(self.ws, name)).read().splitlines()
        h = lines[0].split("\t")
        return [dict(zip(h, line.split("\t"))) for line in lines[1:]]

    def wait_for(self, path, timeout=60):
        import time

        end = time.time() + timeout
        while not os.path.exists(path) and time.time() < end:
            time.sleep(0.2)
        self.assertTrue(os.path.exists(path), f"{path} never appeared")

    def prereg_draft(self, iid, lo, hi):
        draft = os.path.join(self.ws, "drafts", f"{iid}.prereg.md")
        open(draft, "w").write(
            f"## Pre-registration\n- Prediction: loss [{lo}, {hi}]\n- Supports if: loss drops\n"
            "- Refutes if: loss rises\n\n## Contingencies\n- If it lands in the predicted range: go on\n"
            "- If it moves the opposite way: revert\n- If it crashes: fix\n")
        return draft



class TestHypothesisFramework(HypothesisBase):
    def test_meta_steps_are_separate_calls_with_only_their_inputs(self):
        self.start()
        roles = [c["role"] for c in self.calls()]
        self.assertEqual(roles, ["search", "review", "revise", "ideate"])  # seed: revise, then ideate
        for c in self.calls():
            self.assertEqual(c["cwd_files"], [])  # an empty directory: nothing but the prompt
            self.assertEqual(c["tools"], "WebSearch,WebFetch" if c["role"] == "search" else "")
            self.assertEqual(c["model"], "claude-opus-5-5")
            self.assertTrue(c["has_results"])
        tree = self.ok("research.py", "tree")
        self.assertIn("H1.1", tree)
        self.assertIn("H2", tree)
        ideas = self.results_of("ideas.tsv")
        self.assertEqual([i["status"] for i in ideas], ["queued"] * 5)  # the bogus H99 idea was skipped
        meta = self.results_of("meta.tsv")
        self.assertEqual([m["kind"] for m in meta], ["revise", "ideate"])
        self.assertEqual(meta[0]["cost_usd"], "0.03")
        self.assertNotIn("hypo", self.ok("research.py", "--help"))  # the experimenter cannot edit the tree

    def test_large_inputs_reach_the_call(self):
        self.start()
        for k in range(60):  # notebook outcomes grow the "results" input well past the ~128 KB argv limit
            with open(os.path.join(self.ws, "notebook", f"E9{k:02d}.md"), "w") as f:
                f.write("## Outcome\n" + "long outcome text. " * 200 + "\n")
        with open(os.path.join(self.ws, "results.tsv"), "a") as f:
            for k in range(60):
                f.write(f"E9{k:02d}\tabc1234\t1.0\t0.5\tno\tno\tdiscard\t-\tfiller\n")
        research = load_research(self.ws)
        research.run_ideate()
        self.assertGreater(self.calls()[-1]["prompt_bytes"], 140_000)

    def test_full_cycle_and_auto_ideate(self):
        self.start()
        iid = self.ok("research.py", "idea", "next").split()[0]
        self.edit(lr=0.03)
        self.assertIn("sealed at launch", self.ok("research.py", "launch", "--idea", iid,
                                                  "--prereg", self.prereg_draft(iid, 0.5, 1.0)))  # fmt: skip
        self.ok("ar.py", "wait")
        out = self.ok("research.py", "log", "--status", "keep", "--verdict", "supports")
        self.assertIn("loss:hit", out)
        self.assertIn("blind", out)
        hypo = self.results_of("hypothesis.tsv")
        self.assertIn("E001+", [h for h in hypo if h["id"] == "H1.1"][0]["evidence"])
        self.assertEqual(self.results()[-1]["tag"], iid)
        self.assertEqual(self.git("status", "--porcelain"), "")  # research state stays untracked
        # the queue falls below 3 -> ideate starts by itself after a log
        for k in range(2):
            self.ok("research.py", "snapshot")
            i2 = self.ok("research.py", "idea", "next").split()[0]
            self.edit(lr=0.03, width=9 + k)
            self.ok("research.py", "launch", "--idea", i2)
            self.ok("ar.py", "wait")
            out = self.ok("research.py", "log", "--status", "discard", "--verdict", "inconclusive")
            self.git("reset", "-q", "--hard", "HEAD~1")
        self.assertIn("meta-step started in the background: ideate", out)
        self.ok("research.py", "meta", "wait", "--max", "60")
        self.assertEqual([c["role"] for c in self.calls()][-1], "ideate")

    def test_revise_cascades_and_crash_requeues(self):
        self.start()
        iid = self.ok("research.py", "idea", "next").split()[0]
        self.edit(width=128)
        self.ok("research.py", "launch", "--idea", iid)
        self.ok("ar.py", "wait")
        self.ok("research.py", "log", "--status", "crash", "--verdict", "refutes")
        self.assertEqual([i["status"] for i in self.results_of("ideas.tsv") if i["id"] == iid], ["queued"])
        self.assertIn("post-mortems", self.ok("research.py", "status") + "post-mortems")
        # a revise that refutes H1 retires its subtree and drops the queued ideas under it
        os.environ["FAKE_LLM_REFUTE"] = "H1"
        research = load_research(self.ws)
        research.run_revise()
        hypo = {h["id"]: h["status"] for h in self.results_of("hypothesis.tsv")}
        self.assertEqual((hypo["H1"], hypo["H1.1"], hypo["H3"]), ("refuted", "retired", "open"))
        self.assertTrue(all(i["status"] == "dropped" for i in self.results_of("ideas.tsv")))


class TestLiteratureAsync(HypothesisBase):
    framework = "hypothesis-lit"

    def test_agent_requests_become_verified_digests(self):
        self.start()
        out = self.ok("research.py", "lit", "ask", "--question", "does lower lr help?", "--hypothesis", "H1.1",
                      "--must", "reports lr ablations", "--exclude", "pretrained")  # fmt: skip
        self.assertIn("R001 queued", out)
        digest = os.path.join(self.ws, "lit", "digests", "D001.md")
        self.wait_for(digest)
        text = open(digest).read()
        self.assertIn("25 papers found, 4 relevant, 4 read in full, 1 verified claims", text)  # invented quote dropped
        self.assertIn("C0001", text)
        stages = [line.split("\t")[1] for line in open(os.path.join(self.ws, "lit", "ledger.tsv")).read().splitlines()[1:]]
        self.assertEqual(stages, ["plan", "triage", "triage"] + ["extract"] * 4 + ["synthesize"])
        self.assertIn("unread digests: D001", self.ok("research.py", "lit", "inbox"))
        self.assertIn("lower learning rates", self.ok("research.py", "lit", "read", "D001"))
        self.assertIn("unread digests: none", self.ok("research.py", "lit", "inbox"))
        research = load_research(self.ws)
        research.run_ideate()
        self.assertTrue(self.calls()[-1]["has_literature"])
        self.assertIn("(literature: async)", open(os.path.join(self.ws, "framework.json")).read().replace('"', "") + "(literature: async)")


class TestLiteratureForced(HypothesisBase):
    framework = "hypothesis-dr"

    def test_deep_research_runs_after_every_experiment(self):
        self.start()  # logs the baseline: forced deep research, then ideate
        self.ok("research.py", "meta", "wait", "--max", "60")
        self.assertTrue(os.path.exists(os.path.join(self.ws, "lit", "digests", "D001.md")))
        req = json.load(open(os.path.join(self.ws, "lit", "requests", "R001.json")))
        self.assertEqual((req["from"], req["after"], req["status"]), ("auto", "E000", "done"))
        kinds = [m["kind"] for m in self.results_of("meta.tsv")]
        self.assertEqual(kinds[-2:], ["deepresearch", "ideate"])
        self.assertTrue(self.calls()[-1]["has_literature"])
        self.assertIn("requests are automatic", self.fails("research.py", "lit", "ask", "--question", "x"))


def load_research(ws):
    import importlib.util

    spec = importlib.util.spec_from_file_location("research_under_test", os.path.join(ws, "research.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    unittest.main()
