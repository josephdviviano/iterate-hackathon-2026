#!/usr/bin/env python3
"""
test_ar_run.py - GPU-free tests for ar_run.py / ar_bootstrap.py / ar_parse.py (no CUDA context is created anywhere).

Each case builds a throw-away git repo whose train.py is a tiny fake (prints a summary block, crashes, sleeps,
trains a toy model on the CPU, ...) and whose prepare.py is a CPU-only stand-in with prepare's interface, runs
ar_run.py against it (inside the real train sandbox, under the trusted bootstrap) with the real autoresearch venv
and a private ledger and lock dir, and checks stdout, exit code and the ledger line. The GPU-violation case
monkeypatches ar_run.gpu_apps so that it reports the fake training process on another GPU.
The --isolate cases (the h2h runner's mode) also check what a hostile train.py can reach: no key / signer in its
process or the judge's /proc entries, hidden run data and repos, the masked validation shard, no file carried from
one run to the next through the compile cache, arm-neutral paths in tracebacks; plus the warm-up time-budget bounds.

The sandboxed cases need passwordless sudo, the autoresearch checkout with its venv ($AR_TEST_REPO, default
$RLTLDR_ROOT/autoresearch) and the UUID of a GPU visible to nvidia-smi ($AR_TEST_GPU, default agent_gpu_uuid from
rltldr/config.py + config.json); no CUDA context is created unless --gpu is given. The scratch dir must lie under
$HOME (--isolate hides everything else there) and not under /tmp (the train sandbox mounts a private tmpfs there);
default $RLTLDR_ROOT/scratch, override with AR_TEST_TMP. $RLTLDR_ROOT defaults to the directory above tools/.

  python3 tools/test_ar_run.py --unit   # only the pure unit tests (timing rule, summary block): no sudo, GPU or venv
  python3 tools/test_ar_run.py          # all GPU-free cases (~6 min)
  python3 tools/test_ar_run.py --gpu    # + the CUDA path on that GPU (~1 min more):
      a toy CUDA model through --isolate, logits via CUDA IPC in forced small chunks; the judge's val_bpb must equal
      the same quantity computed inside train.py, and a non-causal model must be caught by the probe
"""
import json
import os
import pwd
import shutil
import signal
import subprocess
import sys
import tempfile
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import ar_run  # noqa: E402  (unit tests of the timing rule and the summary block)

ROOT = os.environ.get("RLTLDR_ROOT") or os.path.dirname(TOOLS)
HOME = pwd.getpwuid(os.getuid()).pw_dir                   # = ar_run.HOME (what --isolate hides)
MAIN_REPO = os.environ.get("AR_TEST_REPO") or os.path.join(ROOT, "autoresearch")
VENV = os.path.join(MAIN_REPO, ".venv")
PY = os.environ.get("RLTLDR_TRUSTED_PY") or "/usr/bin/python3"   # trusted stdlib-only interpreter for ar_run.py
TMP_BASE = os.environ.get("AR_TEST_TMP") or os.path.join(ROOT, "scratch")
OTHER_GPU = "GPU-00000000-0000-0000-0000-000000000000"   # a GPU that is not ours (mocked nvidia-smi output)
GPU = os.environ.get("AR_TEST_GPU") or ""                  # set in main() from the config when unset


def config_gpu():
    """agent_gpu_uuid from rltldr/config.py (+ $RLTLDR_ROOT/config.json), or "" if it cannot be loaded."""
    sys.path.insert(0, ROOT)
    try:
        from rltldr.config import load_config
        return load_config().agent_gpu_uuid or ""
    except Exception:
        return ""

FAKE_PREPARE = '''"""CPU-only stand-in for autoresearch's prepare.py (tests): same interface, tiny random data."""
import os
import torch
MAX_SEQ_LEN = 16
TIME_BUDGET = 2
VOCAB_SIZE = 32
DATA_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "data")
VAL_FILENAME = "shard_06542.parquet"


class Tokenizer:
    @classmethod
    def from_directory(cls):
        return cls()

    def get_vocab_size(self):
        return VOCAB_SIZE


def _document_batches(split, tokenizer_batch_size=128):
    while True:
        yield [[1, 2, 3]] * tokenizer_batch_size, 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    g = torch.Generator().manual_seed(0 if split == "train" else 1)
    while True:
        x = torch.randint(0, VOCAB_SIZE, (B, T + 1), generator=g)
        yield x[:, :-1].contiguous(), x[:, 1:].contiguous(), 1


def evaluate_bpb(model, tokenizer, batch_size):
    x, y, _ = next(make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val"))
    loss = model(x, y, reduction="none")
    assert loss.numel() == x.numel()
    return 1.081234
'''

PROGRESS = "".join(f"\rstep {s:05d} ({s / 3:.1f}%) | loss: 3.{s:06d} | lrm: 1.00 | dt: 600ms | "
                   f"tok/sec: 873,000 | mfu: 20.0% | epoch: 1 | remaining: {300 - s}s    " for s in range(1, 60))
SUMMARY = ("print('---'); print('val_bpb:          1.081234'); print('training_seconds: 300.2'); "
           "print('total_seconds:    331.0'); print('peak_vram_mb:     45012.5'); print('mfu_percent:      19.80'); "
           "print('total_tokens_M:   265.3'); print('num_steps:        506'); print('num_params_M:     50.3'); "
           "print('depth:            8')")

# toy model + an optimizer written like train.py's MuonAdamW (torch.optim.Optimizer subclass, custom step)
TOY = '''
import time
import torch
import torch.nn.functional as F
from prepare import MAX_SEQ_LEN, VOCAB_SIZE, make_dataloader, evaluate_bpb


class Model(torch.nn.Module):   # per-position model: causal by construction
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(VOCAB_SIZE, 8)
        self.out = torch.nn.Linear(8, VOCAB_SIZE)

    def forward(self, idx, targets=None, reduction="mean"):
        logits = self.out(self.emb(idx))
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), reduction=reduction)


class MuonAdamWLike(torch.optim.Optimizer):
    def __init__(self, params):
        super().__init__([{"params": list(params), "lr": 0.01}], defaults={})

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    p.add_(p.grad, alpha=-g["lr"])


model = Model()
'''


def toy_train(compile_s, train_s, optimizer="muonlike", pre="", printed_train_s=None, warm_batches=0, warm_sleep=0.0):
    """train.py: `compile_s` of fake compilation in step 0, 10 more warm-up steps, then `train_s` of training
    (like train.py: time counted only for step > 10), one trusted evaluation, upstream summary block.
    warm_batches / warm_sleep: extra train batches / seconds in each of steps 0..10 (hidden extra training)."""
    opt = {"muonlike": "opt = MuonAdamWLike(model.parameters())",
           "adamw": "opt = torch.optim.AdamW(model.parameters(), lr=1e-3)",
           "manual": "opt = None"}[optimizer]
    step_code = ("opt.step(); opt.zero_grad(set_to_none=True)" if optimizer != "manual" else
                 "with torch.no_grad():\n        for p in model.parameters():\n"
                 "            p -= 0.01 * p.grad; p.grad = None")
    return TOY + pre + f'''
{opt}
loader = make_dataloader(None, 2, MAX_SEQ_LEN, "train")
x, y, _ = next(loader)
step, total = 0, 0.0
while True:
    t0 = time.time()
    loss = model(x, y)
    loss.backward()
    x, y, _ = next(loader)
    if step == 0:
        time.sleep({compile_s})
    if step <= 10:
        for _ in range({warm_batches}):
            next(loader)
        time.sleep({warm_sleep})
    {step_code}
    if step > 10:
        time.sleep(0.05)
    dt = time.time() - t0
    if step > 10:
        total += dt
    step += 1
    if step > 10 and total >= {train_s}:
        break
val = evaluate_bpb(model, None, 2)
print("---")
print(f"val_bpb:          {{val:.6f}}")
print(f"training_seconds: {{{printed_train_s if printed_train_s is not None else 'total'}:.1f}}")
print("peak_vram_mb:     12.5")
print(f"num_steps:        {{step}}")
print("depth:            1")
'''


FAKES = {
    "ok": ("import sys\nfrom prepare import evaluate_bpb\nprint('Vocab size: 8,192')\n"
           f"print({PROGRESS!r}, end='', flush=True)\nprint()\n" + TOY.split("model = Model()")[0] +
           "evaluate_bpb(Model(), None, 2)\n" + f"{SUMMARY}\n"),
    "syntax": "import os\nx = (\n",
    "fail_loss": f"print({PROGRESS!r}, end='', flush=True)\nprint('FAIL')\nraise SystemExit(1)\n",
    "oom": ("print('Model config: ...')\n"
            "raise RuntimeError('torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 8.00 GiB')\n"),
    "big_tail": ("import sys\nprint('\\r' * 5000 + 'x' * 100000)\n"
                 "for i in range(400): print(f'noise line {i} ' + 'y' * 300, file=sys.stderr)\n"
                 "raise ValueError('boom ' + 'z' * 1000)\n"),
    "sleep": "import time\nprint('training...', flush=True)\ntime.sleep(120)\n",
    "tamper": "import os\nos.environ['CUDA_VISIBLE_DEVICES'] = '3'\n",
    # trusted training-time anchors (run with --budget 2)
    "anchor_ok": toy_train(1.0, 2.0),
    "anchor_adamw": toy_train(0.5, 2.0, optimizer="adamw"),
    # trains 13 s, prints 2.0 s and rebinds time.monotonic: still flagged (anchor-based 13 s > 2 + 10)
    "anchor_over": toy_train(0.2, 13.0, pre="time.monotonic = lambda: 0.0\n", printed_train_s="2.0"),
    # prefetching moves the 2nd-batch anchor before a long compile: the old rule would count the compile
    "prefetch": toy_train(23.0, 2.0, pre="_pref = make_dataloader(None, 2, MAX_SEQ_LEN, 'train')\n"
                                         "_b = [next(_pref) for _ in range(3)]\n"),
    # no torch optimizer: falls back to the 2nd-batch rule (grace 20 s)
    "no_opt": toy_train(0.5, 2.0, optimizer="manual"),
    # hidden training in the warm-up steps that train.py does not count: 40 extra batches per step / 3.5 s per step
    "warm_batches": toy_train(0.2, 2.0, warm_batches=40),
    "warm_slow": toy_train(0.2, 2.0, warm_sleep=3.5),
    # what can a hostile train.py reach? (it prints its findings and then evaluates honestly)
    "forge": TOY.split("model = Model()")[0] + '''
import gc, os, sys
found = []
f = sys._getframe()
def secret(v):
    if isinstance(v, (bytes, bytearray, str)) and len(v) == 64:
        return all(c in "0123456789abcdef" for c in (v if isinstance(v, str) else v.decode("latin-1")))
    return type(v).__name__ in ("Judge", "Signer") or "record.jsonl" in str(getattr(v, "name", ""))
while f is not None:      # the reviewers' attack: walk up to the bootstrap's frame and use its signer
    for k, v in list(f.f_globals.items()) + list(f.f_locals.items()):
        if k in ("_emit", "_KEY", "_S", "_REC", "key", "rec") or secret(v):
            found.append("frame:" + k)
    f = f.f_back
for o in gc.get_objects():
    try:
        if secret(o) or (callable(o) and getattr(o, "__name__", "") == "_emit"):
            found.append("gc:" + type(o).__name__)
    except Exception:
        pass
for p in ("/proc/1/environ", "/proc/1/mem", "/proc/1/maps", "/proc/1/fd/7", "/proc/1/fd/8", "/proc/1/fd/9"):
    try:
        with open(p, "rb") as fh:
            fh.read(16)
        found.append("read:" + p)
    except Exception:
        pass
try:
    os.kill(1, 9)
except Exception:
    pass
print("FORGE found", sorted(set(found)), flush=True)
evaluate_bpb(Model(), None, 2)
''' + SUMMARY + "\n",
    "persist": ("import os\nd = os.environ['TORCHINDUCTOR_CACHE_DIR']\np = os.path.join(d, 'carry.pt')\n"
                "print('PERSIST before', 'present' if os.path.exists(p) else 'absent', flush=True)\n"
                "os.makedirs(d, exist_ok=True)\nopen(p, 'w').write('weights')\n"
                "print('PERSIST wrote', os.path.exists(p), flush=True)\nraise SystemExit(3)\n"),
}
SNOOP = '''import os
for p in {hidden!r}:
    print("SNOOP", p, "VISIBLE" if os.path.exists(p) else "absent")
val = os.path.join(os.path.expanduser("~"), ".cache/autoresearch/data/shard_06542.parquet")
try:
    print("SNOOP val bytes", len(open(val, "rb").read()))
except Exception as e:
    print("SNOOP val error", type(e).__name__)
print("SNOOP cwd", os.getcwd())
print("SNOOP home", sorted(os.listdir(os.path.expanduser("~"))))
raise RuntimeError("snoop done")
'''


def make_repo(root, kind, src=None):
    repo = os.path.join(root, f"repo_{kind}")
    if os.path.exists(repo):
        return repo
    os.makedirs(repo)
    for f in ("pyproject.toml", "uv.lock", ".python-version"):
        shutil.copy(os.path.join(MAIN_REPO, f), repo)
    with open(os.path.join(repo, "prepare.py"), "w") as f:
        f.write(FAKE_PREPARE)
    with open(os.path.join(repo, "train.py"), "w") as f:
        f.write("# parent version\n")
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)
    git("init", "-q"); git("config", "user.name", "t"); git("config", "user.email", "t@t")
    git("add", "."); git("commit", "-qm", "parent")
    with open(os.path.join(repo, "train.py"), "w") as f:
        f.write(src if src is not None else FAKES[kind])
    git("commit", "-qam", f"candidate {kind}")
    return repo


def ar_cmd(repo, ledger, lockdir, extra=(), prepare_sha=None):
    if prepare_sha is None:
        prepare_sha = subprocess.run(["sha256sum", os.path.join(repo, "prepare.py")], capture_output=True,
                                     text=True).stdout.split()[0]
    return [PY, "-I", os.path.join(TOOLS, "ar_run.py"), "--repo", repo, "--venv", VENV, "--ledger", ledger,
            "--gpu", GPU, "--prepare-sha", prepare_sha, "--lock-dir", lockdir, "--poll", "1",
            "--cache-dir", os.path.join(os.path.dirname(ledger), "cache"), "--desc", "test", *extra]


def gpu_tests(root, ledger, lockdir):
    """The CUDA path of the judge (logits through CUDA IPC, forced into 3-row chunks) on the test GPU."""
    for causal in (True, False):
        kind = "cuda_causal" if causal else "cuda_noncausal"
        repo = make_repo(root, kind, TOY_CUDA.format(causal=causal))
        with open(os.path.join(repo, "prepare.py"), "w") as f:
            f.write(FAKE_PREPARE_CUDA)
        p = run(ar_cmd(repo, ledger, lockdir, extra=("--isolate", "--budget", "2", "--min-steps", "1")),
                {"AR_EVAL_CHUNK_BYTES": str(3 * 64 * 512 * 4)}, timeout=300)
        e = last_entry(ledger)
        ev = (e["trusted"].get("evals") or [{}])[0]
        log = open(e["log_path"]).read()
        loc = [l.split() for l in log.splitlines() if l.startswith("LOCAL ")]
        if causal:
            same = bool(loc) and abs(float(loc[0][1]) - float(loc[0][3])) < 1e-9 and abs(float(loc[0][3]) - e["val_bpb"]) < 1e-9
            check("gpu: judge's val_bpb (CUDA IPC, 3-row chunks) == the same quantity computed inside train.py",
                  e["status"] == "ok" and ev.get("transport") == "ipc" and same and e["val_bpb"] > 5,
                  (e["status"], e["flags"], ev.get("transport"), loc, e["val_bpb"], ev.get("eval_error"), log[-1500:]))
            check("gpu: causal model passes the probe exactly (causal_max_abs 0, future_max_abs > 0)",
                  ev.get("causal_max_abs") == 0.0 and ev.get("future_max_abs", 0) > 0 and "non_causal" not in e["flags"],
                  {k: ev.get(k) for k in ("causal_max_abs", "future_max_abs", "probe_t", "check_error")})
        else:
            check("gpu: a model that peeks at later tokens is caught (non_causal, invalid)",
                  "non_causal" in e["flags"] and p.stdout.startswith("status: invalid")
                  and ev.get("causal_max_abs", 0) > 1e-3, (e["flags"], ev.get("causal_max_abs"), p.stdout[:200]))


def last_entry(ledger):
    with open(ledger) as f:
        return json.loads(f.read().strip().splitlines()[-1])


FAKE_PREPARE_CUDA = '''"""Stand-in for autoresearch's prepare.py on the GPU (tests): prepare's interface and accounting
(every token = 1 byte), a generated token stream instead of the data shards."""
import math
import os
import torch
MAX_SEQ_LEN = 64
TIME_BUDGET = 2
VOCAB_SIZE = 512
EVAL_TOKENS = 6 * 8 * MAX_SEQ_LEN
DATA_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "data")
VAL_FILENAME = "shard_06542.parquet"


class Tokenizer:
    @classmethod
    def from_directory(cls):
        return cls()

    def get_vocab_size(self):
        return VOCAB_SIZE


def _document_batches(split, tokenizer_batch_size=128):
    while True:
        yield [[1, 2, 3]] * tokenizer_batch_size, 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    g = torch.Generator().manual_seed(0 if split == "train" else 1)
    while True:
        x = torch.randint(0, VOCAB_SIZE, (B, T + 1), generator=g)
        yield x[:, :-1].contiguous().cuda(), x[:, 1:].contiguous().cuda(), 1


@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    total_nats, total_bytes = 0.0, 0
    for _ in range(EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)):
        x, y, _ = next(val_loader)
        total_nats += model(x, y, reduction="none").view(-1).sum().item()
        total_bytes += y.numel()
    return total_nats / (math.log(2) * total_bytes)
'''

TOY_CUDA = '''import math
import torch
import torch.nn.functional as F
from prepare import MAX_SEQ_LEN, VOCAB_SIZE, EVAL_TOKENS, make_dataloader, evaluate_bpb
CAUSAL = {causal}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(VOCAB_SIZE, 32)
        self.out = torch.nn.Linear(32, VOCAB_SIZE)

    def forward(self, idx, targets=None, reduction="mean"):
        h = self.emb(idx)
        n = torch.arange(1, idx.size(1) + 1, device=idx.device).view(1, -1, 1)
        h = h.cumsum(1) / n if CAUSAL else h + h.flip(1)       # causal prefix mean / peeks at the future
        logits = self.out(h).float()
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), reduction=reduction)


torch.manual_seed(0)
model = Model().cuda()
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
loader = make_dataloader(None, 8, MAX_SEQ_LEN, "train")
for step in range(30):
    x, y, _ = next(loader)
    model(x, y).backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
model.eval()
val = evaluate_bpb(model, None, 8)
g = torch.Generator().manual_seed(1)       # the stand-in's val stream, regenerated here: the same quantity locally
nats, nbytes = 0.0, 0
with torch.no_grad():
    for _ in range(EVAL_TOKENS // (8 * MAX_SEQ_LEN)):
        t = torch.randint(0, VOCAB_SIZE, (8, MAX_SEQ_LEN + 1), generator=g)
        x, y = t[:, :-1].cuda(), t[:, 1:].cuda()
        nats += F.cross_entropy(model(x).view(-1, VOCAB_SIZE), y.reshape(-1), reduction="none").sum().item()
        nbytes += y.numel()
print(f"LOCAL {{nats / (math.log(2) * nbytes):.10f}} TRUSTED {{val:.10f}}")
print("---")
print(f"val_bpb:          {{val:.6f}}")
print("training_seconds: 1.0")
print("peak_vram_mb:     10.0")
print("num_steps:        30")
'''

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""), flush=True)


def run(cmd, env_extra=None, timeout=200):
    env = {"HOME": os.environ["HOME"], "PATH": "/usr/bin:/bin"}   # deliberately minimal
    env.update(env_extra or {})
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)


def bootstrap_procs(repo):
    """pids of trusted-bootstrap processes (the sandboxed training) working on `repo`."""
    out = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if any(a.endswith(b"ar_bootstrap.py") for a in argv) and repo.encode() in argv:
            out.append(int(pid))
    return out


def unit_tests():
    tt = ar_run.trusted_train_time
    e = {"t_to_eval": 330.0, "t_step_anchor": 25.0, "t_train2": 3.0}
    check("unit: anchor rule preferred (t_to_eval - anchor, limit budget+10)",
          tt(e, 300.0) == (305.0, "step_anchor", 310.0), tt(e, 300.0))
    check("unit: fallback to the 2nd-batch rule (limit budget+20)",
          tt({"t_to_eval": 330.0, "t_step_anchor": None, "t_train2": 3.0}, 300.0) == (327.0, "train_batch2", 320.0))
    check("unit: no anchor at all -> t_to_eval rule (limit budget+300)",
          tt({"t_to_eval": 330.0, "t_train2": None}, 300.0) == (None, "t_to_eval", 600.0))
    m = {"val_bpb": 0.5, "training_seconds": 300.4, "total_seconds": 325.9, "peak_vram_mb": 45060.2,
         "mfu_percent": 39.8, "total_tokens_M": 499.6, "num_steps": 953, "num_params_M": 50.3, "depth": 8,
         "t_train_trusted": 300.9}
    want = ("---\nval_bpb:          0.997900\ntraining_seconds: 300.9\ntotal_seconds:    325.9\n"
            "peak_vram_mb:     45060.2\nmfu_percent:      39.80\ntotal_tokens_M:   499.6\nnum_steps:        953\n"
            "num_params_M:     50.3\ndepth:            8")
    check("unit: full summary = upstream block, trusted val_bpb + training time", ar_run.format_summary(m, 0.9979) == want,
          ar_run.format_summary(m, 0.9979))
    m2 = {"peak_vram_mb": 12.5, "num_steps": 40, "training_seconds": 2.0}
    check("unit: full summary omits unavailable lines, falls back to train.py's training_seconds",
          ar_run.format_summary(m2, 1.0) == "---\nval_bpb:          1.000000\ntraining_seconds: 2.0\n"
                                            "peak_vram_mb:     12.5\nnum_steps:        40", ar_run.format_summary(m2, 1.0))
    tp = ar_run.timing_problems
    # GPU calibration baseline (data/h2h/calibration, base arm) + the warm-up fields of the new record
    honest = {"t_to_eval": 328.6, "t_step_anchor": 27.6, "t_step1": 21.5, "t_train2": 17.3, "opt_steps": 506,
              "train_batches": 1013, "train_batches_at_anchor": 23, "train_batches_at_step1": 3}
    t, _, lim = tt(honest, 300.0)
    check("unit: honest baseline -> no timing problems", not tp(honest, 300.0, t, lim), tp(honest, 300.0, t, lim))
    # review finding: accum x40 in steps 0..10 (uncounted by train.py) -> anchor at ~290 s, eval at ~592 s
    exploit = {"t_to_eval": 592.0, "t_step_anchor": 290.0, "t_step1": 30.0, "t_train2": 17.3, "opt_steps": 514,
               "train_batches": 881 + 1006, "train_batches_at_anchor": 881, "train_batches_at_step1": 81}
    t, _, lim = tt(exploit, 300.0)
    probs = tp(exploit, 300.0, t, lim)
    check("unit: x40 warm-up exploit -> warm-up time, batch and step-1 problems (old rules: valid at 302 s)",
          t <= lim and len(probs) == 3 and "warm-up steps" in probs[0], probs)
    check("unit: outer bound on the judge's clock (t_to_eval > budget + 300) even with a late anchor",
          any("evaluation started" in p for p in tp({"t_to_eval": 640.0, "t_step_anchor": 335.0}, 300.0, 305.0, 310.0)))
    # review finding: no torch optimizer, 1.5 s steps (accum 2): old fallback limit 320 s flagged it at 320.75 s
    slow = {"t_to_eval": 340.75, "t_step_anchor": None, "t_train2": 20.0, "train_batches": 401}
    t, src, lim = tt(slow, 300.0, 200)
    check("unit: no-optimizer fallback scales its grace with the warm-up steps (1.5 s steps not flagged)",
          src == "train_batch2" and t == 320.75 and 335 < lim < 340, (t, src, lim))
    t, src, lim = tt(slow, 300.0, 12)
    check("unit: fallback warm-up estimate is capped (train.py's num_steps is untrusted)", lim < 320 + 11 * 16 + 1, lim)


def main():
    global GPU
    if "--unit" in sys.argv[1:]:
        unit_tests()
        n_fail = sum(1 for _, ok in RESULTS if not ok)
        print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
        return 1 if n_fail else 0
    GPU = GPU or config_gpu()
    if not GPU:
        print("no GPU UUID: set AR_TEST_GPU or agent_gpu_uuid in config.json (ar_run refuses runs without a real GPU)")
        return 2
    if not os.path.isdir(VENV):
        print(f"no autoresearch venv at {VENV} (set AR_TEST_REPO to an autoresearch checkout after `uv sync`)")
        return 2
    os.makedirs(TMP_BASE, exist_ok=True)
    root = tempfile.mkdtemp(prefix="ar_test_", dir=TMP_BASE)
    if os.path.realpath(root).startswith("/tmp/"):
        print("the scratch dir must not be under /tmp (set AR_TEST_TMP)")
        return 2
    ledger = os.path.join(root, "ledger.jsonl")
    lockdir = os.path.join(root, "locks")
    try:
        unit_tests()

        # ---- ok: stdout is exactly the two grep-able lines; ledger fields complete ----
        repo = make_repo(root, "ok")
        p = run(ar_cmd(repo, ledger, lockdir), {"AR_ATTEMPT_ID": "t-ok"})
        e = last_entry(ledger)
        check("ok: exit 0", p.returncode == 0, p.stdout + p.stderr)
        check("ok: stdout is exactly val_bpb + peak_vram_mb",
              p.stdout == "val_bpb:          1.081234\npeak_vram_mb:     45012.5\n", repr(p.stdout))
        req = ["run_id", "attempt_id", "ts", "wall_s", "head", "train_sha256", "parent_train_sha256",
               "prepare_sha256", "desc", "status", "returncode", "val_bpb", "metrics", "error_tail", "flags",
               "diff_path", "log_path", "agent_head"]
        check("ok: ledger has all required fields", all(k in e for k in req), [k for k in req if k not in e])
        check("ok: status/val/metrics/attempt", e["status"] == "ok" and e["val_bpb"] == 1.081234
              and e["metrics"]["num_steps"] == 506 and e["attempt_id"] == "t-ok" and e["error_tail"] is None,
              (e["status"], e["val_bpb"], e["flags"]))
        check("ok: diff + train.py snapshot written", os.path.getsize(e["diff_path"]) > 0
              and open(e["train_copy_path"]).read() == FAKES["ok"])
        # gpu_shared* only report other processes on the real GPU (e.g. a concurrent calibration run).
        check("ok: no integrity flags", not set(e["flags"]) - {"gpu_shared_at_start", "gpu_shared"}, e["flags"])
        check("ok: one ledger line per invocation", len(open(ledger).read().splitlines()) == 1)
        check("ok: no agent_head without AR_AGENT_HEAD", e["agent_head"] is None, e["agent_head"])

        # ---- prepare.py hash mismatch -> flag ----
        p = run(ar_cmd(repo, ledger, lockdir, prepare_sha="0" * 64))
        check("prepare sha mismatch: flagged + reeval", {"prepare_py_modified", "reeval"} <= set(last_entry(ledger)["flags"]),
              last_entry(ledger)["flags"])

        # ---- runner workspace (detached HEAD = parent commit): an unchanged submission is compared with
        #      HEAD itself (empty diff), not with HEAD^ ----
        subprocess.run(["git", "checkout", "-q", "--detach", "HEAD"], cwd=repo, check=True)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
        p = run(ar_cmd(repo, ledger, lockdir), {"AR_AGENT_HEAD": "ABCdef0123" + "0" * 30})
        e = last_entry(ledger)
        check("detached no-op: parent_rev=HEAD, empty_diff + reeval, delta 0",
              e["parent_rev"] == head and e["diff_lines"] == 0 and {"empty_diff", "reeval"} <= set(e["flags"])
              and e["delta_vs_parent"] == 0.0, (e["parent_rev"], head, e["diff_lines"], e["flags"]))
        check("agent_head: valid 40-hex head recorded (lower-cased)", e["agent_head"] == "abcdef0123" + "0" * 30,
              e["agent_head"])
        for bad in ("xyz", "a" * 41, "abc def", "../../etc"):
            run(ar_cmd(repo, ledger, lockdir, extra=("--timeout", "1")), {"AR_AGENT_HEAD": bad})
            if last_entry(ledger)["agent_head"] is not None:
                break
        check("agent_head: malformed values are dropped (null)", last_entry(ledger)["agent_head"] is None,
              last_entry(ledger)["agent_head"])

        # ---- failure statuses, sanitized tails ----
        for kind, status, needle in (("syntax", "crash", "SyntaxError"), ("fail_loss", "fail_loss", "last progress"),
                                     ("oom", "oom", "CUDA out of memory"), ("big_tail", "crash", "ValueError: boom")):
            repo = make_repo(root, kind)
            p = run(ar_cmd(repo, ledger, lockdir, extra=("--full-summary",)))
            e = last_entry(ledger)
            out_ok = p.stdout.startswith(f"status: {status}\n") and "val_bpb:" not in p.stdout and needle in p.stdout
            check(f"{kind}: status={status}, exit 1, tail has '{needle}' (also with --full-summary)",
                  p.returncode == 1 and e["status"] == status and out_ok, p.stdout[:300])
            check(f"{kind}: error_tail <= 4096 B, no '\\r', no progress records",
                  len(e["error_tail"].encode()) <= 4096 and "\r" not in e["error_tail"] and "| lrm:" not in e["error_tail"]
                  and len(p.stdout.encode()) <= 4096 + 64, len(e["error_tail"].encode()))

        # ---- per-attempt cap ----
        repo = os.path.join(root, "repo_syntax")
        for i in range(3):
            p = run(ar_cmd(repo, ledger, lockdir), {"AR_ATTEMPT_ID": "t-cap", "AR_MAX_RUNS": "2"})
        e = last_entry(ledger)
        check("cap: 3rd run refused (exit 3, status refused, flag run_cap)",
              p.returncode == 3 and e["status"] == "refused" and "run_cap" in e["flags"]
              and p.stdout.startswith("status: refused\nRun cap reached"), p.stdout)
        n = sum(1 for l in open(ledger) if json.loads(l)["attempt_id"] == "t-cap")
        check("cap: refused attempt still wrote exactly one line (3 lines total)", n == 3, n)
        p = run(ar_cmd(repo, ledger, lockdir), {"AR_ATTEMPT_ID": "t-cap", "AR_MAX_RUNS": "0"})
        check("cap: AR_MAX_RUNS=0 disables the cap", last_entry(ledger)["status"] == "crash")

        # ---- tamper refusal ----
        repo = make_repo(root, "tamper")
        p = run(ar_cmd(repo, ledger, lockdir))
        e = last_entry(ledger)
        check("tamper: refused, flag gpu_tamper", p.returncode == 3 and e["status"] == "refused"
              and "gpu_tamper" in e["flags"], p.stdout)

        # ---- trusted training time (--budget 2) ----
        repo = make_repo(root, "anchor_ok")
        p = run(ar_cmd(repo, ledger, lockdir, extra=("--budget", "2", "--min-steps", "1", "--full-summary")))
        e = last_entry(ledger)
        m = e["metrics"]
        check("anchor: hook fires for a MuonAdamW-like Optimizer subclass; t_train = eval - anchor ~ budget",
              e["status"] == "ok" and m.get("t_train_source") == "step_anchor"
              and 1.9 <= m.get("t_train_trusted", 0) <= 2.6 and m.get("t_step_anchor", 0) >= 1.0
              and "over_time_budget" not in e["flags"], (e["status"], e["flags"], m))
        tr = (e["trusted"].get("evals") or [{}])[0]
        check("anchor: signed record has t_step_anchor at the 11th step, opt_steps = train.py's step count",
              tr.get("opt_steps") == m.get("num_steps") and tr.get("n_optimizers") == 1
              and (tr.get("opt_anchors") or [[None]])[0][0] == "MuonAdamWLike" and tr.get("train_batches_at_anchor") == 12, tr)
        lines = p.stdout.splitlines()
        check("full summary: upstream block, trusted values, grep-able",
              p.returncode == 0 and lines[0] == "---" and lines[1] == "val_bpb:          1.081234"
              and lines[2] == f"training_seconds: {m['t_train_trusted']:.1f}" and "peak_vram_mb:     12.5" in lines
              and f"num_steps:        {m['num_steps']}" in lines and lines[-1] == "depth:            1", p.stdout)

        repo = make_repo(root, "anchor_adamw")
        p = run(ar_cmd(repo, ledger, lockdir, extra=("--budget", "2", "--min-steps", "1")))
        m = last_entry(ledger)["metrics"]
        check("anchor: hook fires for torch.optim.AdamW", m.get("t_train_source") == "step_anchor"
              and 1.9 <= m.get("t_train_trusted", 0) <= 2.6, m)

        repo = make_repo(root, "anchor_over")
        p = run(ar_cmd(repo, ledger, lockdir, extra=("--budget", "2", "--min-steps", "1")))
        e = last_entry(ledger)
        m = e["metrics"]
        check("anchor: 13 s of training (printed 2.0 s, time.monotonic rebound) -> over_time_budget, invalid",
              "over_time_budget" in e["flags"] and p.stdout.startswith("status: invalid")
              and 12.5 <= m.get("t_train_trusted", 0) <= 14.5 and m.get("training_seconds") == 2.0,
              (e["flags"], m, p.stdout[:200]))

        repo = make_repo(root, "prefetch")
        p = run(ar_cmd(repo, ledger, lockdir, extra=("--budget", "2", "--min-steps", "1")))
        e = last_entry(ledger)
        m = e["metrics"]
        check("anchor: prefetch + 23 s compile is not counted (2nd-batch rule would give > budget + 20)",
              e["status"] == "ok" and "over_time_budget" not in e["flags"] and m.get("t_train_trusted", 99) <= 2.6
              and m.get("t_train_batch2", 0) > 22, (e["flags"], m))

        repo = make_repo(root, "no_opt")
        p = run(ar_cmd(repo, ledger, lockdir, extra=("--budget", "2", "--min-steps", "1")))
        e = last_entry(ledger)
        m = e["metrics"]
        check("fallback: no torch optimizer -> 2nd-batch rule, not flagged",
              e["status"] == "ok" and m.get("t_train_source") == "train_batch2" and "t_step_anchor" not in m
              and "over_time_budget" not in e["flags"] and m.get("opt_steps") == 0, (e["flags"], m))

        # ---- timeout: the sandboxed training ignores SIGTERM (PID 1 of its namespace) ----
        repo = make_repo(root, "sleep")
        t = time.time()
        p = run(ar_cmd(repo, ledger, lockdir, extra=("--timeout", "4")))
        e = last_entry(ledger)
        check("timeout: status timeout, killed near the limit", e["status"] == "timeout" and p.returncode == 1
              and 4 <= e["wall_s"] < 15 and "exceeded the 4s" in p.stdout, (e["status"], e["wall_s"], time.time() - t))
        check("timeout: nothing of the run survives", not bootstrap_procs(repo), bootstrap_procs(repo))

        # ---- GPU lock: a 2nd concurrent run is refused; the lock is held by the training tree ----
        bg = subprocess.Popen(ar_cmd(repo, ledger, lockdir, extra=("--timeout", "8")), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, env={"HOME": os.environ["HOME"], "PATH": "/usr/bin:/bin"})
        time.sleep(2.5)
        p = run(ar_cmd(repo, ledger, lockdir))
        e = last_entry(ledger)
        check("lock: concurrent run refused (gpu_locked)", p.returncode == 3 and "gpu_locked" in e["flags"], p.stdout)
        bg.wait(timeout=60)

        # ---- interrupt: SIGTERM to the wrapper kills the run and still writes the ledger line ----
        bg = subprocess.Popen(ar_cmd(repo, ledger, lockdir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              env={"HOME": os.environ["HOME"], "PATH": "/usr/bin:/bin"})
        time.sleep(3)
        bg.send_signal(signal.SIGTERM)
        out, _ = bg.communicate(timeout=60)
        e = last_entry(ledger)
        check("interrupt: crash + flag interrupted, training killed",
              e["status"] == "crash" and "interrupted" in e["flags"] and bg.returncode == 1, (e["status"], e["flags"], out[:200]))
        check("interrupt: no leftover sandboxed training", not bootstrap_procs(repo), bootstrap_procs(repo))

        # ---- runner kill pattern: TERM, then SIGKILL a few seconds later. With the default 5 s poll the
        #      ledger line must still be written well inside that grace period. ----
        cmd = ar_cmd(repo, ledger, lockdir)
        i = cmd.index("--poll")
        del cmd[i:i + 2]
        n0 = len(open(ledger).read().splitlines())
        bg = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              env={"HOME": os.environ["HOME"], "PATH": "/usr/bin:/bin"})
        time.sleep(3)
        t = time.time()
        bg.send_signal(signal.SIGTERM)
        bg.communicate(timeout=60)
        dt = time.time() - t
        check("interrupt (poll 5 s): ledger line written < 2 s after TERM",
              dt < 2 and len(open(ledger).read().splitlines()) == n0 + 1 and "interrupted" in last_entry(ledger)["flags"],
              round(dt, 2))

        # ---- wrapper SIGKILLed mid-run: training must die too (PDEATHSIG -> timeout(1) -> its kill-after KILLs
        #      the process group) and release the GPU lock; no ledger line is written for it ----
        bg = subprocess.Popen(ar_cmd(repo, ledger, lockdir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              env={"HOME": os.environ["HOME"], "PATH": "/usr/bin:/bin"})
        time.sleep(3)
        bg.kill()
        bg.wait()
        time.sleep(ar_run.SANDBOX_KILL_AFTER + 2.5)
        orphans = bootstrap_procs(repo)
        p = run(ar_cmd(repo, ledger, lockdir, extra=("--timeout", "2")))
        check("wrapper SIGKILL: no orphaned training, GPU lock released",
              not orphans and last_entry(ledger)["status"] == "timeout", (orphans, p.stdout[:120]))
        if orphans:
            subprocess.run(["sudo", "-n", "kill", "-KILL", *map(str, orphans)])

        # ---- GPU violation: nvidia-smi (mocked) reports our training process on another GPU ----
        driver = f"""
import os, sys
sys.path.insert(0, {TOOLS!r})
import ar_run
def fake_gpu_apps():
    apps = []
    for pid in filter(str.isdigit, os.listdir('/proc')):
        try:
            argv = open(f'/proc/{{pid}}/cmdline', 'rb').read().split(b'\\0')
        except OSError:
            continue
        if any(a.endswith(b'ar_bootstrap.py') for a in argv) and {repo.encode()!r} in argv:
            apps.append((int(pid), {OTHER_GPU!r}, 500))
    return apps
ar_run.gpu_apps = fake_gpu_apps
sys.argv = ['ar_run.py'] + {ar_cmd(repo, ledger, lockdir)[3:]!r}
sys.exit(ar_run.main())
"""
        p = run([PY, "-I", "-c", driver])
        e = last_entry(ledger)
        check("violation: killed, status crash, flag gpu_violation",
              e["status"] == "crash" and "gpu_violation" in e["flags"] and e["wall_s"] < 30
              and "GPU other than the one assigned" in p.stdout, (e["status"], e["flags"], e["wall_s"], p.stdout[-200:]))
        check("violation: nothing of the run survives", not bootstrap_procs(repo), bootstrap_procs(repo))

        # ---- the reviewers' forging attack (walk up to the bootstrap's signer) in the default sandbox ----
        repo = make_repo(root, "forge")
        p = run(ar_cmd(repo, ledger, lockdir))
        e = last_entry(ledger)
        log = open(e["log_path"]).read()
        check("forge (default sandbox): no key/signer in train.py's process, judge's /proc closed; trusted value",
              "FORGE found []" in log and e["status"] == "ok" and e["val_bpb"] == 1.081234
              and e["trusted"]["bad_lines"] == 0, [l for l in log.splitlines() if "FORGE" in l] + [e["status"]])

        # ---- --isolate (the h2h runner's mode) ----
        iso = ("--isolate",)
        N_REPO = os.path.join(HOME, ar_run.IN_REPO)   # where the repo appears inside the --isolate sandbox
        repo = make_repo(root, "ok")
        p = run(ar_cmd(repo, ledger, lockdir, extra=iso))
        e = last_entry(ledger)
        check("isolate ok: exit 0, trusted val_bpb from the judge, no integrity flags",
              p.returncode == 0 and e["status"] == "ok" and e["val_bpb"] == 1.081234 and e["trusted"]["evals"][0]["judge"]
              and not set(e["flags"]) - {"gpu_shared_at_start", "gpu_shared", "reeval", "empty_diff"}, (p.stdout, e["flags"]))
        repo = make_repo(root, "anchor_ok")
        p = run(ar_cmd(repo, ledger, lockdir, extra=iso + ("--budget", "2", "--min-steps", "1", "--full-summary")))
        e = last_entry(ledger)
        m = e["metrics"]
        check("isolate anchor: t_train = eval - anchor ~ budget; warm-up counts recorded; not flagged",
              e["status"] == "ok" and m.get("t_train_source") == "step_anchor" and 1.9 <= m.get("t_train_trusted", 0) <= 2.8
              and m.get("train_batches_at_step1") == 2 and m.get("train_batches_at_anchor") == 12
              and "over_time_budget" not in e["flags"] and p.stdout.startswith("---\nval_bpb:          1.081234"),
              (e["flags"], m, p.stdout[:120]))
        for kind, needle in (("warm_batches", "train batches before"), ("warm_slow", "warm-up steps 2-11 took")):
            repo = make_repo(root, kind)
            p = run(ar_cmd(repo, ledger, lockdir, extra=iso + ("--budget", "2", "--min-steps", "1")), timeout=300)
            e = last_entry(ledger)
            check(f"isolate {kind}: hidden training in the uncounted warm-up steps -> invalid (over_time_budget)",
                  "over_time_budget" in e["flags"] and p.stdout.startswith("status: invalid") and needle in p.stdout
                  and e["metrics"].get("t_train_trusted", 99) <= 2.8, (e["flags"], e["metrics"].get("timing_problems"),
                                                                       p.stdout[:400]))
        hidden = [root, ROOT, os.path.join(ROOT, "data"), os.path.join(ROOT, "data", "h2h"),
                  os.path.join(ROOT, "canon.git"), os.path.join(HOME, ".claude"), os.path.join(HOME, "envs"), MAIN_REPO]
        repo = make_repo(root, "snoop", SNOOP.format(hidden=hidden))
        p = run(ar_cmd(repo, ledger, lockdir, extra=iso))
        e = last_entry(ledger)
        out = p.stdout
        check("isolate snoop: run data, canon.git, other repos, ~/.claude hidden; val shard unreadable; neutral cwd",
              e["status"] == "crash" and out.count(" absent") == len(hidden) and "VISIBLE" not in out
              and "SNOOP val error" in out and f"SNOOP cwd {N_REPO}" in out, out[-2000:])
        check("isolate snoop: the val shard access is reported to the judge (val_leak)", e["trusted"]["val_leak"],
              e["trusted"])
        frames = [l for l in out.splitlines() if l.lstrip().startswith('File "')]
        check("isolate: traceback frames name only neutral paths (~/autoresearch/train.py, ~/.ar/)",
              any(f'"{N_REPO}/train.py"' in l for l in frames)
              and all(f'"{N_REPO}/' in l or f'"{HOME}/.ar/' in l for l in frames), frames)
        repo = make_repo(root, "forge")
        p = run(ar_cmd(repo, ledger, lockdir, extra=iso))
        e = last_entry(ledger)
        log = open(e["log_path"]).read()
        check("isolate forge: nothing to find, judge untouchable; trusted value recorded",
              "FORGE found []" in log and e["status"] == "ok" and e["val_bpb"] == 1.081234,
              [l for l in log.splitlines() if "FORGE" in l] + [e["status"]])
        repo = make_repo(root, "persist")
        outs = [run(ar_cmd(repo, ledger, lockdir, extra=iso)).stdout for _ in range(2)]
        cache = os.path.join(root, "cache")
        check("isolate persist: a file written to the compile cache reaches neither the next run nor the host cache",
              all("PERSIST before absent" in o and "PERSIST wrote True" in o for o in outs)
              and not any("carry.pt" in fs for _, _, fs in os.walk(cache)), [o[-300:] for o in outs])
        repo = make_repo(root, "sleep")
        p = run(ar_cmd(repo, ledger, lockdir, extra=iso + ("--timeout", "4")))
        e = last_entry(ledger)
        check("isolate timeout: killed near the limit, nothing survives",
              e["status"] == "timeout" and 4 <= e["wall_s"] < 15 and not bootstrap_procs(repo)
              and not bootstrap_procs(N_REPO), (e["status"], e["wall_s"]))
        check("isolate: no key file left in any trusted dir",
              not [d for d, _, fs in os.walk(root) if d.endswith("/trusted") and "key" in fs])
        if "--gpu" in sys.argv[1:]:
            gpu_tests(root, ledger, lockdir)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
