"""Trusted evaluation bootstrap for an (untrusted) autoresearch train.py: the metric is computed here, not by train.py.

    <run venv python> -I ar_bootstrap.py <workdir> <trusted_dir | -> [--key-fd N] [--record-fd N] [--val-fd N]

Two processes, both inside tools/train_sandbox.sh (one GPU, no network, read-only filesystem):

judge (this process, started by tools/ar_run.py): never imports or executes train.py's code and never unpickles
  anything the child sends. It holds the one-time HMAC key and the only handle on the signed record file, and makes
  itself non-dumpable (PR_SET_DUMPABLE 0) before the child exists: no ptrace, and its /proc/<pid>/{mem,fd,environ}
  are root-only, so the child cannot read its memory, key or files (as PID 1 of the sandbox it cannot be killed from
  inside either). All of its imports happen before the child starts.
child (`--child`, spawned by the judge; inherits only stdio and one socketpair end): replaces prepare.evaluate_bpb /
  make_dataloader (train.py imports them from prepare) and runs train.py via runpy. Its hooks report progress events,
  which the judge timestamps on receipt with its own clock.

Evaluation: train.py's evaluate_bpb(model, tokenizer, batch_size) call turns the child into a logits server. The
judge runs prepare's own (pristine) evaluate_bpb, with its own tokenizer, validation loader and targets, against a
proxy model: per batch it sends only the input tokens, the child returns model(x) through a GPU buffer the judge
allocated and exported (CUDA IPC; over the socket for CPU tensors in tests), and the judge computes the
cross-entropy. The child never sees validation targets, cannot choose the loss and cannot write a signed record.
Before that a causality probe on the same kind of request (an eval-sized batch whose suffixes after a random position
t are swapped with other validation rows): logits at positions <= t must not change.

Trusted training-time anchors (tools/ar_run.py: training time = t_to_eval - anchor, plus bounds on the warm-up):
  * t_step_anchor: a global optimizer step post-hook counts step() calls per optimizer object (any
    torch.optim.Optimizer subclass, e.g. train.py's MuonAdamW); the anchor is when the first optimizer completes its
    WARMUP_OPT_STEPS-th step. train.py at c7666de excludes steps 0..10 (`if step > 10`) from its own training time,
    so the anchor sits at the same boundary and an honest run measures ~300 s;
  * t_step1, train_batches_at_step1, train_batches_at_anchor: the warm-up phase itself (bounded by ar_run.py);
  * t_train2 (fallback when no optimizer step was seen): the 2nd train batch request;
  * t_to_eval: when evaluate_bpb was called, on the judge's clock since the judge started.
The hooks run in the child: code that deliberately tampers with them could misreport the anchors (not the metric);
ar_run.py also bounds t_to_eval, which only the judge measures.

Validation data: with --val-fd (tools/ar_run.py --isolate) the validation shard is masked inside the sandbox and only
the judge can read it (through the fd); the child gets a clear error for the val split. Without it, accesses outside
the evaluation are reported (flag val_leak) as before.

Records: HMAC-signed JSON lines (start, val_leak, eval, end) appended to <trusted_dir>/record.jsonl (or --record-fd).
The key comes from <trusted_dir>/key, read and deleted before the child starts (or from --key-fd). A fake summary
block printed on stdout has no effect on the recorded metric.
"""
import ctypes
import hashlib
import hmac
import json
import os
import runpy
import secrets
import socket
import struct
import subprocess
import sys
import time
import weakref

_now = time.monotonic   # bound here: later rebinding of time.monotonic does not affect the trusted clock
T0 = _now()
WARMUP_OPT_STEPS = 11   # anchor = completion of the 11th optimizer step (train.py counts time for step > 10)
PROBE_ROWS = 2          # rows of the probe batch whose logits are compared
# logits travel in row chunks of at most this many bytes (the judge's shared buffer); env override for tests
CHUNK_BYTES = int(os.environ.get("AR_EVAL_CHUNK_BYTES") or 1 << 30)
MAX_HEADER = 1 << 16
PR_SET_DUMPABLE = 4
TIMING_EVENTS = ("start", "train2", "step1", "anchor", "val_leak", "end")


class ProtocolError(Exception):
    pass


class ChildError(Exception):
    """The child reported that the model failed or returned something that is not logits."""


# ------------------------------------------------------------------------------------------ framing
def send_msg(sock, obj, payload=b""):
    head = json.dumps(obj, separators=(",", ":")).encode()
    sock.sendall(struct.pack(">II", len(head), len(payload)) + head)
    if payload:
        sock.sendall(payload)


def _recv_exact(sock, n):
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if not k:
            raise EOFError("the other process closed the channel")
        got += k
    return buf


def recv_msg(sock, max_payload=0):
    """(header dict, payload bytearray or None). Oversized or malformed messages raise ProtocolError."""
    hn, pn = struct.unpack(">II", _recv_exact(sock, 8))
    if hn > MAX_HEADER or pn > max_payload:
        raise ProtocolError(f"unexpected message size ({hn}, {pn})")
    try:
        head = json.loads(bytes(_recv_exact(sock, hn)))
    except ValueError as e:
        raise ProtocolError("malformed message") from e
    if not isinstance(head, dict):
        raise ProtocolError("malformed message")
    return head, (_recv_exact(sock, pn) if pn else None)


def _int(v, lo=0, hi=1 << 62):
    return v if isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi else None


def _float(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


# ---------------------------------------------------------------------------------------- the child
class Hooks:
    """Installed in the process that runs train.py: train-batch counting, per-optimizer step counts, validation-data
    access. Every event is passed to notify(name, **fields) when it happens."""

    def __init__(self, prepare, notify, val_masked):
        self.prepare, self.notify, self.val_masked = prepare, notify, val_masked
        self.orig_loader, self.orig_docs = prepare.make_dataloader, prepare._document_batches
        self.val_name = prepare.VAL_FILENAME
        self.train_batches = self.opt_steps = self.n_optimizers = self.n_anchors = self.n_leaks = 0
        self.opt_n = weakref.WeakKeyDictionary()   # optimizer -> completed step() calls (never keeps one alive)
        self.opt_n_by_id = {}                      # fallback for optimizers that are not weak-referenceable

    def counts(self):
        return {"train_batches": self.train_batches, "opt_steps": self.opt_steps, "n_optimizers": self.n_optimizers}

    def leak(self, what):
        if self.n_leaks < 20:
            self.n_leaks += 1
            self.notify("val_leak", what=str(what)[:300])
        if self.val_masked:
            raise RuntimeError("the validation split is reserved for the final evaluation (prepare.evaluate_bpb); "
                               "train.py may only use the 'train' split")

    def timed(self, gen):
        """Count train batch requests; the 2nd one happens after the first (compiling) forward/backward."""
        for item in gen:
            self.train_batches += 1
            if self.train_batches == 2:
                self.notify("train2")
            yield item

    def make_dataloader(self, tokenizer, B, T, split, *a, **k):
        if split == "val":
            self.leak("make_dataloader(val)")
        gen = self.orig_loader(tokenizer, B, T, split, *a, **k)
        return self.timed(gen) if split == "train" else gen

    def document_batches(self, split, *a, **k):
        if split == "val":
            self.leak("_document_batches(val)")
        return self.orig_docs(split, *a, **k)

    def audit(self, event, args):
        # file opens of the validation shard (python-level opens)
        if event in ("open", "os.open") and args and isinstance(args[0], (str, bytes, os.PathLike)):
            try:
                p = os.fsdecode(args[0])
            except TypeError:
                return
            if self.val_name in p:
                self.leak(f"open:{p}")

    def opt_step_post_hook(self, opt, args, kwargs):
        """Global post-hook of every torch.optim.Optimizer.step(): per-optimizer step counts, warm-up anchors."""
        try:
            n = self.opt_n.get(opt, 0) + 1
            self.opt_n[opt] = n
        except TypeError:
            n = self.opt_n_by_id.get(id(opt), 0) + 1
            self.opt_n_by_id[id(opt)] = n
        if n == 1:
            self.n_optimizers += 1
            if self.n_optimizers == 1:
                self.notify("step1", opt=type(opt).__name__[:80], batches=self.train_batches)
        if n > self.opt_steps:
            self.opt_steps = n
        if n == WARMUP_OPT_STEPS and self.n_anchors < 8:
            self.n_anchors += 1
            self.notify("anchor", opt=type(opt).__name__[:80], batches=self.train_batches)

    def install(self, torch):
        p = self.prepare
        p.make_dataloader, p._document_batches = self.make_dataloader, self.document_batches
        sys.addaudithook(self.audit)          # cannot be removed by later code
        try:  # pyarrow opens files natively (no audit event): also watch its python entry points
            import pyarrow.parquet as pq
            pf, rt = pq.ParquetFile, pq.read_table

            def _pf(source, *a, **k):
                if isinstance(source, (str, os.PathLike)) and self.val_name in os.fspath(source):
                    self.leak(f"ParquetFile:{os.fspath(source)}")
                return pf(source, *a, **k)

            def _rt(source, *a, **k):
                if isinstance(source, (str, os.PathLike)) and self.val_name in os.fspath(source):
                    self.leak(f"read_table:{os.fspath(source)}")
                return rt(source, *a, **k)
            pq.ParquetFile, pq.read_table = _pf, _rt
        except ImportError:
            pass
        try:
            from torch.optim.optimizer import register_optimizer_step_post_hook
        except ImportError:
            return False
        hook = self.opt_step_post_hook
        try:   # never traced into a graph if train.py compiles a function that calls optimizer.step()
            hook = torch.compiler.disable(hook)
        except AttributeError:
            pass
        register_optimizer_step_post_hook(hook)
        return True


def open_shared(torch, d):
    """The judge's exported logits buffer (CUDA IPC) as a flat float32 tensor."""
    st = torch.UntypedStorage._new_shared_cuda(
        int(d["device"]), bytes.fromhex(d["handle"]), int(d["size"]), int(d["offset"]), bytes.fromhex(d["rc"]),
        int(d["rco"]), bytes.fromhex(d["ev"]), bool(d["evs"]))
    return torch.empty(0, dtype=torch.float32, device=f"cuda:{int(d['device'])}").set_(st, 0, (int(d["size"]) // 4,),
                                                                                      (1,))


class LogitsServer:
    """The child's evaluate_bpb: serves model(x) logits to the judge until it sends the result."""

    def __init__(self, torch, chan, hooks):
        self.torch, self.chan, self.hooks, self.shared = torch, chan, hooks, None

    def _expect(self, cmd):
        head, _ = recv_msg(self.chan)
        if head.get("cmd") != cmd:
            raise RuntimeError(f"evaluation protocol error (expected {cmd!r}, got {head.get('cmd')!r})")
        return head

    def evaluate_bpb(self, model, tokenizer, batch_size):
        torch = self.torch
        mem = None
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            mem = torch.cuda.max_memory_allocated() / 2**20
            torch.cuda.empty_cache()          # room for the judge's buffers
        send_msg(self.chan, {"ev": "eval", "batch_size": int(batch_size), "max_mem_mb": mem, **self.hooks.counts()})
        while True:
            head, payload = recv_msg(self.chan, 1 << 31)
            cmd = head.get("cmd")
            if cmd == "fwd":
                self.forward(model, head, payload)
            elif cmd == "result":
                return float(head["val_bpb"])
            elif cmd == "error":
                raise RuntimeError(f"the trusted evaluation failed: {head.get('what')}")
            else:
                raise RuntimeError(f"evaluation protocol error (got {cmd!r})")

    def forward(self, model, head, payload):
        torch = self.torch
        r, T = int(head["rows"]), int(head["T"])
        x = torch.frombuffer(payload, dtype=torch.int64).view(r, T)
        if head.get("device") == "cuda":
            x = x.to("cuda")
        with torch.no_grad():
            try:
                out = model(x)
            except Exception as e:
                send_msg(self.chan, {"ev": "fwd_error", "what": repr(e)[:300]})
                raise
            if not (torch.is_tensor(out) and out.dim() == 3 and tuple(out.shape[:2]) == (r, T)):
                what = (f"model(idx) must return logits [B, T, V]; got {type(out).__name__} "
                        f"{tuple(out.shape) if torch.is_tensor(out) else ''}")
                send_msg(self.chan, {"ev": "bad_output", "what": what})
                raise RuntimeError(what)
            V = int(out.shape[-1])
            send_msg(self.chan, {"ev": "out", "V": V})
            go = self._expect("go")
            cr = int(go["chunk_rows"])
            ipc = go.get("transport") == "ipc"
            if go.get("ipc"):
                self.shared = open_shared(torch, go["ipc"])
            for i, s in enumerate(range(0, r, cr)):
                part = out[s:s + cr]
                n = int(part.shape[0])
                if ipc:
                    self.shared[: n * T * V].view(n, T, V).copy_(part)
                    torch.cuda.synchronize()
                    send_msg(self.chan, {"ev": "chunk", "i": i, "n": n})
                else:
                    send_msg(self.chan, {"ev": "chunk", "i": i, "n": n},
                             part.detach().float().contiguous().cpu().numpy().tobytes())
                self._expect("ack")
            del out


def child_main(workdir, fd, val_masked):
    chan = socket.socket(fileno=fd)

    def notify(name, **fields):
        try:
            send_msg(chan, {"ev": name, **fields})
        except OSError:       # the judge is gone: the sandbox is being torn down anyway
            pass

    os.chdir(workdir)
    sys.path.insert(0, workdir)
    sys.argv = ["train.py"]
    import torch
    if os.environ.get("AR_NO_AUTOTUNE") == "1":
        # Runtime autotuning picks kernels by timing -> different numerics on every fresh compile, which is
        # the dominant source of run-to-run val_bpb noise. Heuristic configs are deterministic.
        import torch._inductor.config as _ic
        _ic.triton.autotune_pointwise = False
        _ic.triton.autotune_cublasLt = False
        _ic.max_autotune = _ic.max_autotune_pointwise = _ic.max_autotune_gemm = False
        _ic.coordinate_descent_tuning = False
    import prepare  # pristine copy in the runner workspace
    hooks = Hooks(prepare, notify, val_masked)
    step_hook = hooks.install(torch)
    prepare.evaluate_bpb = LogitsServer(torch, chan, hooks).evaluate_bpb
    notify("start", step_hook=step_hook)
    try:
        runpy.run_path(os.path.join(workdir, "train.py"), run_name="__main__")
    finally:
        notify("end", **hooks.counts())


# ---------------------------------------------------------------------------------------- the judge
class _ValShard:
    """prepare.pq inside the judge: the validation shard is read from memory (the sandbox masks its path)."""

    def __init__(self, pq, name, data):
        self._pq, self._name, self._data = pq, name, data

    def __getattr__(self, k):
        return getattr(self._pq, k)

    def ParquetFile(self, source, *a, **k):
        if isinstance(source, (str, os.PathLike)) and os.path.basename(os.fspath(source)) == self._name:
            import pyarrow as pa
            source = pa.BufferReader(self._data)
        return self._pq.ParquetFile(source, *a, **k)


class Proxy:
    """The model as prepare.evaluate_bpb sees it in the judge: model(x, y, reduction) -> cross-entropy computed here
    from the logits the child returns for x."""

    def __init__(self, judge):
        self.j = judge

    def __call__(self, x, y=None, reduction="mean"):
        F, torch = self.j.F, self.j.torch
        losses = []
        for s, n, lg in self.j.logits(x):
            losses.append(F.cross_entropy(lg.reshape(-1, lg.size(-1)), y[s:s + n].reshape(-1), ignore_index=-1,
                                          reduction="none"))
        loss = torch.cat(losses)
        if reduction == "none":
            return loss
        if reduction == "sum":
            return loss.sum()
        return loss.sum() / (y.reshape(-1) != -1).sum()


class Judge:
    def __init__(self, workdir, key, rec, val_data):
        self.workdir, self.key, self.rec, self.val_data = workdir, key, rec, val_data
        self.chan = self.proc = self.buf = self.V = self.tok = self.transport = None
        self.n_eval = self.bad = 0
        self.step_hook = None
        self.t_train2 = self.t_step1 = self.t_step_anchor = None
        self.batches_at_step1 = self.batches_at_anchor = None
        self.opt_anchors, self.val_leak, self.counts = [], [], {}

    # ---- records
    def emit(self, d):
        payload = json.dumps(d, sort_keys=True)
        sig = hmac.new(self.key, payload.encode(), hashlib.sha256).hexdigest()
        self.rec.write(json.dumps({"payload": payload, "sig": sig}) + "\n")
        self.rec.flush()

    def timing(self):
        return {"t_step_anchor": self.t_step_anchor, "opt_anchors": list(self.opt_anchors), "t_step1": self.t_step1,
                "train_batches_at_step1": self.batches_at_step1, "train_batches_at_anchor": self.batches_at_anchor,
                "opt_steps": self.counts.get("opt_steps"), "n_optimizers": self.counts.get("n_optimizers"),
                "step_hook": self.step_hook}

    # ---- the child's events
    def on_event(self, head):
        """Progress events from the child, timestamped now. Returns False for anything that is not one."""
        ev, t = head.get("ev"), _now() - T0
        if ev == "start":
            self.step_hook = bool(head.get("step_hook"))
        elif ev == "train2":
            self.t_train2 = self.t_train2 if self.t_train2 is not None else t
        elif ev == "step1":
            if self.t_step1 is None:
                self.t_step1, self.batches_at_step1 = t, _int(head.get("batches"))
        elif ev == "anchor":
            if len(self.opt_anchors) < 8:
                self.opt_anchors.append([str(head.get("opt"))[:80], round(t, 3)])
            if self.t_step_anchor is None:
                self.t_step_anchor, self.batches_at_anchor = t, _int(head.get("batches"))
        elif ev == "val_leak":
            if len(self.val_leak) < 20:
                what = str(head.get("what"))[:300]
                self.val_leak.append(what)
                self.emit({"event": "val_leak", "what": what, "t": t})
        elif ev == "end":
            self.counts.update({k: _int(head.get(k)) for k in ("train_batches", "opt_steps", "n_optimizers")})
        else:
            return False
        return True

    def expect(self, ev, max_payload=0):
        """The next protocol message of kind `ev` (progress events in between are recorded)."""
        while True:
            head, payload = recv_msg(self.chan, max_payload)
            if head.get("ev") == ev:
                return head, payload
            if head.get("ev") in ("fwd_error", "bad_output"):
                raise ChildError(str(head.get("what"))[:300])
            if head.get("ev") == "end":
                self.on_event(head)
                raise ChildError("train.py stopped during the evaluation")
            if not self.on_event(head):
                raise ProtocolError(f"unexpected message {str(head.get('ev'))[:40]!r} during the evaluation")

    # ---- evaluation
    def export_buffer(self, n):
        """A float32 CUDA buffer of >= n elements in a CUDA allocation of its own (the child maps the whole allocation:
        nothing else of the judge may live in it), exported for CUDA IPC."""
        torch = self.torch
        # The caching allocator gives a request of > 10 MiB in whole 2 MiB pages a cudaMalloc of exactly that size
        # (smaller ones share 2/20 MiB segments); emptying the cache first rules out carving it from a cached block.
        unit = (2 << 20) // 4
        n = -(-max(n, (12 << 20) // 4) // unit) * unit
        torch.cuda.empty_cache()
        buf = torch.empty(n, dtype=torch.float32, device="cuda")
        segs = [s for s in torch.cuda.memory_snapshot() if any(b["address"] == buf.data_ptr() for b in s["blocks"])]
        if len(segs) != 1 or segs[0]["total_size"] != n * 4 or len(segs[0]["blocks"]) != 1:
            raise RuntimeError("could not give the shared logits buffer a CUDA allocation of its own")
        h = buf.untyped_storage()._share_cuda_()
        return buf, {"device": h[0], "handle": h[1].hex(), "size": h[2], "offset": h[3], "rc": h[4].hex(),
                     "rco": h[5], "ev": h[6].hex() if h[6] else "", "evs": bool(h[7])}

    def logits(self, x):
        """Yields (first row, rows, logits [rows, T, V] float32) chunks of model(x) computed by the child. The chunk is
        only valid until the next item is requested."""
        torch = self.torch
        r, T = (int(v) for v in x.shape)
        ipc = x.is_cuda
        self.transport = "ipc" if ipc else "socket"
        send_msg(self.chan, {"cmd": "fwd", "rows": r, "T": T, "device": "cuda" if ipc else "cpu"},
                 x.detach().to("cpu", torch.int64).contiguous().numpy().tobytes())
        head, _ = self.expect("out")
        V = _int(head.get("V"), 1, 1 << 22)
        if V is None:
            raise ChildError(f"bad logits dimension {head.get('V')!r}")
        if V < self.vocab:
            raise ChildError(f"model(idx) returned {V} logits per position; the vocabulary has {self.vocab} tokens")
        if self.V is not None and V != self.V:
            raise ChildError("model(idx) changed its logits dimension between calls")
        self.V = V
        cr = max(1, min(r, CHUNK_BYTES // (T * V * 4)))
        go = {"cmd": "go", "chunk_rows": cr, "transport": "ipc" if ipc else "socket"}
        if ipc and self.buf is None:
            self.buf, go["ipc"] = self.export_buffer(cr * T * V)   # kept until the judge exits (never reused)
        if ipc and self.buf.numel() < cr * T * V:
            raise ChildError("the evaluation batch shape changed")
        send_msg(self.chan, go)
        for i, s in enumerate(range(0, r, cr)):
            n = min(cr, r - s)
            head, payload = self.expect("chunk", 0 if ipc else n * T * V * 4)
            if head.get("i") != i or head.get("n") != n:
                raise ProtocolError("logits chunks out of order")
            if ipc:
                lg = self.buf[: n * T * V].view(n, T, V)
            else:
                if payload is None or len(payload) != n * T * V * 4:
                    raise ProtocolError("logits chunk of the wrong size")
                lg = torch.frombuffer(payload, dtype=torch.float32).view(n, T, V)
            yield s, n, lg
            if ipc:
                torch.cuda.synchronize()          # done reading the shared buffer before the child overwrites it
            send_msg(self.chan, {"cmd": "ack"})

    def probe(self, tokenizer, bs):
        """Causality probe: the same eval-sized request twice, the second with every row's suffix after a random
        position t replaced (by the next row's suffix: real text, not noise); logits at positions <= t must match."""
        torch, prepare = self.torch, self.prepare
        T = prepare.MAX_SEQ_LEN
        x, _, _ = next(prepare.make_dataloader(tokenizer, bs, T, "val"))
        x = x.clone()
        t = T // 4 + secrets.randbelow(max(1, T // 2))
        x2 = x.clone()
        if bs > 1:
            x2[:, t + 1:] = torch.roll(x, 1, 0)[:, t + 1:]
        else:
            x2[:, t + 1:] = (x2[:, t + 1:] + 1 + secrets.randbelow(self.vocab - 1)) % self.vocab
        P = min(PROBE_ROWS, bs)
        a, b = self.rows_of(x, P), self.rows_of(x2, P)
        return {"causal_max_abs": float((a[:, : t + 1] - b[:, : t + 1]).abs().max()),
                "future_max_abs": float((a[:, t + 1:] - b[:, t + 1:]).abs().max()) if t + 1 < T else 0.0,
                "probe_t": t}

    def rows_of(self, x, P):
        keep = []
        for s, n, lg in self.logits(x):
            if s < P:
                keep.append(lg[: min(n, P - s)].clone())
        return self.torch.cat(keep)

    def evaluate(self, head):
        t_eval = _now() - T0
        self.n_eval += 1
        self.counts.update({k: _int(head.get(k)) for k in ("train_batches", "opt_steps", "n_optimizers")})
        bs = _int(head.get("batch_size"), 1, 1 << 16)
        rec = {"event": "eval", "n_eval": self.n_eval, "t_to_eval": t_eval, "batch_size": bs, "judge": True,
               "val_leak": list(self.val_leak), "t_train2": self.t_train2,
               "train_batches": self.counts.get("train_batches"), "max_mem_mb": _float(head.get("max_mem_mb")),
               **self.timing()}
        try:
            if bs is None:
                raise ChildError(f"bad batch_size {head.get('batch_size')!r}")
            if self.tok is None:
                self.tok = self.prepare.Tokenizer.from_directory()
                self.vocab = int(self.tok.get_vocab_size())
            try:
                rec.update(self.probe(self.tok, bs))
            except ChildError as e:   # the model failed on the probe (the evaluation will fail the same way)
                rec["check_error"] = repr(e)[:300]
            rec["val_bpb"] = float(self.orig_eval(Proxy(self), self.tok, bs))
        except Exception as e:  # noqa: BLE001
            rec["eval_error"] = repr(e)[:300]
            self.emit(rec)
            self.reply({"cmd": "error", "what": str(e)[:300]})
            return
        rec["eval_s"] = _now() - T0 - t_eval
        rec["transport"] = self.transport
        self.emit(rec)
        self.reply({"cmd": "result", "val_bpb": rec["val_bpb"]})

    def reply(self, msg):
        try:
            send_msg(self.chan, msg)
        except OSError:
            pass

    # ---- main loop
    def run(self):
        child_env = dict(os.environ)
        for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):   # the shared buffer needs plain cudaMalloc
            os.environ.pop(k, None)
        os.environ["CUDA_CACHE_DISABLE"] = "1"    # never load JIT artifacts the child could have written
        os.chdir(self.workdir)
        sys.path.insert(0, self.workdir)
        import torch
        import torch.nn.functional as F
        import prepare  # pristine copy in the runner workspace (the child cannot write it)
        self.torch, self.F, self.prepare = torch, F, prepare
        self.orig_eval = prepare.evaluate_bpb
        if self.val_data is not None and hasattr(prepare, "pq"):
            prepare.pq = _ValShard(prepare.pq, prepare.VAL_FILENAME, self.val_data)
        self.emit({"event": "start", "judge": True, "no_autotune": os.environ.get("AR_NO_AUTOTUNE") == "1",
                   "warmup_opt_steps": WARMUP_OPT_STEPS, "val_masked": self.val_data is not None})
        mine, theirs = socket.socketpair()
        cmd = [sys.executable, "-I", os.path.abspath(__file__), "--child", self.workdir, str(theirs.fileno())]
        if self.val_data is not None:
            cmd.append("--val-masked")
        self.proc = subprocess.Popen(cmd, pass_fds=(theirs.fileno(),), close_fds=True, stdin=subprocess.DEVNULL,
                                     env=child_env)
        theirs.close()
        self.chan = mine
        try:
            while True:
                try:
                    head, _ = recv_msg(self.chan)
                except EOFError:
                    break
                except (ProtocolError, OSError):
                    self.bad += 1
                    break                   # a desynchronised channel: stop listening, wait for the child
                if head.get("ev") == "eval":
                    self.evaluate(head)
                elif not self.on_event(head):
                    self.bad += 1
        finally:
            rc = self.proc.wait()
            self.emit({"event": "end", "t": _now() - T0, "n_eval": self.n_eval, "child_rc": rc,
                       "protocol_errors": self.bad, **self.timing()})
        return rc if rc >= 0 else 128 - rc


def judge_main(argv):
    ctypes.CDLL(None, use_errno=True).prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)   # before any secret is read
    workdir, tdir = argv[0], argv[1]
    fds = {}
    rest = argv[2:]
    for i in range(0, len(rest) - 1, 2):
        if rest[i] in ("--key-fd", "--record-fd", "--val-fd"):
            fds[rest[i]] = int(rest[i + 1])
    if "--key-fd" in fds:
        with os.fdopen(fds["--key-fd"], "rb") as f:
            key = f.read().strip()
    else:
        with open(os.path.join(tdir, "key"), "rb") as f:
            key = f.read().strip()
        os.unlink(os.path.join(tdir, "key"))
    if not key:
        sys.exit("ar_bootstrap: empty key")
    rec = (os.fdopen(fds["--record-fd"], "a", buffering=1) if "--record-fd" in fds else
           open(os.path.join(tdir, "record.jsonl"), "a", buffering=1))
    val = None
    if "--val-fd" in fds:
        with os.fdopen(fds["--val-fd"], "rb") as f:
            val = f.read()
    return Judge(workdir, key, rec, val).run()


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child_main(sys.argv[2], int(sys.argv[3]), "--val-masked" in sys.argv[4:])
    else:
        sys.exit(judge_main(sys.argv[1:]))
