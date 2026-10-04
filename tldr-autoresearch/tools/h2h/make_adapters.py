#!/usr/bin/env python3
"""make_adapters.py - the two LoRA adapters the head-to-head (h2h) arms are served through.

  data/h2h/adapters/v5     frozen byte-for-byte copy of data/adapters/policy-v5 (the RLTL;DR v5 policy, unmerged)
  data/h2h/adapters/base0  same adapter_config.json; adapter_model.safetensors with exactly the same keys, shapes,
                           dtypes, offsets and metadata as v5 but all-zero data (A = B = 0), so the base arm runs the
                           identical vLLM LoRA kernel path while its outputs are exactly the base model's.

Stdlib only (safetensors = 8-byte little-endian header length + JSON header + raw tensor data; all-zero bytes are
0.0 in every float dtype), so it runs under any python:
  python3 tools/h2h/make_adapters.py [--check] [--force]

Idempotent: existing outputs that already match are left untouched (vLLM may have them loaded). A v5 copy that
differs from its source is an error unless --force (a frozen adapter must never change under a running
experiment). Outputs are written to a temp dir next to the target and renamed into place; files are made
read-only. A manifest with sha256 sums is written to data/h2h/adapters/manifest.json (outside the adapter dirs,
so vLLM never sees extra files).
"""
import argparse
import hashlib
import json
import logging
import os
import shutil
import struct
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rltldr.h2h_config import load_h2h_config  # noqa: E402

CONFIG = "adapter_config.json"
WEIGHTS = "adapter_model.safetensors"
FILES = (CONFIG, WEIGHTS)
CHUNK = 1 << 24
log = logging.getLogger("make_adapters")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                return h.hexdigest()
            h.update(b)


def read_header(path: str) -> tuple:
    """-> (raw header bytes incl. the 8-byte length prefix, parsed header dict, data length)."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        prefix = f.read(8)
        (n,) = struct.unpack("<Q", prefix)
        raw = f.read(n)
    hdr = json.loads(raw)
    data_len = size - 8 - n
    end = max((v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__"), default=0)
    if end != data_len:
        raise ValueError(f"{path}: tensor data ends at {end} but file holds {data_len} data bytes")
    return prefix + raw, hdr, data_len


def is_zero_twin(path: str, ref_path: str) -> bool:
    """True iff `path` has byte-identical safetensors header to `ref_path` and only zero bytes after it."""
    try:
        raw, _, data_len = read_header(path)
        ref_raw, _, ref_len = read_header(ref_path)
    except (OSError, ValueError, json.JSONDecodeError, struct.error):
        return False
    if raw != ref_raw or data_len != ref_len:
        return False
    zero = bytes(CHUNK)
    with open(path, "rb") as f:
        f.seek(len(raw))
        while True:
            b = f.read(CHUNK)
            if not b:
                return True
            if b != zero[:len(b)]:
                return False


def write_zero_twin(dst: str, ref_path: str) -> None:
    raw, _, data_len = read_header(ref_path)
    zero = bytes(CHUNK)
    with open(dst, "wb") as f:
        f.write(raw)
        left = data_len
        while left:
            n = min(left, CHUNK)
            f.write(zero[:n])
            left -= n
        f.flush()
        os.fsync(f.fileno())


def same_files(a: str, b: str) -> bool:
    return all(os.path.isfile(os.path.join(d, f)) for d in (a, b) for f in FILES) and \
        all(sha256(os.path.join(a, f)) == sha256(os.path.join(b, f)) for f in FILES)


def install(tmp: str, dst: str, force: bool) -> None:
    """Make files read-only, then atomically move the finished temp dir to dst (old one removed if forced)."""
    for f in os.listdir(tmp):
        os.chmod(os.path.join(tmp, f), 0o444)
    os.chmod(tmp, 0o755)
    if os.path.exists(dst):
        if not force:
            raise SystemExit(f"refusing to replace {dst} (differs from what it should be); use --force")
        old = f"{dst}.old-{int(time.time())}"
        os.rename(dst, old)
        os.rename(tmp, dst)
        shutil.rmtree(old)
    else:
        os.rename(tmp, dst)


def make_v5(src: str, dst: str, force: bool, check: bool) -> str:
    if os.path.isdir(dst) and same_files(src, dst):
        log.info("v5: %s already identical to %s", dst, src)
        return "ok"
    if check:
        return "missing" if not os.path.exists(dst) else "differs"
    tmp = tempfile.mkdtemp(prefix=".tmp-v5-", dir=os.path.dirname(dst))
    try:
        for f in FILES:
            shutil.copyfile(os.path.join(src, f), os.path.join(tmp, f))
        if not same_files(src, tmp):
            raise SystemExit("v5 copy does not match its source (source changed while copying?)")
        install(tmp, dst, force)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log.info("v5: copied %s -> %s", src, dst)
    return "written"


def make_base0(v5_dir: str, dst: str, check: bool) -> str:
    ref_cfg, ref_w = os.path.join(v5_dir, CONFIG), os.path.join(v5_dir, WEIGHTS)
    cfg_ok = os.path.isfile(os.path.join(dst, CONFIG)) and sha256(os.path.join(dst, CONFIG)) == sha256(ref_cfg)
    if cfg_ok and is_zero_twin(os.path.join(dst, WEIGHTS), ref_w):
        log.info("base0: %s already an all-zero twin of %s", dst, v5_dir)
        return "ok"
    if check:
        return "missing" if not os.path.exists(dst) else "differs"
    tmp = tempfile.mkdtemp(prefix=".tmp-base0-", dir=os.path.dirname(dst))
    try:
        shutil.copyfile(ref_cfg, os.path.join(tmp, CONFIG))
        write_zero_twin(os.path.join(tmp, WEIGHTS), ref_w)
        if not is_zero_twin(os.path.join(tmp, WEIGHTS), ref_w):
            raise SystemExit("freshly written base0 weights failed verification")
        install(tmp, dst, force=True)      # base0 is derived data: always safe to regenerate
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log.info("base0: wrote all-zero twin of %s -> %s", v5_dir, dst)
    return "written"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="only verify; exit 1 if anything is missing/differs")
    ap.add_argument("--force", action="store_true", help="replace a v5 copy that differs from its source")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_h2h_config()
    v5_dst = cfg.arm("v5").adapter_dir
    base0_dst = cfg.arm("base").adapter_dir
    if os.path.normpath(base0_dst) != os.path.normpath(cfg.zero_adapter):
        raise SystemExit(f"config mismatch: arm base adapter_dir {base0_dst} != zero_adapter {cfg.zero_adapter}")
    for d in (v5_dst, base0_dst):
        os.makedirs(os.path.dirname(d), exist_ok=True)
    src = cfg.v5_adapter_src
    if not all(os.path.isfile(os.path.join(src, f)) for f in FILES):
        raise SystemExit(f"v5 source adapter incomplete: {src}")
    res = {"v5": make_v5(src, v5_dst, a.force, a.check)}
    # base0 is built from the frozen copy when it exists (from the source in --check mode before the copy)
    res["base0"] = make_base0(v5_dst if res["v5"] in ("ok", "written") else src, base0_dst, a.check)
    if a.check:
        print(json.dumps(res))
        return 0 if all(v == "ok" for v in res.values()) else 1
    manifest = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "v5_source": src,
                "adapters": {name: {"dir": d, **{f: sha256(os.path.join(d, f)) for f in FILES}}
                             for name, d in (("v5", v5_dst), ("base0", base0_dst))}}
    mpath = os.path.join(os.path.dirname(v5_dst), "manifest.json")
    old = None
    if os.path.exists(mpath):
        with open(mpath) as f:
            old = json.load(f)
    if not old or old.get("adapters") != manifest["adapters"]:
        with open(mpath + ".tmp", "w") as f:
            json.dump(manifest, f, indent=1)
        os.replace(mpath + ".tmp", mpath)
    print(json.dumps({**res, "manifest": mpath}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
