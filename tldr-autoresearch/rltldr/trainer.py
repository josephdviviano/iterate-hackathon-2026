"""RLTL;DR policy trainer (runs on the trainer GPU, config trainer_gpu_uuid; ctl.sh pins it).

Waits for completed rollout groups (data/groups/gNNNN/READY, written by the driver), performs one update
phase per group and publishes the new LoRA adapter as `policy-v{N}` to vLLM (via the gateway, which
switches attempts over at the next attempt boundary).

Objective (paper Sec. 3, App. A.3):  L = L_GRPO + lambda * L_SFT
  * L_GRPO: PPO-clip (eps=0.2) on every action token of every LLM call of the 8 sequential attempts, which
    form ONE group. Dr.GRPO advantages A_k = r_k - mean(r) (no std division), DAPO token-level averaging,
    no KL / reference model, zero-advantage groups dropped, positive-ratio filtering (>= 75% positive).
    The rollouts were sampled asynchronously (possibly by an older policy version, by an FP8 server), so
    we use decoupled PPO: ratio vs. a proximal policy (current weights before the update) and a truncated
    importance weight min(pi_prox / pi_behaviour, C) against the vLLM sampling log-probs.
  * L_SFT: next-token NLL of the self-generated TL;DR insight tokens given the task (and earlier insights),
    i.e. the backprop mask flipped on the injected insight messages of conditioned attempts. This is
    applied even when the GRPO term is empty (all-fail groups => online SFTL;DR).
Training data are the exact token ids served by vLLM (recorded by the gateway); consecutive calls whose
prompt extends the previous prompt+completion are merged into one sequence.
"""
import argparse
import glob
import json
import logging
import math
import os
import random
import sys
import time

import requests
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rltldr.config import dump_config, load_config  # noqa: E402
from rltldr.io_utils import append_jsonl, read_json, read_jsonl, write_json_atomic  # noqa: E402

log = logging.getLogger("trainer")


# ------------------------------------------------------------------------------------------------------
# data preparation
# ------------------------------------------------------------------------------------------------------
def build_segments(calls: list) -> list:
    """Merge consecutive calls into training sequences when the next prompt extends prompt+completion.

    Returns [{"ids": [...], "act": [bool], "blp": [float], "n_calls": int}]; act marks sampled tokens.
    Calls with an incomplete or truncated completion contribute context but no action tokens.
    """
    segs, cur = [], None
    for c in calls:
        p, comp, lp = c["prompt_ids"], c["completion_ids"] or [], c["logprobs"] or []
        usable = c.get("status") == "ok" and c.get("finish_reason") in ("stop", "tool_calls") \
            and len(comp) == len(lp) and len(comp) > 0
        if cur is not None and len(p) >= len(cur["ids"]) and p[: len(cur["ids"])] == cur["ids"]:
            ext = p[len(cur["ids"]):]
        else:
            if cur is not None:
                segs.append(cur)
            cur, ext = {"ids": [], "act": [], "blp": [], "n_calls": 0}, list(p)
        cur["ids"] += ext + list(comp)
        cur["act"] += [False] * len(ext) + [usable] * len(comp)
        cur["blp"] += [0.0] * len(ext) + (list(lp) if usable else [0.0] * len(comp))
        cur["n_calls"] += 1
    if cur is not None:
        segs.append(cur)
    return segs


def bytes_to_unicode() -> dict:
    """GPT-2 byte-level BPE byte <-> printable-unicode table (removed from transformers 5)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class ByteOffsets:
    """Byte offsets of tokens for a byte-level BPE tokenizer (Qwen), used to locate insight spans."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.byte_dec = {v: k for k, v in bytes_to_unicode().items()}
        self.added = set(tokenizer.get_added_vocab().values())

    def token_bytes(self, ids):
        out = []
        for tid, piece in zip(ids, self.tok.convert_ids_to_tokens(ids)):
            if tid in self.added or any(ch not in self.byte_dec for ch in piece):
                out.append(piece.encode("utf-8"))
            else:
                out.append(bytes(self.byte_dec[ch] for ch in piece))
        return out

    def insight_positions(self, ids: list, injected: list) -> list:
        """Token positions covering each injected hint sentence (after 'Hint on what went wrong: ')."""
        pieces = self.token_bytes(ids)
        starts, pos = [], 0
        for b in pieces:
            starts.append(pos)
            pos += len(b)
        blob = b"".join(pieces)
        positions, search_from = [], 0
        for inj in injected:
            text, hint = inj["text"].encode("utf-8"), inj["hint"].encode("utf-8")
            at = blob.find(text, search_from)
            if at < 0:
                raise ValueError(f"injected insight text not found in prompt: {inj['text'][:80]!r}")
            h0 = at + len(text) - len(hint)
            h1 = at + len(text)
            search_from = h1
            for t, s in enumerate(starts):
                e = s + len(pieces[t])
                if e > h0 and s < h1:
                    positions.append(t)
        return sorted(set(positions))


def positive_ratio_filter(items: list, ratio: float, rng: random.Random) -> list:
    """Keep all positive-advantage rollouts and at most n_pos*(1-ratio)/ratio negative ones (App. A.3.2)."""
    if ratio <= 0:
        return items
    pos = [x for x in items if x["adv"] > 0]
    neg = [x for x in items if x["adv"] < 0]
    n_keep = int(math.floor(len(pos) * (1 - ratio) / ratio + 1e-9))
    rng.shuffle(neg)
    return pos + neg[:n_keep]


def chain_calls(calls: list) -> list:
    """The attempt's own conversation: calls whose prompt starts with the first call's system+task prefix.
    Requests made some other way (e.g. by hand from the agent's shell) are not policy rollouts."""
    if not calls:
        return []
    head = calls[0]["prompt_ids"]
    n = min(len(head), 256)
    return [c for c in calls if c["prompt_ids"][:n] == head[:n]]


def prepare_group(cfg, offsets: ByteOffsets, gdir: str):
    group = read_json(os.path.join(gdir, "group.json"))
    rollouts = [read_json(cfg.path("attempts", a, "rollout.json")) for a in group["attempts"]]
    rng = random.Random(cfg.seed * 100003 + group["group"])
    rewards = [float(r["reward"]) for r in rollouts]
    mean_r = sum(rewards) / len(rewards)
    stats = {"group": group["group"], "rewards": rewards, "n_rollouts": len(rollouts), "skipped_long": 0,
             "sft_span_errors": [], "n_sft_seqs": 0, "n_sft_tokens": 0, "n_grpo_seqs": 0, "n_grpo_tokens": 0,
             "dropped_calls": 0, "blocked_insights": 0}
    # insights retracted by the operator (e.g. written from a harness bug's false verdict) are never SFT targets;
    # they may still appear as context before other insights
    blocked = set((read_json(cfg.path("insight_blocklist.json"), {}) or {}).get("hints", []))

    per_rollout, sft = [], []
    for ro in rollouts:
        calls = [c for c in read_jsonl(ro["calls_path"])
                 if c.get("kind") == "agent" and c.get("prompt_ids") and c.get("status", "").startswith("ok")]
        chain = chain_calls(calls)
        stats["dropped_calls"] += len(calls) - len(chain)
        adv = float(ro["reward"]) - mean_r
        segs = []
        for sgm in build_segments(chain):
            if sum(sgm["act"]) == 0:
                continue
            if len(sgm["ids"]) > cfg.max_train_seq_len:
                stats["skipped_long"] += 1
                continue
            segs.append(sgm)
        per_rollout.append({"attempt_id": ro["attempt_id"], "adv": adv, "segments": segs,
                            "version": (ro.get("policy") or {}).get("version")})
        # L_SFT: insight tokens of the first call of a conditioned attempt, given task + earlier insights
        inj = chain[0].get("injected") if chain else None
        if inj and blocked:
            kept_inj = [x for x in inj if x["hint"] not in blocked]
            stats["blocked_insights"] += len(inj) - len(kept_inj)
            inj = kept_inj
        if inj:
            ids = chain[0]["prompt_ids"]
            try:
                pos = offsets.insight_positions(ids, inj)
                decoded = offsets.tok.decode([ids[p] for p in pos])
                if not pos or not all(x["hint"].split()[0] in decoded for x in inj):
                    raise ValueError(f"span check failed: {decoded[:200]!r}")
            except Exception as e:  # never let one odd hint stall training: skip its SFT term, log it
                log.warning("insight span error in %s: %s", ro["attempt_id"], e)
                stats["sft_span_errors"].append(ro["attempt_id"])
                continue
            sft.append({"attempt_id": ro["attempt_id"], "ids": ids[: pos[-1] + 1], "pos": pos})
            stats["n_sft_seqs"] += 1
            stats["n_sft_tokens"] += len(pos)

    # GRPO candidates: non-zero advantage and at least one trainable sequence (BEFORE ratio filtering, so the
    # positive-ratio guarantee holds for what is actually trained on)
    grpo = [x for x in per_rollout if abs(x["adv"]) > 1e-8 and x["segments"]]
    grpo = positive_ratio_filter(grpo, cfg.positive_ratio, rng)
    items = []
    for x in grpo:
        for sgm in x["segments"]:
            items.append({"attempt_id": x["attempt_id"], "adv": x["adv"], "ids": sgm["ids"],
                          "pos": [i for i, a in enumerate(sgm["act"]) if a],
                          "blp": [bb for bb, a in zip(sgm["blp"], sgm["act"]) if a]})
            stats["n_grpo_seqs"] += 1
            stats["n_grpo_tokens"] += sum(sgm["act"])
    stats["grpo_rollouts"] = [x["attempt_id"] for x in grpo]
    stats["segments_per_rollout"] = [len(x["segments"]) for x in per_rollout]
    stats["seq_lens"] = [len(sg["ids"]) for x in per_rollout for sg in x["segments"]]
    return items, sft, stats


# ------------------------------------------------------------------------------------------------------
# trainer
# ------------------------------------------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg):
        from rltldr import model_utils as mu
        self.mu, self.cfg = mu, cfg
        self.state_path = cfg.path("trainer_state.json")
        self.ckpt_dir = cfg.path("trainer_ckpt")
        torch.manual_seed(cfg.seed)
        t0 = time.time()
        self.model, self.tok = mu.load_policy(cfg.trainer_base_dir, lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=cfg.lr, betas=tuple(cfg.adam_betas),
                                     weight_decay=cfg.weight_decay, fused=True)
        self.state = read_json(self.state_path) or {"version": 0, "groups_done": []}
        if os.path.exists(os.path.join(self.ckpt_dir)):
            meta, _ = mu.load_trainer_state(self.model, self.opt, self.ckpt_dir)
            # crash between checkpoint save and state write: the checkpoint is the source of truth
            if meta.get("version", 0) != self.state["version"]:
                log.warning("reconciling trainer state %s with checkpoint %s", self.state, meta)
                self.state["version"] = meta["version"]
                g = meta.get("group")
                if g is not None and g not in self.state["groups_done"]:
                    self.state["groups_done"].append(g)
                    done = cfg.path("groups", f"g{g:04d}", "DONE")
                    if not os.path.exists(done):
                        write_json_atomic(done, {"group": g, "updated": True, "version": meta["version"],
                                                 "note": "reconciled after trainer restart"})
                write_json_atomic(self.state_path, self.state)
        for g in self.state["groups_done"]:
            done = cfg.path("groups", f"g{g:04d}", "DONE")
            if os.path.isdir(os.path.dirname(done)) and not os.path.exists(done):
                write_json_atomic(done, {"group": g, "note": "reconciled after trainer restart"})
        self.offsets = ByteOffsets(self.tok)
        log.info("model loaded in %.0fs; version=%d; trainable=%.1fM; mem=%.1fGiB", time.time() - t0,
                 self.state["version"], sum(p.numel() for p in self.params) / 1e6,
                 torch.cuda.memory_allocated() / 2**30)

    # -- log-probs ------------------------------------------------------------------------------------
    def logprobs(self, ids, pos, grad: bool):
        ids_t = torch.tensor(ids, dtype=torch.long, device="cuda")
        pos_t = torch.tensor(pos, dtype=torch.long, device="cuda")
        offload = len(ids) > self.cfg.offload_threshold
        if grad:
            return self.mu.token_logprobs(self.model, ids_t, pos_t, offload=offload)
        with torch.no_grad():
            return self.mu.token_logprobs(self.model, ids_t, pos_t, offload=False).float()

    # -- one update phase ----------------------------------------------------------------------------
    def update(self, items, sft):
        cfg = self.cfg
        self.model.train()
        m = {"steps": 0, "loss_grpo": [], "loss_sft": [], "grad_norm": [], "clipfrac_by_epoch": [],
             "ratio_mean_by_epoch": []}
        # proximal policy = weights before this phase (decoupled PPO). Stability diagnostics vs. the vLLM
        # behaviour log-probs (processed: after top-k/top-p, hence typically >= the trainer's untruncated ones)
        d_sum, d_abs, lp_sum, n_tok, clamped = 0.0, 0.0, 0.0, 0, 0
        for it in items:
            it["lp_prox"] = self.logprobs(it["ids"], it["pos"], grad=False).cpu()
            d = it["lp_prox"] - torch.tensor(it["blp"])
            d_sum += d.sum().item()
            d_abs += d.abs().sum().item()
            lp_sum += it["lp_prox"].sum().item()
            clamped += (d.exp() > cfg.tis_cap).sum().item()
            n_tok += d.numel()
        m.update(mismatch_signed=d_sum / max(1, n_tok), mismatch_abs=d_abs / max(1, n_tok),
                 mean_logprob_sampled=lp_sum / max(1, n_tok), is_w_clamped_frac=clamped / max(1, n_tok))

        rng = random.Random(cfg.seed + self.state["version"])
        # minibatches are anchored on the GRPO sequences; insight-SFT sequences are spread over the same
        # minibatches, so every optimizer step mixes both terms in the ratio set by lambda
        nmb = max(1, min(cfg.num_minibatches, len(items) if items else len(sft)))
        for epoch in range(cfg.ppo_epochs):
            order_i, order_s = list(range(len(items))), list(range(len(sft)))
            rng.shuffle(order_i)
            rng.shuffle(order_s)
            clipped, toks, rsum = 0.0, 0, 0.0
            for b in range(nmb):
                mb_i = [items[j] for j in order_i[b::nmb]]
                mb_s = [sft[j] for j in order_s[b::nmb]]
                n_act = sum(len(it["pos"]) for it in mb_i)
                n_ins = sum(len(x["pos"]) for x in mb_s)
                if n_act == 0 and n_ins == 0:
                    continue
                self.opt.zero_grad(set_to_none=True)
                lg, ls = 0.0, 0.0
                for it in mb_i:
                    lp = self.logprobs(it["ids"], it["pos"], grad=True)
                    lp_prox = it["lp_prox"].to(lp.device)
                    blp = torch.tensor(it["blp"], device=lp.device)
                    ratio = torch.exp(lp - lp_prox)
                    is_w = torch.exp(lp_prox - blp).clamp(max=cfg.tis_cap)
                    a = it["adv"]
                    surr = torch.minimum(ratio * a, ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * a)
                    loss = -(is_w * surr).sum() / n_act
                    loss.backward()
                    lg += loss.item()
                    with torch.no_grad():
                        clipped += ((ratio - 1).abs() > cfg.clip_eps).float().sum().item()
                        toks += ratio.numel()
                        rsum += ratio.sum().item()
                for x in mb_s:
                    lp = self.logprobs(x["ids"], x["pos"], grad=True)
                    loss = -cfg.sft_lambda * lp.sum() / n_ins
                    loss.backward()
                    ls += loss.item()
                gn = torch.nn.utils.clip_grad_norm_(self.params, cfg.grad_clip)
                self.opt.step()
                m["steps"] += 1
                m["grad_norm"].append(float(gn))
                m["loss_grpo"].append(lg)
                m["loss_sft"].append(ls)
            m["clipfrac_by_epoch"].append(clipped / toks if toks else None)
            m["ratio_mean_by_epoch"].append(rsum / toks if toks else None)
        self.opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return m

    # -- publishing ----------------------------------------------------------------------------------
    def publish(self, version: int):
        name = f"{self.cfg.run_name}-v{version}"
        path = self.cfg.path("adapters", f"policy-v{version}")
        for i in range(60):
            try:
                r = requests.post(f"{self.cfg.gateway_url}/control/policy",
                                  json={"name": name, "path": path, "version": version}, timeout=900)
                if r.status_code == 200:
                    log.info("published %s", name)
                    return True
                log.warning("publish %s: HTTP %s %s", name, r.status_code, r.text[:300])
            except requests.RequestException as e:
                log.warning("publish %s: %s", name, e)
            time.sleep(min(60, 5 * (i + 1)))
        return False

    def ensure_published(self):
        """After a restart: make sure the gateway knows our latest version."""
        v = self.state["version"]
        if v == 0:
            return
        try:
            gs = requests.get(f"{self.cfg.gateway_url}/control/state", timeout=30).json()
            known = max((gs.get("policy") or {}).get("version", 0), (gs.get("pending") or {}).get("version", 0))
        except requests.RequestException:
            known = -1
        if known < v:
            self.publish(v)

    def process_group(self, gdir: str):
        cfg = self.cfg
        t0 = time.time()
        items, sft, stats = prepare_group(cfg, self.offsets, gdir)
        log.info("group %s: grpo seqs=%d (%d tok), sft seqs=%d (%d tok), rewards=%s, seq_lens=%s",
                 stats["group"], stats["n_grpo_seqs"], stats["n_grpo_tokens"], stats["n_sft_seqs"],
                 stats["n_sft_tokens"], stats["rewards"], stats["seq_lens"])
        result = {"group": stats["group"], "data": stats, "t_prepare": time.time() - t0}
        if not items and not sft:
            log.info("group %s: no learning signal (no advantage, no insights); no update", stats["group"])
            result["updated"] = False
        else:
            torch.cuda.reset_peak_memory_stats()
            t1 = time.time()
            result["update"] = self.update(items, sft)
            result["t_update"] = time.time() - t1
            result["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
            v = self.state["version"] + 1
            self.mu.export_adapter(self.model, cfg.path("adapters", f"policy-v{v}"), overwrite=True)
            self.state["version"] = v
            self.mu.save_trainer_state(self.model, self.opt, self.ckpt_dir, {"version": v, "group": stats["group"]})
            result["updated"], result["version"] = True, v
        # order matters for crash recovery: DONE first, then the state (a restart re-derives either from the
        # checkpoint metadata, see __init__), so a group is never trained twice
        write_json_atomic(os.path.join(gdir, "DONE"), result)
        self.state["groups_done"].append(stats["group"])
        write_json_atomic(self.state_path, self.state)
        append_jsonl(cfg.path("metrics_trainer.jsonl"), {**result, "t": time.time()})
        if result.get("updated"):
            self.publish(self.state["version"])
        log.info("group %s done in %.0fs: %s", stats["group"], time.time() - t0,
                 json.dumps({k: v for k, v in result.items() if k != "data"})[:1500])

    def run(self, once: bool = False):
        self.ensure_published()
        while True:
            done = set(self.state["groups_done"])
            pending = sorted(d for d in glob.glob(self.cfg.path("groups", "g*"))
                             if os.path.exists(os.path.join(d, "READY")) and not os.path.exists(os.path.join(d, "DONE"))
                             and read_json(os.path.join(d, "group.json"), {}).get("group") not in done)
            if pending:
                gdir = pending[0]
                fails = read_json(os.path.join(gdir, "FAILS"), {"n": 0})
                try:
                    self.process_group(gdir)
                except torch.cuda.OutOfMemoryError:
                    raise                     # let the supervisor restart with a clean CUDA context
                except Exception as e:  # a deterministic failure must not crash-loop the trainer
                    log.exception("group %s failed", gdir)
                    fails = {"n": fails["n"] + 1, "last_error": repr(e)[:1000]}
                    write_json_atomic(os.path.join(gdir, "FAILS"), fails)
                    if fails["n"] >= 3:
                        log.error("group %s failed 3 times; skipping it", gdir)
                        write_json_atomic(os.path.join(gdir, "DONE"), {"updated": False, "skipped": True, **fails})
                    self.opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                continue
            if once:
                return
            time.sleep(10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="process pending groups then exit")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cfg = load_config()
    log.info("config:\n%s", dump_config(cfg))
    Trainer(cfg).run(once=args.once)


if __name__ == "__main__":
    main()
