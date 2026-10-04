"""
lit.py: the deep-research literature pipeline of the hypothesis framework. Task-agnostic.

One request (a question plus relevance criteria) becomes one digest, in five stages:

    S0 plan        slow model   search queries + a relevance rubric, from the request and the tree
    S1 harvest     no model     arXiv and OpenAlex search results, de-duplicated
    S2 triage      fast model   title + abstract scored 0-3 against the rubric, 20 per call
    S3 extract     fast model   full text of the best papers -> claim cards with exact quotes
                                (a card whose quote is not in the text is discarded)
    S4 synthesize  slow model   a one-page digest: claims mapped to hypotheses and reconciled
                                with the experiments run so far

Requests come from the experiment agent (`research.py lit ask`) or are generated automatically
after every experiment, depending on framework.json ("lit": "async" | "forced").

State (workspace root, untracked by git):
    lit/requests/R003.json   lit/digests/D003.md   lit/cards.jsonl   lit/ledger.tsv   lit/cache/

    python lit.py run R003     # process one request
    python lit.py serve        # process open requests as they arrive (the async feed)
"""

import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import research as R  # noqa: E402

LIT_DIR = os.path.join(R.ROOT, "lit")
REQ_DIR = os.path.join(LIT_DIR, "requests")
DIGEST_DIR = os.path.join(LIT_DIR, "digests")
CACHE_DIR = os.path.join(LIT_DIR, "cache")
CARDS_FILE = os.path.join(LIT_DIR, "cards.jsonl")
LEDGER_FILE = os.path.join(LIT_DIR, "ledger.tsv")
SERVE_FILE = os.path.join(R.STATE_DIR, "lit_serve.json")

FAST_MODEL = os.environ.get("RESEARCH_LIT_FAST_MODEL", "claude-haiku-4-5-20251001")
SLOW_MODEL = os.environ.get("RESEARCH_LIT_SLOW_MODEL", R.META_MODEL)
BUDGET = float(
    os.environ.get("RESEARCH_LIT_BUDGET", "1.0")
)  # USD per request, all stages
FIXTURE = os.environ.get(
    "RESEARCH_LIT_FIXTURE"
)  # tests: canned search results and texts
MAX_PAPERS, BATCH, MAX_FULLTEXT, TEXT_CHARS = 80, 20, 5, 30000
TRIAGE_CAP, EXTRACT_CAP = (
    0.12,
    0.15,
)  # USD per fast-model call (each carries ~$0.02 of fixed overhead)

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "queries": {"type": "array", "items": {"type": "string"}},
        "rubric": {"type": "array", "items": {"type": "string"}},
        "exclude": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["question", "queries", "rubric", "exclude"],
}
TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {"scores": {"type": "array", "items": {"type": "object", "properties": {
        "index": {"type": "integer"}, "score": {"type": "integer"}, "reason": {"type": "string"}},
        "required": ["index", "score"]}}},
    "required": ["scores"],
}  # fmt: skip
EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {"cards": {"type": "array", "items": {"type": "object", "properties": {
        "claim": {"type": "string"}, "conditions": {"type": "string"}, "effect": {"type": "string"},
        "evidence": {"type": "string"}, "quote": {"type": "string"},
        "hypotheses": {"type": "array", "items": {"type": "string"}}},
        "required": ["claim", "conditions", "effect", "evidence", "quote", "hypotheses"]}}},
    "required": ["cards"],
}  # fmt: skip
SYNTH_SCHEMA = {
    "type": "object",
    "properties": {
        "digest": {"type": "string"},
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "card": {"type": "string"},
            "kind": {"type": "string", "enum": ["explains", "conditional", "contradicts", "untested"]},
            "hypotheses": {"type": "array", "items": {"type": "string"}},
            "use": {"type": "string"}},
            "required": ["card", "kind", "hypotheses", "use"]}},
        "summary": {"type": "string"},
    },
    "required": ["digest", "claims", "summary"],
}  # fmt: skip


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def ensure_dirs():
    for d in (REQ_DIR, DIGEST_DIR, CACHE_DIR):
        os.makedirs(d, exist_ok=True)


def ledger(rid, stage, model, cost):
    new = not os.path.exists(LEDGER_FILE)
    with open(LEDGER_FILE, "a") as f:
        if new:
            f.write("request\tstage\tmodel\tcost_usd\tat\n")
        f.write(
            f"{rid}\t{stage}\t{model}\t{cost:.4f}\t{time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
        )


def requests():
    if not os.path.isdir(REQ_DIR):
        return []
    return [
        R.read_json(os.path.join(REQ_DIR, f))
        for f in sorted(os.listdir(REQ_DIR))
        if f.endswith(".json")
    ]


def save_request(req):
    R.write_json(os.path.join(REQ_DIR, f"{req['id']}.json"), req)


def new_request(question, hypotheses, must, exclude, source, after):
    ensure_dirs()
    nums = [int(r["id"][1:]) for r in requests()]
    req = {"id": f"R{max(nums, default=0) + 1:03d}", "from": source, "after": after, "question": question,
           "hypotheses": hypotheses, "must_have": must, "exclude": exclude, "status": "open",
           "created": time.time(), "digest": ""}  # fmt: skip
    save_request(req)
    return req


def http(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "autoresearch-lit/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def compact_results():
    """The experiments so far, compactly: the task's results table and the verdicts."""
    return (f"=== RESULTS (results.tsv) ===\n{R.read_text(R.RESULTS_FILE)[-20000:]}\n"
            f"=== VERDICTS (predictions.tsv) ===\n{R.read_text(R.PRED_FILE)[-6000:]}\n")  # fmt: skip


# ---------------------------------------------------------------------------
# S1 harvest (no model)
# ---------------------------------------------------------------------------


def arxiv_search(query, n=25):
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": f"all:{query}", "max_results": n, "sortBy": "relevance"})  # fmt: skip
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in ET.fromstring(http(url)).findall("a:entry", ns):
        aid = e.findtext("a:id", "", ns).rsplit("/abs/", 1)[-1]
        out.append({"title": " ".join(e.findtext("a:title", "", ns).split()),
                    "abstract": " ".join(e.findtext("a:summary", "", ns).split()),
                    "year": e.findtext("a:published", "", ns)[:4], "arxiv": re.sub(r"v\d+$", "", aid),
                    "url": f"https://arxiv.org/abs/{aid}", "source": "arxiv"})  # fmt: skip
    return out


def openalex_search(query, n=25):
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
        {"search": query, "per-page": n, "select": "title,publication_year,abstract_inverted_index,ids,cited_by_count"})  # fmt: skip
    out = []
    for w in json.loads(http(url)).get("results", []):
        inv = w.get("abstract_inverted_index") or {}
        words = sorted((pos, word) for word, poss in inv.items() for pos in poss)
        arxiv = ""
        for v in (w.get("ids") or {}).values():
            m = re.search(r"arxiv\.org/abs/([\w.]+)", str(v))
            if m:
                arxiv = re.sub(r"v\d+$", "", m.group(1))
        out.append({"title": w.get("title") or "", "abstract": " ".join(word for _, word in words),
                    "year": str(w.get("publication_year") or ""), "arxiv": arxiv,
                    "url": (w.get("ids") or {}).get("doi") or (w.get("ids") or {}).get("openalex", ""),
                    "citations": w.get("cited_by_count", 0), "source": "openalex"})  # fmt: skip
    return out


def harvest(queries):
    if FIXTURE:
        return R.read_json(FIXTURE)["papers"][:MAX_PAPERS]
    papers, seen = [], set()
    for q in queries:
        for fn in (arxiv_search, openalex_search):
            try:
                found = fn(q)
            except Exception as exc:  # a source being down never stops the request
                print(f"harvest: {fn.__name__}({q!r}) failed: {exc}", flush=True)
                found = []
            for p in found:
                key = re.sub(r"\W+", "", p["title"].lower())[:80]
                if key and key not in seen and p["abstract"]:
                    seen.add(key)
                    papers.append(p)
            if fn is arxiv_search:
                time.sleep(3)  # arXiv's API asks for one request per 3 s
    return papers[:MAX_PAPERS]


def full_text(paper):
    if FIXTURE:
        return (
            R.read_json(FIXTURE).get("texts", {}).get(paper["title"], paper["abstract"])
        )
    if not paper.get("arxiv"):
        return paper["abstract"]
    path = os.path.join(CACHE_DIR, paper["arxiv"].replace("/", "_") + ".txt")
    if os.path.exists(path):
        return R.read_text(path)
    try:
        page = http(f"https://arxiv.org/html/{paper['arxiv']}", timeout=60)
    except Exception:
        return paper["abstract"]
    page = re.sub(r"(?s)<(script|style|math)\b.*?</\1>", " ", page)
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", page))).strip()
    with open(path, "w") as f:
        f.write(text)
    return text


def normalize(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------


def run_request(req):
    """Process one request into a digest. Returns the digest id."""
    ensure_dirs()
    rid, spent = req["id"], 0.0

    def call(stage, prompt, schema, model, effort, cap):
        nonlocal spent
        out, cost = R.llm(
            f"lit-{stage}",
            prompt,
            schema,
            model=model,
            effort=effort,
            budget=max(0.02, cap),
        )
        spent += cost
        ledger(rid, stage, model, cost)
        return out

    task, tree = R.inputs_task(), R.inputs_hypotheses()
    asked = (f"Question: {req['question']}\nHypotheses: {', '.join(req['hypotheses']) or '-'}\n"
             f"Must have: {'; '.join(req['must_have']) or '-'}\nExclude: {'; '.join(req['exclude']) or '-'}"
             if req["question"] else
             f"(automatic request after {req['after']}: choose the most decision-relevant open question yourself)")  # fmt: skip
    plan = call(
        "plan",
        f"""ROLE: lit-plan. You plan a literature search for an autonomous research loop. Do not use tools.

{R.TREE_RULES}

{task}{tree}{compact_results()}=== REQUEST ===
{asked}

Return: the question the search must answer (restate it, or choose it for an automatic request:
prefer what would most change the next experiments: an open hypothesis near 0.5 confidence, a
surprising or failed result, a direction the tree never considered); 3-6 short search queries
(plain keywords, as typed into arXiv); a rubric of 2-4 concrete criteria a relevant paper must
meet; and exclusions (including anything the task's rules forbid).""",
        PLAN_SCHEMA,
        SLOW_MODEL,
        "medium",
        0.25 * BUDGET,
    )
    req.update(
        question=req["question"] or plan["question"], plan=plan, status="running"
    )
    save_request(req)

    papers = harvest(plan["queries"][:6])
    scored = []
    for start in range(0, len(papers), BATCH):
        if spent > 0.45 * BUDGET:
            break
        batch = papers[start : start + BATCH]
        listing = "\n".join(
            f"[{i}] {p['title']} ({p['year']}): {p['abstract'][:900]}"
            for i, p in enumerate(batch)
        )
        try:
            out = call("triage", f"""ROLE: lit-triage. Score each paper 0-3 for relevance to the question. Do not use tools.
Question: {req['question']}
Rubric (a 3 meets all): {'; '.join(plan['rubric'])}
Exclude (score 0): {'; '.join(plan['exclude'])}

{listing}""", TRIAGE_SCHEMA, FAST_MODEL, "low", TRIAGE_CAP)  # fmt: skip
        except (
            RuntimeError
        ) as exc:  # one failed batch costs that batch, not the request
            print(f"triage batch skipped: {exc}", flush=True)
            continue
        for s in out["scores"]:
            if 0 <= s["index"] < len(batch) and s["score"] >= 2:
                scored.append((s["score"], batch[s["index"]]))
    best = [p for _, p in sorted(scored, key=lambda x: -x[0])][:MAX_FULLTEXT]

    cards = []
    for p in best:
        if spent > 0.6 * BUDGET:
            break
        text = full_text(p)
        try:
            out = call("extract", f"""ROLE: lit-extract. Extract the claims of this paper that bear on the question. Do not use tools.
Question: {req['question']}
Hypothesis ids you may map claims to: {', '.join(h['id'] for h in R.read_tsv(R.HYPO_FILE, R.HYPO_COLS))}
For each claim: the claim, its conditions (data, model, budget, hardware), the effect size, the
kind of evidence, and an EXACT quote copied from the text that supports it (claims without an
exact quote are discarded).

=== PAPER: {p['title']} ({p['year']}, {p['url']}) ===
{text[:TEXT_CHARS]}""", EXTRACT_SCHEMA, FAST_MODEL, "low", EXTRACT_CAP)  # fmt: skip
        except RuntimeError as exc:
            print(f"extraction skipped for {p['title'][:60]}: {exc}", flush=True)
            continue
        haystack = normalize(text)
        for c in out["cards"]:
            if len(normalize(c["quote"])) >= 20 and normalize(c["quote"]) in haystack:
                cards.append(
                    {**c, "paper": p["title"], "url": p["url"], "year": p["year"]}
                )

    existing = [
        json.loads(line)
        for line in R.read_text(CARDS_FILE).splitlines()
        if line.strip()
    ]
    base = len(existing)
    for k, c in enumerate(cards):
        c["id"] = f"C{base + k + 1:04d}"
        c["request"] = rid
    with open(CARDS_FILE, "a") as f:
        for c in cards:
            f.write(json.dumps(c) + "\n")

    card_text = "\n".join(f"{c['id']} [{c['paper']} ({c['year']})] {c['claim']} | conditions: {c['conditions']} | "
                          f"effect: {c['effect']} | evidence: {c['evidence']} | maps to {c['hypotheses']}" for c in cards)  # fmt: skip
    synth = call("synthesize", f"""ROLE: lit-synthesize. Write a short digest for the experimenter of an autonomous research loop. Do not use tools.

{R.TREE_RULES}

{task}{tree}{compact_results()}=== QUESTION ===
{req['question']}
=== VERIFIED CLAIM CARDS (quotes checked against the papers) ===
{card_text or '(no verified claims were found)'}

Write the digest in markdown, at most ~400 words: what the literature says about the question,
under which conditions, and how it bears on THIS loop's hypotheses and results. For each claim
you use, give its card id and say whether it explains a result already seen, holds only in a
regime not yet tested here (conditional), contradicts a current belief, or points at a mechanism
the tree lacks (untested), and what to do with it. If nothing relevant was found, say so plainly.""",
                  SYNTH_SCHEMA, SLOW_MODEL, "high", max(0.3, BUDGET - spent))  # fmt: skip

    did = "D" + rid[1:]
    header = (f"# {did}: {req['question']}\n\nRequest {rid} ({req['from']}, after {req['after']}); "
              f"{len(papers)} papers found, {len(scored)} relevant, {len(best)} read in full, "
              f"{len(cards)} verified claims; cost ${spent:.2f}\n\n")  # fmt: skip
    claims = "\n".join(
        f"- {c['card']} ({c['kind']}; {', '.join(c['hypotheses']) or '-'}): {c['use']}"
        for c in synth["claims"]
    )
    sources = "\n".join(
        f"- {c['id']}: {c['paper']} ({c['year']}) {c['url']}" for c in cards
    )
    with open(os.path.join(DIGEST_DIR, f"{did}.md"), "w") as f:
        f.write(header + synth["digest"].strip() + "\n\n## Claims\n" + (claims or "- none") +
                "\n\n## Sources\n" + (sources or "- none") + "\n")  # fmt: skip
    req.update(status="done", digest=did, cost=round(spent, 4))
    save_request(req)
    return did


def run_and_record(req):
    try:
        did = run_request(req)
        print(f"{req['id']} -> {did}", flush=True)
        return did
    except Exception as exc:
        req.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:400])
        save_request(req)
        print(f"{req['id']} failed: {exc}", flush=True)
        return None


def serve():
    R.write_json(SERVE_FILE, {"pid": os.getpid(), "started": time.time()})
    while True:
        pending = [r for r in requests() if r["status"] == "open"]
        if pending:
            run_and_record(pending[0])
        else:
            time.sleep(15)


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "serve":
        serve()
    elif len(sys.argv) >= 3 and sys.argv[1] == "run":
        req = R.read_json(os.path.join(REQ_DIR, f"{sys.argv[2]}.json"))
        if not req:
            sys.exit(f"no request {sys.argv[2]}")
        run_and_record(req)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
