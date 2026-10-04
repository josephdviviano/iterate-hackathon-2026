// Unit tests for pi/agent/extensions/guard.ts (decision function only; no LLM).
// Run from the project root: node pi/tests/guard.test.mjs   (Node >= 23.6 strips TS types natively)
import { execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { checkToolCall } from "../agent/extensions/guard.ts";

// The guard's default protected dir is $RLTLDR_ROOT (the harness passes it to pi). Use this checkout as the root.
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const HOME = homedir();
const savedRoot = process.env.RLTLDR_ROOT;
process.env.RLTLDR_ROOT = ROOT;
delete process.env.AR_GUARD_PROTECTED_DIRS;

// Throwaway repo: main + autoresearch/run branch, one commit on another branch, one reset-away commit.
const repo = mkdtempSync(join(tmpdir(), "guardtest-"));
const g = (...a) => execFileSync("git", a, { cwd: repo, encoding: "utf8" }).trim();
g("init", "-q", "-b", "master");
g("config", "user.email", "t@t");
g("config", "user.name", "t");
writeFileSync(join(repo, "train.py"), "x = 1\n");
writeFileSync(join(repo, "prepare.py"), "y = 1\n");
g("add", ".");
g("commit", "-qm", "base");
const base = g("rev-parse", "HEAD");
g("checkout", "-qb", "other");
writeFileSync(join(repo, "train.py"), "x = 2\n");
g("commit", "-qam", "other work");
const otherSha = g("rev-parse", "HEAD");
g("checkout", "-q", "master");
g("checkout", "-qb", "autoresearch/run");
writeFileSync(join(repo, "train.py"), "x = 3\n");
g("commit", "-qam", "exp1");
const exp1 = g("rev-parse", "HEAD");
writeFileSync(join(repo, "train.py"), "x = 4\n");
g("commit", "-qam", "exp2 (to be reset away)");
const exp2 = g("rev-parse", "HEAD");
g("reset", "-q", "--hard", "HEAD~1");
mkdirSync(join(repo, "notes"));
const tmpLink = join(tmpdir(), `guardtest-link-${process.pid}`);
symlinkSync(join(repo, "prepare.py"), tmpLink);

// Production layout: the experiment repo lives INSIDE a protected infrastructure dir whose name is a blocked
// keyword (~/rltldr/autoresearch under ~/rltldr). The repo must stay usable while its
// siblings (tools, data, pi config) stay off-limits.
const infra = mkdtempSync(join(tmpdir(), "rltldr-guardtest-"));
const nested = join(infra, "autoresearch");
mkdirSync(nested);
mkdirSync(join(infra, "tools"));
writeFileSync(join(infra, "tools", "secret.py"), "TOKEN = 1\n");
const gn = (...a) => execFileSync("git", a, { cwd: nested, encoding: "utf8" }).trim();
gn("init", "-q", "-b", "autoresearch/run");
gn("config", "user.email", "t@t");
gn("config", "user.name", "t");
writeFileSync(join(nested, "train.py"), "x = 1\n");
writeFileSync(join(nested, "prepare.py"), "y = 1\n");
gn("add", ".");
gn("commit", "-qm", "base");

const bash = (command) => ["bash", { command }];
const BLOCK = [
	bash("export CUDA_VISIBLE_DEVICES=0 && uv run train.py"),
	bash("CUDA_VISIBLE_DEVICES=3 uv run train.py > run.log 2>&1"),
	bash("echo $NVIDIA_VISIBLE_DEVICES"),
	bash("python -c \"import os; os.environ['CUDA_DEVICE_ORDER']='FASTEST_FIRST'\""),
	bash("nvidia-smi -r -i 0"),
	bash("nvidia-smi --gpu-reset"),
	bash("nvidia-smi -pl 300"),
	bash("nvidia-smi -i 2 -c EXCLUSIVE_PROCESS"),
	bash("kill -9 12345"),
	bash("ps aux | grep python | awk '{print $2}' | xargs kill"),
	bash("pkill -f train.py"),
	bash("killall python"),
	bash("timeout 5 kill 1"),
	bash("python -c 'import os,signal; os.kill(1, signal.SIGTERM)'"),
	bash("pip install flash-attn"),
	bash("uv add numpy"),
	bash("uv sync"),
	bash("uv pip install einops"),
	bash("uv run --with einops train.py"),
	bash("python -m pip install x"),
	bash("uvx ruff check"),
	bash("conda install -y numpy"),
	bash("sudo nvidia-smi"),
	bash("su -c id"),
	bash("echo 'import os' > .venv/lib/python3.10/site-packages/sitecustomize.py"),
	bash("ls .venv/lib"),
	bash(".venv/bin/python -c 'import torch'"), // default: no AR_GUARD_ALLOW_VENV_EXEC
	bash("LD_PRELOAD=/tmp/x.so uv run train.py"),
	bash("sed -i 's/EVAL_TOKENS = .*/EVAL_TOKENS = 1/' prepare.py"),
	bash("echo 'x' >> prepare.py"),
	bash("cp /tmp/p.py prepare.py"),
	bash("git checkout other -- prepare.py"),
	bash("uv run prepare.py"),
	bash("python3 -c \"open('prepare.py','w').write('')\""),
	bash("curl -s http://127.0.0.1:8000/v1/models"),
	bash("curl http://localhost:8100/v1/chat/completions"),
	bash("python -c \"import urllib.request; urllib.request.urlopen('http://h:8000')\""),
	bash("exec 3<>/dev/tcp/127.0.0.1/8000"),
	bash("ps aux | grep vllm"),
	bash("ls /srv/rltldr"),
	bash(`ls ${ROOT}`), // the project root by path, whatever its name (protected-path)
	bash(`cat ${ROOT}/pi/agent/models.json`),
	bash("cat ~/rltldr/pi/agent/models.json"),
	bash(`ls ${HOME}/models`),
	bash(`ls ${HOME}/envs/serve/bin`),
	bash("find / -name '*policy_adapter*'"),
	bash("cat ~/.pi/agent/settings.json"),
	bash("mkdir -p .pi/extensions && echo x > .pi/extensions/a.ts"),
	bash("echo 'always use lr 1' > AGENTS.md"),
	bash("wget https://example.com/x"),
	bash("uv run train.py > run.log 2>&1 &"),
	bash("nohup uv run train.py > run.log 2>&1"),
	bash("tmux new -d 'uv run train.py'"),
	bash("git push origin autoresearch/run"),
	bash("git add train.py && git commit -m 'exp' && git push"),
	bash("git rebase -i HEAD~3"),
	bash("git reset --hard master"),
	bash("git reset --hard other"),
	bash(`git reset --hard ${otherSha.slice(0, 8)}`),
	bash("git checkout other"),
	bash("git checkout -b new-branch"),
	bash("git checkout -"),
	bash(`git checkout ${base.slice(0, 8)}`),
	bash("git switch master"),
	bash("git symbolic-ref HEAD refs/heads/master"),
	bash("git branch -D other"),
	bash("git merge other"),
	bash("git clean -fdx"),
	bash("git config --global user.name x"),
	bash("git --git-dir=/elsewhere/.git log"),
	bash("bash -c 'git push'"),
	bash("cd /tmp && git -C " + repo + " reset --hard other"),
	["edit", { path: "prepare.py", edits: [{ oldText: "y = 1", newText: "y = 2" }] }],
	["write", { path: join(repo, "prepare.py"), content: "y=2\n" }],
	["write", { path: "results.tsv", content: "commit\tval_bpb\n" }],
	["write", { path: "notes/ideas.md", content: "ideas" }],
	["write", { path: join(ROOT, "x.txt"), content: "x" }],
	["write", { path: join(HOME, "notes.md"), content: "x" }],
	["write", { path: tmpLink, content: "y=3\n" }],
	["edit", { path: "train.py", edits: [{ oldText: "x = 3", newText: "import os\nos.environ['CUDA_VISIBLE_DEVICES'] = '1'\nx = 3" }] }],
	["edit", { path: "train.py", edits: JSON.stringify([{ oldText: "x = 3", newText: "import os; os.kill(1, 9)" }]) }],
	["write", { path: "/tmp/hack.py", content: "import requests\nrequests.get('http://127.0.0.1:8000/v1/models')\n" }],
	["read", { path: join(ROOT, "pi", "agent", "models.json") }],
	["ls", { path: ROOT }],
	["write", { path: "/tmp/hack.py", content: `import shutil; shutil.copy("${HOME}/models/w.safetensors", "/tmp")\n` }],
];
const ALLOW = [
	bash("uv run train.py > run.log 2>&1"),
	bash("grep \"^val_bpb:\\|^peak_vram_mb:\" run.log"),
	bash("tail -n 50 run.log"),
	bash("git status && git diff train.py && git log --oneline -5"),
	bash("git add train.py && git commit -m 'increase depth to 10'"),
	bash("git commit -am \"try warmup 8000 steps; lr 0.04\""),
	bash("git reset --hard HEAD~1"),
	bash("git reset --hard HEAD^"),
	bash(`git reset --hard ${exp1.slice(0, 8)}`),
	bash(`git reset --hard ${exp2.slice(0, 8)}`),
	bash("git reset --hard autoresearch/run"),
	bash("git reset HEAD train.py"),
	bash("git checkout -- train.py"),
	bash("git checkout train.py"),
	bash(`git checkout ${base.slice(0, 8)} -- train.py`),
	bash("git show HEAD~1:train.py | head -40"),
	bash("git branch --show-current && git branch -a"),
	bash("git stash && git stash pop"),
	bash("git clean -fd"),
	bash("nvidia-smi"),
	bash("nvidia-smi --query-gpu=index,memory.used --format=csv"),
	bash("cat prepare.py | head -50"),
	bash("grep -n EVAL_TOKENS prepare.py"),
	bash("uv run python -c 'from prepare import MAX_SEQ_LEN; print(MAX_SEQ_LEN)'"),
	bash("sed -i 's/DEPTH = 8/DEPTH = 10/' train.py"),
	bash("sed -i 's/WARMUP_STEPS = 1000/WARMUP_STEPS = 8000/' train.py"),
	bash("python -c 'import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))'"),
	bash("cat > /tmp/notes.md <<'EOF'\nidea: git push is not needed; kill nothing\nEOF"),
	bash("echo -e 'abc1234\\t1.0791\\t45.0\\tkeep\\tbaseline' >> results.tsv"),
	bash("ls -la && pwd && df -h ."),
	bash("python -c \"import math; print(math.pi)\""),
	bash("ps aux | head"),
	["edit", { path: "train.py", edits: [{ oldText: "x = 3", newText: "x = 8000  # tokens[:8000]" }] }],
	["write", { path: join(repo, "train.py"), content: "import os\nos.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'\n" }],
	["write", { path: "/tmp/scratch/notes.md", content: "Try lower LR next." }],
	["read", { path: "prepare.py" }],
	["read", { path: "/tmp/x.txt" }],
];

let fail = 0;
for (const [tool, input] of BLOCK) {
	const v = checkToolCall(tool, input, repo);
	if (!v) {
		fail++;
		console.log("FAIL (should block):", tool, JSON.stringify(input));
	}
}
for (const [tool, input] of ALLOW) {
	const v = checkToolCall(tool, input, repo);
	if (v) {
		fail++;
		console.log("FAIL (should allow):", tool, JSON.stringify(input), "->", v.rule);
	}
}

// [tool, input, cwd] cases for the nested production layout, with AR_GUARD_PROTECTED_DIRS=<infra>.
const NESTED_BLOCK = [
	["read", { path: join(infra, "tools", "secret.py") }, nested],
	["read", { path: "../tools/secret.py" }, nested],
	["read", { path: "@" + join(infra, "tools", "secret.py") }, nested],
	["read", { path: "file://" + join(infra, "tools", "secret.py") }, nested],
	["ls", { path: infra }, nested],
	["grep", { pattern: "TOKEN", path: ".." }, nested],
	["write", { path: join(infra, "tools", "secret.py"), content: "x" }, nested],
	["edit", { path: join(nested, "prepare.py"), edits: [{ oldText: "y = 1", newText: "y = 2" }] }, nested],
	bash(`cat ${nested}/../tools/secret.py`).concat(nested),
	bash(`cat ${infra}/tools/secret.py`).concat(nested),
	bash(`ls ${nested}X`).concat(nested),
	bash(`cp /tmp/p.py ${nested}/prepare.py`).concat(nested),
	// pi started in the protected dir itself (not a git repo): no exemption at all
	["read", { path: "tools/secret.py" }, infra],
	["read", { path: "autoresearch/train.py" }, infra],
];
const NESTED_ALLOW = [
	["read", { path: "train.py" }, nested],
	["read", { path: join(nested, "train.py") }, nested],
	["read", { path: "@train.py" }, nested],
	["ls", {}, nested],
	["grep", { pattern: "x", path: "." }, nested],
	["find", { pattern: "*.py" }, nested],
	["edit", { path: "train.py", edits: [{ oldText: "x = 1", newText: "x = 2" }] }, nested],
	["write", { path: join(nested, "train.py"), content: "x = 3\n" }, nested],
	bash(`cd ${nested} && git log --oneline -3`).concat(nested),
	bash(`cat "${nested}/train.py" | head -5`).concat(nested),
	bash('./run.sh "try swiglu" > run.log 2>&1').concat(nested),
	// AR_GUARD_PROTECTED_DIRS replaces the $RLTLDR_ROOT default
	["read", { path: join(ROOT, "pi", "README.md") }, nested],
];
const savedProtected = process.env.AR_GUARD_PROTECTED_DIRS;
process.env.AR_GUARD_PROTECTED_DIRS = infra;
for (const [tool, input, cwd] of NESTED_BLOCK) {
	if (!checkToolCall(tool, input, cwd)) {
		fail++;
		console.log("FAIL (nested layout, should block):", tool, JSON.stringify(input), "cwd", cwd);
	}
}
for (const [tool, input, cwd] of NESTED_ALLOW) {
	const v = checkToolCall(tool, input, cwd);
	if (v) {
		fail++;
		console.log("FAIL (nested layout, should allow):", tool, JSON.stringify(input), "->", v.rule);
	}
}
if (savedProtected === undefined) delete process.env.AR_GUARD_PROTECTED_DIRS;
else process.env.AR_GUARD_PROTECTED_DIRS = savedProtected;

// A protected project root whose name is NOT a blocked keyword: bash text naming it is blocked by path
// (protected-path), while an experiment repo inside it stays usable through its relativized paths.
const plain = mkdtempSync(join(tmpdir(), "guardtest-proj-"));
const plainRepo = join(plain, "work");
mkdirSync(plainRepo);
mkdirSync(join(plain, "tools"));
writeFileSync(join(plain, "tools", "run_client.py"), "x = 1\n");
execFileSync("git", ["init", "-q", "-b", "autoresearch/run"], { cwd: plainRepo });
writeFileSync(join(plainRepo, "train.py"), "x = 1\n");
const PLAIN_BLOCK = [
	bash(`cat ${plain}/tools/run_client.py`),
	bash(`ls ${plain}`),
	bash(`ls "${plain}/"`),
	bash(`cat ${plainRepo}/../tools/run_client.py`),
	["read", { path: join(plain, "tools", "run_client.py") }],
];
const PLAIN_ALLOW = [
	bash(`cat ${plainRepo}/train.py`),
	bash(`cd ${plainRepo} && git status`),
	bash(`ls ${plain}-other`),
	bash("ls /tmp"),
	["read", { path: "train.py" }],
];
process.env.RLTLDR_ROOT = plain;
for (const [tool, input] of PLAIN_BLOCK) {
	if (!checkToolCall(tool, input, plainRepo)) {
		fail++;
		console.log("FAIL (plain-named root, should block):", tool, JSON.stringify(input));
	}
}
for (const [tool, input] of PLAIN_ALLOW) {
	const v = checkToolCall(tool, input, plainRepo);
	if (v) {
		fail++;
		console.log("FAIL (plain-named root, should allow):", tool, JSON.stringify(input), "->", v.rule);
	}
}
// Neither AR_GUARD_PROTECTED_DIRS nor RLTLDR_ROOT set: ~/rltldr is protected.
delete process.env.RLTLDR_ROOT;
const fallbackBlocked = checkToolCall("read", { path: join(HOME, "rltldr", "data", "x.json") }, repo);
if (!fallbackBlocked || fallbackBlocked.rule !== "protected-read") {
	fail++;
	console.log("FAIL: without RLTLDR_ROOT, ~/rltldr should stay protected:", fallbackBlocked);
}
process.env.RLTLDR_ROOT = ROOT;

// AR_GUARD_ALLOW_VENV_EXEC=1 (h2h supervisor): .venv/bin/python may run as a command; nothing else about .venv.
const VENV_ALLOW = [
	bash(".venv/bin/python -c \"import torch; print(torch.__version__)\""),
	bash("./.venv/bin/python -c 'import torch, numpy; print(torch.zeros(2, 3).shape)'"),
	bash(".venv/bin/python3 -c 'print(1)'"),
	bash(".venv/bin/python3.10 -c 'print(1)'"),
	bash(`${repo}/.venv/bin/python -c 'print(1)'`),
	bash(`cd ${repo} && .venv/bin/python -c "import train" 2>&1 | tail -5`),
	bash("timeout 120 .venv/bin/python -c 'import torch' > /tmp/out.txt 2>&1; cat /tmp/out.txt"),
	bash("env OMP_NUM_THREADS=4 .venv/bin/python /tmp/count_params.py"),
	bash("\".venv/bin/python\" -c 'print(2)'"),
	bash("git diff train.py && .venv/bin/python -c 'import ast; ast.parse(open(\"train.py\").read())'"),
	bash("x=$(.venv/bin/python -c 'print(3)'); echo $x"),
];
const VENV_BLOCK = [
	bash("ls .venv/lib"),
	bash("ls .venv/bin/"),
	bash("cat .venv/bin/python"),
	bash("cp /tmp/x .venv/bin/python"),
	bash("echo x > .venv/bin/python"),
	bash("echo x >.venv/bin/python3"),
	bash(".venv/bin/python -c 1 > .venv/bin/python"),
	bash(".venv/bin/pip install einops"),
	bash(".venv/bin/python -m pip install einops"),
	bash(".venv/bin/python -m venv /tmp/v"),
	bash(".venv/bin/python prepare.py"),
	bash(".venv/bin/python -c \"import site; open(site.getsitepackages()[0] + '/x.pth', 'w')\""),
	bash(".venv/bin/python -c \"open('.venv/lib/python3.10/site-packages/sitecustomize.py', 'w')\""),
	bash(".venv/bin/python .venv/bin/python"),
	bash("bash -c '.venv/bin/python -c 1'"),
	bash("P=.venv/bin/python; $P -c 1"),
	bash(".venv/bin/python - <<'EOF'\nprint(open('.venv/pyvenv.cfg').read())\nEOF"),
	bash(".venv/bin/python -c 'import os; os.kill(1, 9)'"),
	bash(".venv/bin/python -c \"import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0'\""),
	bash(".venv/bin/python -c 1 &"),
	bash("VIRTUAL_ENV=.venv .venv/bin/python -c 1"),
];
const savedVenv = process.env.AR_GUARD_ALLOW_VENV_EXEC;
process.env.AR_GUARD_ALLOW_VENV_EXEC = "1";
for (const [tool, input] of VENV_BLOCK) {
	if (!checkToolCall(tool, input, repo)) {
		fail++;
		console.log("FAIL (venv exec allowed, should still block):", tool, JSON.stringify(input));
	}
}
for (const [tool, input] of VENV_ALLOW) {
	const v = checkToolCall(tool, input, repo);
	if (v) {
		fail++;
		console.log("FAIL (venv exec allowed, should allow):", tool, JSON.stringify(input), "->", v.rule);
	}
}
const venvMsg = checkToolCall("bash", { command: "ls .venv/lib" }, repo);
if (!venvMsg || !venvMsg.reason.includes("You may run `.venv/bin/python`")) {
	fail++;
	console.log("FAIL: the venv block reason should say that .venv/bin/python may be run:", venvMsg?.reason);
}
// the rest of the guard is unchanged with the flag on
for (const [tool, input] of BLOCK) {
	if (!checkToolCall(tool, input, repo) && !(tool === "bash" && /^\.venv\/bin\/python /.test(input.command))) {
		fail++;
		console.log("FAIL (flag on, should still block):", tool, JSON.stringify(input));
	}
}
for (const [tool, input] of ALLOW) {
	if (checkToolCall(tool, input, repo)) {
		fail++;
		console.log("FAIL (flag on, should still allow):", tool, JSON.stringify(input));
	}
}
if (savedVenv === undefined) delete process.env.AR_GUARD_ALLOW_VENV_EXEC;
else process.env.AR_GUARD_ALLOW_VENV_EXEC = savedVenv;

rmSync(repo, { recursive: true, force: true });
rmSync(infra, { recursive: true, force: true });
rmSync(plain, { recursive: true, force: true });
rmSync(tmpLink, { force: true });
if (savedRoot === undefined) delete process.env.RLTLDR_ROOT;
else process.env.RLTLDR_ROOT = savedRoot;
const nBlock = BLOCK.length + NESTED_BLOCK.length + PLAIN_BLOCK.length + 1 + VENV_BLOCK.length;
const nAllow = ALLOW.length + NESTED_ALLOW.length + PLAIN_ALLOW.length + VENV_ALLOW.length;
const total = nBlock + nAllow + 1 + BLOCK.length + ALLOW.length;
console.log(`guard unit tests: ${total - fail}/${total} passed (${nBlock} block cases, ${nAllow} allow cases; ${NESTED_BLOCK.length + NESTED_ALLOW.length} in the nested production layout, ${PLAIN_BLOCK.length + PLAIN_ALLOW.length + 1} for the protected-dir default, ${VENV_BLOCK.length + VENV_ALLOW.length} + ${BLOCK.length + ALLOW.length + 1} with AR_GUARD_ALLOW_VENV_EXEC=1)`);
process.exit(fail ? 1 : 0);
