/**
 * Experiment guard for autonomous autoresearch runs (RLTL;DR).
 *
 * Hooks pi's `tool_call` event and blocks tool calls that could escape the
 * experiment sandbox: changing GPU selection, killing processes, installing
 * packages, touching the inference server / gateway / policy trainer, editing
 * the venv or prepare.py, rewriting git history outside the current branch,
 * planting pi project resources, or modifying any repo file except train.py.
 * A blocked call becomes an error tool result whose text is the `reason`, so
 * the model sees why and can adapt.
 *
 * It also answers pi's `project_trust` event with "no": project-local `.pi/`
 * resources live in the agent-writable repo and must never be loaded.
 *
 * This is one layer of defence in depth, not a sandbox: same-uid code can
 * still do anything (see the GPU watchdog / venv sitecustomize / run wrapper).
 *
 * Configuration (environment of the pi process, all optional):
 *   AR_GUARD_REPO            repo root (default: git toplevel of pi's cwd, else cwd)
 *   AR_GUARD_WRITABLE        comma-separated repo-relative files edit/write may touch (default "train.py")
 *   AR_GUARD_SCRATCH_DIRS    comma-separated dirs outside the repo edit/write may touch (default "/tmp")
 *   AR_GUARD_PROTECTED_DIRS  comma-separated dirs no tool may read or write (default: $RLTLDR_ROOT, the project
 *                            root, which the harness passes to pi; "~/rltldr" if that is unset too); the experiment
 *                            repo itself is exempt when it lies strictly inside one of them. bash text that names a
 *                            protected dir (after rewriting absolute repo paths, see relativizeRepoPaths) is blocked
 *   AR_GUARD_LOG             JSONL file that receives one line per blocked call
 *   AR_GUARD_ALLOW_VENV_EXEC "1": the repo's `.venv/bin/python` may be run as the command of a bash call (CPU-only
 *                            checks); every other mention of .venv / site-packages stays blocked (default: off)
 * Every block is also recorded in the session as a custom entry `ar-guard` (not sent to the model).
 */
import { execFileSync } from "node:child_process";
import { appendFileSync, existsSync, realpathSync } from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export interface Verdict {
	block: true;
	rule: string;
	reason: string;
}

interface Rule {
	id: string;
	re: RegExp;
	reason: string;
}

const envList = (name: string, fallback: string): string[] =>
	(process.env[name] ?? fallback)
		.split(",")
		.map((s) => s.trim())
		.filter(Boolean);

// A command word: not preceded or followed by characters that would make it part of a longer token.
const word = (alt: string) => `(?<![\\w.-])(?:${alt})(?![\\w.-])`;

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

// The invoking user's model weights and python envs (~/models, ~/envs), as an absolute path in the regexes below.
const HOME_DIR = homedir().replace(/\/+$/, "");
const HOME_INFRA = HOME_DIR ? `${escapeRe(HOME_DIR)}/(?:models|envs)\\b` : "(?!)";

// ---------------------------------------------------------------------------------------------
// Rules on the raw bash command text (quoted strings, heredocs and `python -c` code included).
// ---------------------------------------------------------------------------------------------
const BASH_RULES: Rule[] = [
	{
		id: "gpu-env",
		re: /CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES|CUDA_DEVICE_ORDER/i,
		reason:
			"GPU selection is fixed by the harness. Do not read, set or unset CUDA_VISIBLE_DEVICES, NVIDIA_VISIBLE_DEVICES or CUDA_DEVICE_ORDER; the training GPU is already selected for every command.",
	},
	{
		id: "nvidia-smi-mutation",
		re: new RegExp(
			String.raw`\bnvidia-smi\b[^\n;&|]*?(?:\s(?:-r|--gpu-reset|-pm|--persistence-mode|-pl|--power-limit|-c|--compute-mode|-e|--ecc-config|-p|--reset-ecc-errors|-ac|--applications-clocks|-rac|--reset-applications-clocks|-lgc|--lock-gpu-clocks|-rgc|--reset-gpu-clocks|-lmc|--lock-memory-clocks|-rmc|--reset-memory-clocks|-mig|--multi-instance-gpu|-am|--accounting-mode|-caa|--clear-accounted-apps|-cc|--cuda-clocks|--auto-boost-default|--auto-boost-permission)(?=[\s=]|$)|\s(?:drain|mig|compute-policy|boost-slider|power-hint|conf-compute|power-smoothing|clocks)\b)`,
		),
		reason: "nvidia-smi may only be used to query GPU state; changing GPU settings is not allowed.",
	},
	{
		id: "kill",
		re: new RegExp(
			`${word("pkill|killall|killall5|xkill|tkill|fuser")}|os\\.kill|os\\.killpg|pthread_kill|psutil|["']kill["']`,
		),
		reason:
			"Do not signal or kill processes: other jobs on this machine (inference server, trainer) are off-limits. Let your own commands finish, or pass the bash tool's `timeout` parameter to bound a command.",
	},
	{
		id: "package-install",
		re: new RegExp(
			[
				`${word("pip3?|uv|conda|mamba|micromamba|npm|pnpm|yarn|poetry|apt|apt-get|easy_install")}[^\\n;&|]*?${word("install|add|remove|uninstall|sync|lock|upgrade|update|venv|create|tool")}`,
				word("npx|uvx|pipx|bunx"),
				String.raw`\buv\s+(?:[^\n;&|]*\s)?(?:pip|--with|--with-requirements|--with-editable|--upgrade|--reinstall|--refresh)\b`,
				String.raw`-m\s+(?:pip|ensurepip|venv|virtualenv)\b`,
			].join("|"),
		),
		reason:
			"The Python environment is fixed: do not install, add, remove or sync packages (only what is already in the project's dependencies can be used). Run experiments only with ./run.sh \"<description>\" (this shell has no GPU).",
	},
	{
		id: "privilege",
		re: new RegExp(word("sudo|doas|pkexec|chroot|unshare|nsenter")),
		reason: "Privilege escalation and namespace tools are not allowed.",
	},
	{
		id: "python-startup-hooks",
		re: /sitecustomize|usercustomize|PYTHONSTARTUP|PYTHONPATH|LD_PRELOAD|LD_LIBRARY_PATH|\.pth(?![\w])/,
		reason:
			"Python/loader startup hooks (sitecustomize, usercustomize, .pth files, PYTHONSTARTUP, PYTHONPATH, LD_PRELOAD, LD_LIBRARY_PATH) are managed by the harness and must not be touched.",
	},
	{
		id: "venv",
		re: /(?<![\w])\.venv(?![\w])|site-packages|dist-packages|VIRTUAL_ENV/,
		reason:
			"The project virtualenv is managed by the harness: do not inspect or modify .venv or site-packages. Run experiments only with ./run.sh \"<description>\" (this shell has no GPU).",
	},
	{
		id: "prepare-py",
		// prepare.py mentioned together with anything that could write it, or executed.
		re: new RegExp(
			[
				String.raw`>\s*["']?[^\s;&|"']*prepare\.py`,
				String.raw`prepare\.py[\s\S]*${word("sed|perl|tee|cp|mv|rm|ln|chmod|chown|truncate|dd|patch|install|rsync|touch|unlink|shred")}`,
				String.raw`${word("sed|perl|tee|cp|mv|rm|ln|chmod|chown|truncate|dd|patch|install|rsync|touch|unlink|shred")}[\s\S]*prepare\.py`,
				String.raw`\bgit\s+(?:apply|am|checkout|restore|stash|mv|rm|update-index|add)\b[^\n;&|]*prepare\.py`,
				String.raw`prepare\.py[\s\S]*(?:write_text|write_bytes|shutil|os\.(?:remove|rename|replace|unlink|chmod))|open\([^)]*prepare\.py[^)]*["'][wax+]`,
				String.raw`${word("python3?|uv\\s+run|bash|sh|exec")}[^\n;&|]*?(?<![\w-])prepare\.py`,
			].join("|"),
		),
		reason:
			"prepare.py is read-only (fixed evaluation, data and tokenizer, already prepared): do not modify, move, delete or run it. You may read it (e.g. with the read tool). If you only meant to read it, run that read as a separate command.",
	},
	{
		id: "infrastructure",
		re: new RegExp(
			[
				"vllm",
				"sglang",
				"rltldr",
				word("trainer"),
				String.raw`policy[\s_-]*(?:adapter|lora|trainer|server|weights|checkpoint)s?`,
				String.raw`lora[_-]?(?:adapter|path|name|module)`,
				"load_lora|unload_lora|update_weights|weight_transfer",
				HOME_INFRA,
				"PI_CODING_AGENT_DIR",
				String.raw`(?<![\w.])\.pi(?=/|\s|$|["'])`,
			].join("|"),
			"i",
		),
		reason:
			"The inference server, gateway, policy trainer, model weights, adapters and agent configuration are off-limits to experiments. Work only inside the experiment repo.",
	},
	{
		id: "local-network",
		re: /localhost|127\.\d+\.\d+\.\d+|0\.0\.0\.0|\[::1?\](?=[:/])|\/dev\/(?:tcp|udp)\/|(?:[:/]|\bport\b\W{0,3})(?:8000|8100|8199)(?!\d)/i,
		reason: "Network access to local services (inference server, gateway) is not allowed.",
	},
	{
		id: "network-clients",
		re: new RegExp(
			`${word("curl|wget|nc|ncat|netcat|socat|telnet|ssh|scp|sftp|ftp|aria2c")}|\\brequests\\.(?:get|post|put|delete|request|Session)|urllib\\.request|http\\.client|\\bsocket\\.socket|\\bhttpx\\b|\\baiohttp\\b`,
		),
		reason: "Network clients are not needed for experiments and are not allowed.",
	},
	{
		id: "background",
		re: new RegExp(word("nohup|setsid|disown|tmux|crontab|systemd-run|daemonize")),
		reason:
			"Run commands in the foreground and wait for them (the bash tool has no default timeout). Background or detached processes are not allowed.",
	},
	{
		id: "pi-context-files",
		re: /AGENTS(?:\.override)?\.md|CLAUDE\.md|SYSTEM\.md|APPEND_SYSTEM\.md/i,
		reason: "Agent context/config files (AGENTS.md, CLAUDE.md, SYSTEM.md) must not be created or edited.",
	},
	{
		id: "git-global",
		re: /\bgit\s+(?:[^\n;&|]*\s)?config\s+(?:[^\n;&|]*\s)?--(?:global|system)\b|\bgit\s+(?:[^\n;&|]*\s)?--(?:git-dir|work-tree)\b/,
		reason: "Only repository-local git operations in the current repo are allowed (no --global/--system config, --git-dir or --work-tree).",
	},
];

// Ambiguous English words are only blocked when they are the command being run (argv[0] of a simple command).
const COMMAND_WORD_RULES: Record<string, Rule> = Object.fromEntries(
	[
		["kill", "kill"],
		["skill", "kill"],
		["su", "privilege"],
		["at", "background"],
		["batch", "background"],
		["screen", "background"],
	].map(([cmd, id]) => [cmd, BASH_RULES.find((r) => r.id === id)!]),
);

// Rules applied to text the agent writes into files that could later be executed (train.py, /tmp scripts).
// Narrower than BASH_RULES so that ordinary training code (e.g. `x[:8000]`, `WARMUP = 8000`) is never blocked.
const CONTENT_RULE_IDS = new Set(["gpu-env", "python-startup-hooks", "venv"]);
const CONTENT_EXTRA_RULES: Rule[] = [
	{
		id: "content-network",
		re: /localhost|127\.0\.0\.1|0\.0\.0\.0|\/dev\/(?:tcp|udp)\/|\brequests\.(?:get|post|put|delete|request|Session)|urllib\.request|http\.client|\bsocket\.(?:socket|create_connection)|\bhttpx\b|\baiohttp\b|["'](?:curl|wget|nc|ncat|ssh|scp)["']/,
		reason: "Code that opens network connections is not allowed.",
	},
	{
		id: "content-kill",
		re: /os\.kill|os\.killpg|pthread_kill|psutil|\bpkill\b|\bkillall\b/,
		reason: "Code that signals or kills processes is not allowed.",
	},
	{
		id: "content-infrastructure",
		re: new RegExp(`vllm|sglang|rltldr|load_lora|update_weights|PI_CODING_AGENT_DIR|${HOME_INFRA}`, "i"),
		reason: "Code that touches the inference server, trainer, model weights or agent configuration is not allowed.",
	},
	{
		id: "content-install",
		re: /\bpip3?\s+install\b|\buv\s+(?:pip|add|sync|remove)\b|-m\s+pip\b|ensurepip/,
		reason: "Installing packages from code is not allowed; the environment is fixed.",
	},
	{
		id: "content-prepare-write",
		re: /open\([^)]*prepare\.py[^)]*["'][wax+]|prepare\.py["']?\s*\)\s*\.write_(?:text|bytes)/,
		reason: "prepare.py is read-only.",
	},
];
const CONTENT_RULES: Rule[] = [...BASH_RULES.filter((r) => CONTENT_RULE_IDS.has(r.id)), ...CONTENT_EXTRA_RULES];

// ---------------------------------------------------------------------------------------------
// AR_GUARD_ALLOW_VENV_EXEC=1: the repo's `.venv/bin/python` may be run (e.g. for CPU-only shape checks).
// The venv-python mentions are exempt from the `venv` rule only if every one of them in the raw text is the
// command word of a simple command (after env/timeout/... wrappers): not an argument (cat/cp/tee
// .venv/bin/python), not a redirection target, not inside a heredoc or a nested `bash -c`. Exempt mentions are
// shown to the keyword rules as plain `python`, so e.g. `-m pip`, `prepare.py` and `site-packages` stay blocked.
// ---------------------------------------------------------------------------------------------
const VENV_PY = String.raw`(?:\./)?\.venv/bin/python(?:3(?:\.\d+)?)?`;
const VENV_PY_TOKEN = new RegExp(String.raw`(?<![\w./-])(?<![<>]\s*)${VENV_PY}(?![\w./-])`, "g");
const VENV_PY_WORD = new RegExp(`^${VENV_PY}$`);
const venvExecAllowed = () => process.env.AR_GUARD_ALLOW_VENV_EXEC === "1";
const VENV_EXEC_REASON =
	"The project virtualenv is managed by the harness: do not inspect or modify .venv or site-packages. You may run `.venv/bin/python` itself as the command (CPU only, e.g. `.venv/bin/python -c \"...\"` for parameter counts or tensor shapes). Run experiments only with ./run.sh \"<description>\" (this shell has no GPU).";

/** The text the keyword rules see: the command, with exempt venv-python command words replaced by `python`. */
function maskVenvExec(command: string, segments: string[][]): string {
	if (!venvExecAllowed()) return command;
	const tokens = command.match(VENV_PY_TOKEN)?.length ?? 0;
	if (!tokens) return command;
	let asCommand = 0;
	let words = 0;
	for (const seg of segments) {
		words += seg.filter((w) => VENV_PY_WORD.test(w)).length;
		const argv = effectiveArgv(seg);
		if (argv.length && VENV_PY_WORD.test(argv[0])) asCommand++;
	}
	return tokens === asCommand && words === asCommand ? command.replace(VENV_PY_TOKEN, "python") : command;
}

const ruleReason = (r: Rule): string => (r.id === "venv" && venvExecAllowed() ? VENV_EXEC_REASON : r.reason);

// ---------------------------------------------------------------------------------------------
// Minimal POSIX-shell splitter: simple commands (word lists) separated by ; & && || | newlines,
// parentheses and command substitutions; quotes and escapes respected; redirections dropped.
// Only used for structural git checks; keyword rules above run on the raw text.
// ---------------------------------------------------------------------------------------------
export function splitShell(cmd: string): { segments: string[][]; background: boolean } {
	const segments: string[][] = [];
	let words: string[] = [];
	let cur = "";
	let inWord = false;
	let skipNextWord = false; // the next word is a redirection target
	let background = false;
	const flushWord = () => {
		if (inWord) {
			if (skipNextWord) skipNextWord = false;
			else words.push(cur);
		}
		cur = "";
		inWord = false;
	};
	const flushSegment = () => {
		flushWord();
		if (words.length) segments.push(words);
		words = [];
	};
	let i = 0;
	const n = cmd.length;
	while (i < n) {
		const c = cmd[i];
		if (c === "\\" && i + 1 < n) {
			if (cmd[i + 1] !== "\n") {
				cur += cmd[i + 1];
				inWord = true;
			}
			i += 2;
			continue;
		}
		if (c === "'") {
			const j = cmd.indexOf("'", i + 1);
			const end = j === -1 ? n : j;
			cur += cmd.slice(i + 1, end);
			inWord = true;
			i = end + 1;
			continue;
		}
		if (c === '"') {
			i++;
			inWord = true;
			while (i < n && cmd[i] !== '"') {
				if (cmd[i] === "\\" && i + 1 < n && '"\\$`\n'.includes(cmd[i + 1])) {
					if (cmd[i + 1] !== "\n") cur += cmd[i + 1];
					i += 2;
				} else {
					cur += cmd[i++];
				}
			}
			i++;
			continue;
		}
		if (c === " " || c === "\t") {
			flushWord();
			i++;
			continue;
		}
		if (c === "#" && !inWord) {
			// comment to end of line
			while (i < n && cmd[i] !== "\n") i++;
			continue;
		}
		if (c === "$" && cmd[i + 1] === "{") {
			// ${VAR...} parameter expansion stays part of the word
			const j = cmd.indexOf("}", i + 2);
			const end = j === -1 ? n : j + 1;
			cur += cmd.slice(i, end);
			inWord = true;
			i = end;
			continue;
		}
		if (c === "<" && cmd[i + 1] === "<" && cmd[i + 2] !== "<") {
			// heredoc: skip its body (keyword rules already saw the raw text)
			flushWord();
			i += 2;
			const stripTabs = cmd[i] === "-";
			if (stripTabs) i++;
			while (cmd[i] === " " || cmd[i] === "\t") i++;
			let delim = "";
			while (i < n && !/[\s;&|<>()]/.test(cmd[i])) {
				if (cmd[i] !== "'" && cmd[i] !== '"' && cmd[i] !== "\\") delim += cmd[i];
				i++;
			}
			// the rest of the current line still belongs to the command
			const eol = cmd.indexOf("\n", i);
			const rest = eol === -1 ? cmd.slice(i) : cmd.slice(i, eol);
			const inner = splitShell(rest);
			if (inner.background) background = true;
			if (inner.segments.length) {
				words.push(...inner.segments[0]);
				flushSegment();
				segments.push(...inner.segments.slice(1));
			}
			if (eol === -1) {
				i = n;
				continue;
			}
			// skip body lines up to the terminating delimiter line
			let k = eol + 1;
			while (k < n) {
				const lineEnd = cmd.indexOf("\n", k);
				const line = cmd.slice(k, lineEnd === -1 ? n : lineEnd);
				k = lineEnd === -1 ? n : lineEnd + 1;
				if ((stripTabs ? line.replace(/^\t+/, "") : line) === delim) break;
			}
			flushSegment();
			i = k;
			continue;
		}
		if (c === ">" || c === "<") {
			// fd prefix such as 2>; drop it
			if (inWord && /^\d+$/.test(cur)) {
				cur = "";
				inWord = false;
			} else flushWord();
			i++;
			while (i < n && (cmd[i] === ">" || cmd[i] === "<" || cmd[i] === "|")) i++;
			if (cmd[i] === "&") {
				i++;
				while (i < n && /[\d-]/.test(cmd[i])) i++;
			} else if (cmd[i] === "(") {
				// process substitution <( ... ): parse the inside as its own command
				flushSegment();
				i++;
			} else {
				skipNextWord = true;
			}
			continue;
		}
		if (c === "&") {
			if (cmd[i + 1] === "&") {
				flushSegment();
				i += 2;
			} else if (cmd[i + 1] === ">") {
				// &> / &>> redirect of stdout+stderr
				flushWord();
				i += 2;
				if (cmd[i] === ">") i++;
				skipNextWord = true;
			} else {
				background = true;
				flushSegment();
				i++;
			}
			continue;
		}
		if (c === "$" && cmd[i + 1] === "(") {
			flushSegment();
			i += 2;
			continue;
		}
		if (c === ";" || c === "\n" || c === "|" || c === "(" || c === ")" || c === "`" || ((c === "{" || c === "}") && !inWord)) {
			flushSegment();
			i++;
			continue;
		}
		cur += c;
		inWord = true;
		i++;
	}
	flushSegment();
	return { segments, background };
}

// Wrappers whose arguments end with a command we want to inspect.
const WRAPPERS = new Set(["env", "command", "builtin", "exec", "time", "nice", "ionice", "timeout", "stdbuf", "xargs", "taskset", "chrt", "caffeinate", "then", "do", "else", "if", "while", "until", "!"]);

/** Strip env assignments and wrapper commands (with their options), returning the effective argv. */
export function effectiveArgv(words: string[]): string[] {
	let i = 0;
	while (i < words.length) {
		const w = words[i];
		if (/^[A-Za-z_][A-Za-z0-9_]*=/.test(w)) {
			i++;
			continue;
		}
		if (WRAPPERS.has(basename(w))) {
			i++;
			// options and numeric/duration/cpu-list arguments of the wrapper
			while (i < words.length && (/^-/.test(words[i]) || /^[\d.,:-]+[smhd]?$/.test(words[i]) || /^[A-Za-z_][A-Za-z0-9_]*=/.test(words[i]))) i++;
			continue;
		}
		break;
	}
	return words.slice(i);
}

// ---------------------------------------------------------------------------------------------
// git structural checks
// ---------------------------------------------------------------------------------------------
function git(cwd: string, args: string[]): { ok: boolean; out: string } {
	try {
		const out = execFileSync("git", args, { cwd, encoding: "utf8", timeout: 5000, stdio: ["ignore", "pipe", "ignore"] });
		return { ok: true, out: out.trim() };
	} catch {
		return { ok: false, out: "" };
	}
}

const HEAD_RELATIVE = /^(?:HEAD|@|ORIG_HEAD)(?:[~^@].*)?$/;

function refKind(cwd: string, name: string): "branch" | "remote" | "tag" | null {
	if (git(cwd, ["show-ref", "--verify", "--quiet", `refs/heads/${name}`]).ok) return "branch";
	if (git(cwd, ["show-ref", "--verify", "--quiet", `refs/remotes/${name}`]).ok) return "remote";
	if (git(cwd, ["show-ref", "--verify", "--quiet", `refs/tags/${name}`]).ok) return "tag";
	return null;
}

/** Is moving HEAD (reset/checkout) to `target` confined to the current branch's own history? */
function targetOnCurrentBranch(cwd: string, target: string): boolean {
	if (HEAD_RELATIVE.test(target)) return true;
	const current = git(cwd, ["symbolic-ref", "--quiet", "--short", "HEAD"]).out;
	const kind = refKind(cwd, target);
	if (kind) return kind === "branch" && target === current;
	const commit = git(cwd, ["rev-parse", "--verify", "--quiet", `${target}^{commit}`]);
	if (!commit.ok) return true; // not a commit: git itself will fail or treat it as a path
	if (git(cwd, ["merge-base", "--is-ancestor", commit.out, "HEAD"]).ok) return true;
	// A commit that is not an ancestor of HEAD: allowed only if no other branch contains it
	// (e.g. a commit of this branch that was reset away); otherwise it belongs to another branch.
	const holders = git(cwd, ["for-each-ref", "--format=%(refname:short)", "--contains", commit.out, "refs/heads", "refs/remotes"])
		.out.split("\n")
		.filter((b) => b && b !== current);
	return holders.length === 0;
}

const GIT_FORBIDDEN_SUBCOMMANDS: Record<string, string> = {
	push: "git push is not allowed.",
	pull: "git pull is not allowed.",
	fetch: "git fetch is not allowed.",
	rebase: "git rebase is not allowed: keep a linear history on the current branch. To undo your edits use `git checkout -- train.py`; the harness handles keep/discard.",
	merge: "git merge is not allowed: stay on the current branch.",
	"cherry-pick": "git cherry-pick is not allowed: stay on the current branch.",
	switch: "Switching branches is not allowed: stay on the current branch.",
	worktree: "git worktree is not allowed.",
	remote: "git remote is not allowed.",
	"filter-branch": "Rewriting history is not allowed.",
	"filter-repo": "Rewriting history is not allowed.",
	"update-ref": "Direct ref manipulation is not allowed.",
	"symbolic-ref": "Direct ref manipulation is not allowed: stay on the current branch.",
	replace: "git replace is not allowed.",
	submodule: "git submodule is not allowed.",
	clone: "git clone is not allowed.",
};

function checkGit(argv: string[], cwd: string): Verdict | undefined {
	let i = 1;
	let dir = cwd;
	// global options before the subcommand
	while (i < argv.length && argv[i].startsWith("-")) {
		const opt = argv[i];
		if (opt === "-C" && i + 1 < argv.length) {
			dir = resolve(dir, argv[i + 1]);
			i += 2;
		} else if (opt === "-c" && i + 1 < argv.length) i += 2;
		else i++;
	}
	const sub = argv[i];
	if (!sub) return undefined;
	const args = argv.slice(i + 1);
	const block = (reason: string): Verdict => ({ block: true, rule: `git-${sub}`, reason });
	if (sub in GIT_FORBIDDEN_SUBCOMMANDS) return block(GIT_FORBIDDEN_SUBCOMMANDS[sub]);
	if (sub === "reflog" && args.some((a) => a === "expire" || a === "delete")) return block("Deleting reflog entries is not allowed.");
	if (sub === "clean" && args.some((a) => /^-[a-zA-Z]*[xX]/.test(a))) return block("git clean -x/-X would delete ignored files such as the virtualenv; use `git clean -fd` or `git checkout -- <file>`.");
	if (sub === "branch" && args.some((a) => /^-(?:[a-zA-Z]*[dDfmMcC]|-delete|-force|-move|-copy|-set-upstream-to|-unset-upstream)/.test(a)))
		return block("Deleting, moving, copying or force-updating branches is not allowed.");
	if (sub === "reset") {
		// `git reset <commit>` moves the branch tip (any mode); `git reset [<commit>] -- <paths>` / `git reset <path>` only unstage.
		const dd = args.indexOf("--");
		const target = (dd === -1 ? args : args.slice(0, dd)).find((a) => !a.startsWith("-"));
		if (target && !existsSync(resolve(dir, target)) && !targetOnCurrentBranch(dir, target))
			return block(`git reset may only move within the current branch's history, not to ${target}. To undo edits use \`git checkout -- train.py\`; the harness handles keep/discard.`);
	}
	if (sub === "checkout") {
		if (args.some((a) => a === "-" || /^-(?:[a-zA-Z]*[bB]|-orphan|-detach)/.test(a))) return block("Creating or switching branches is not allowed: stay on the current branch.");
		const dd = args.indexOf("--");
		const before = (dd === -1 ? args : args.slice(0, dd)).filter((a) => !a.startsWith("-"));
		for (const a of before) {
			if (existsSync(resolve(dir, a))) continue; // a path: restoring files is fine
			const kind = refKind(dir, a);
			const current = git(dir, ["symbolic-ref", "--quiet", "--short", "HEAD"]).out;
			if (kind && !(kind === "branch" && a === current)) return block(`Checking out ${a} would leave the current branch; restore files with \`git checkout -- <file>\` instead.`);
			if (!kind && dd === -1 && git(dir, ["rev-parse", "--verify", "--quiet", `${a}^{commit}`]).ok && before.length === 1)
				return block(`Checking out commit ${a} would detach HEAD; restore files with \`git checkout ${a} -- <file>\` instead.`);
		}
	}
	return undefined;
}

// ---------------------------------------------------------------------------------------------
// Path checks for read / edit / write
// ---------------------------------------------------------------------------------------------
const UNICODE_SPACES = /[  -   　]/g;

/**
 * Absolute path with symlinks resolved as far as the path exists.
 * The input is first normalized exactly like pi's tools do (utils/paths.js `resolveToCwd`: unicode
 * spaces -> " ", a leading "@" stripped, "~" expanded, file:// URLs decoded); otherwise e.g.
 * `@/abs/path` would be judged as `<cwd>/@/abs/path` while pi opens `/abs/path`.
 */
export function canonicalPath(p: string, cwd: string): string {
	let abs = p.replace(UNICODE_SPACES, " ");
	if (abs.startsWith("@")) abs = abs.slice(1);
	if (abs === "~" || abs.startsWith("~/")) abs = join(homedir(), abs.slice(1));
	else if (/^file:\/\//.test(abs)) {
		try {
			abs = fileURLToPath(abs);
		} catch {
			/* not a valid file URL: pi would fail on it too; judge it lexically */
		}
	}
	abs = isAbsolute(abs) ? resolve(abs) : resolve(cwd, abs);
	let head = abs;
	const tail: string[] = [];
	while (!existsSync(head) && dirname(head) !== head) {
		tail.unshift(basename(head));
		head = dirname(head);
	}
	try {
		head = realpathSync(head);
	} catch {
		/* keep lexical path */
	}
	return tail.length ? join(head, ...tail) : head;
}

const isInside = (child: string, parent: string): boolean => {
	const rel = relative(parent, child);
	return rel === "" || (!rel.startsWith("..") && !isAbsolute(rel));
};

const repoRootCache = new Map<string, string>();
function repoRoot(cwd: string): string {
	if (process.env.AR_GUARD_REPO) return canonicalPath(process.env.AR_GUARD_REPO, cwd);
	let root = repoRootCache.get(cwd);
	if (!root) {
		const top = git(cwd, ["rev-parse", "--show-toplevel"]);
		root = canonicalPath(top.ok && top.out ? top.out : cwd, cwd);
		repoRootCache.set(cwd, root);
	}
	return root;
}

const protectedDirsRaw = (): string[] => envList("AR_GUARD_PROTECTED_DIRS", process.env.RLTLDR_ROOT || "~/rltldr");

function protectedDirs(cwd: string): string[] {
	return protectedDirsRaw().map((d) => canonicalPath(d, cwd));
}

/**
 * Is `abs` (canonical) inside a protected dir? The experiment repo itself is exempt when it lies strictly
 * inside a protected dir (production layout: repo <root>/autoresearch under the protected project root
 * <root>); the rest of that dir (tools, data, pi config, ...) stays protected. A protected dir that is the
 * repo root or contains it from above (e.g. pi started in <root> itself) gets no exemption.
 */
function isProtected(abs: string, cwd: string): boolean {
	const root = repoRoot(cwd);
	return protectedDirs(cwd).some((d) => isInside(abs, d) && !(root !== d && isInside(root, d) && isInside(abs, root)));
}

/**
 * Rewrite absolute paths into the experiment repo as `./...` before the keyword rules run: the repo may
 * live under a directory whose name is itself a blocked keyword (~/rltldr/autoresearch) or that is protected, and
 * pi's system prompt shows the model that absolute cwd. Paths with a `..` segment after the repo root are
 * left untouched (and so stay blocked), as are longer names that merely start with the root path.
 */
function relativizeRepoPaths(command: string, cwd: string): string {
	const root = repoRoot(cwd);
	if (root === "/" || !command.includes(root)) return command;
	const esc = escapeRe(root);
	const re = new RegExp(`(?<=^|[\\s"'\`=:;&|<>()])${esc}(?=$|[/\\s"'\`;&|<>()])([^\\s"'\`;&|<>()]*)`, "g");
	return command.replace(re, (m: string, tail: string) => (tail.split("/").includes("..") ? m : `.${tail}`));
}

/**
 * Does bash text name a protected dir (as configured, or its canonical path)? Checked after relativizeRepoPaths,
 * so paths into an experiment repo that lies inside a protected dir have already become `./...`. On the original
 * layout the keyword rule `rltldr` already covers this; it matters when the project root has another name.
 */
function mentionsProtectedDir(command: string, cwd: string): boolean {
	const spellings = new Set<string>();
	for (const d of protectedDirsRaw()) {
		spellings.add(d.replace(/\/+$/, ""));
		spellings.add(canonicalPath(d, cwd));
	}
	for (const s of spellings) {
		if (!s || s === "/" || !(isAbsolute(s) || s.startsWith("~/"))) continue; // relative entries: path tools only
		if (new RegExp(`(?<![\\w./~-])${escapeRe(s)}(?![\\w.-])`).test(command)) return true;
	}
	return false;
}

function checkContent(text: string, where: string): Verdict | undefined {
	for (const r of CONTENT_RULES) {
		if (r.re.test(text)) return { block: true, rule: r.id, reason: `${r.reason} (found in the text written to ${where})` };
	}
	return undefined;
}

function contentOf(input: Record<string, unknown>): string {
	const parts: string[] = [];
	if (typeof input.content === "string") parts.push(input.content);
	if (typeof input.newText === "string") parts.push(input.newText);
	let edits = input.edits as unknown;
	if (typeof edits === "string") {
		try {
			edits = JSON.parse(edits);
		} catch {
			parts.push(edits as string);
		}
	}
	if (Array.isArray(edits)) for (const e of edits) if (e && typeof e.newText === "string") parts.push(e.newText);
	return parts.join("\n");
}

// ---------------------------------------------------------------------------------------------
// Decision function (exported for tests)
// ---------------------------------------------------------------------------------------------
export function checkBash(rawCommand: string, cwd: string, depth = 0): Verdict | undefined {
	const command = relativizeRepoPaths(rawCommand, cwd);
	const { segments, background } = splitShell(command);
	const ruleText = maskVenvExec(command, segments);
	for (const r of BASH_RULES) if (r.re.test(ruleText)) return { block: true, rule: r.id, reason: ruleReason(r) };
	if (mentionsProtectedDir(command, cwd))
		return { block: true, rule: "protected-path", reason: "That path belongs to the experiment infrastructure and is off-limits. Work only inside the experiment repo." };
	if (background)
		return { block: true, rule: "background", reason: "Run commands in the foreground and wait for them (the bash tool has no default timeout). Background jobs (`&`) are not allowed." };
	for (const seg of segments) {
		const argv = effectiveArgv(seg);
		if (!argv.length) continue;
		const cmd = basename(argv[0]);
		const wordRule = COMMAND_WORD_RULES[cmd];
		if (wordRule) return { block: true, rule: wordRule.id, reason: wordRule.reason };
		if (cmd === "git") {
			const v = checkGit(argv, cwd);
			if (v) return v;
		}
		// nested shells: bash -c "...", sh -c '...', eval ...
		if (depth < 3 && ["bash", "sh", "zsh", "dash", "ksh"].includes(cmd)) {
			const ci = argv.findIndex((a, k) => k > 0 && /^-[a-z]*c[a-z]*$/.test(a));
			if (ci !== -1 && argv[ci + 1]) {
				const v = checkBash(argv[ci + 1], cwd, depth + 1);
				if (v) return v;
			}
		}
		if (depth < 3 && cmd === "eval") {
			const v = checkBash(argv.slice(1).join(" "), cwd, depth + 1);
			if (v) return v;
		}
	}
	return undefined;
}

export function checkToolCall(toolName: string, input: Record<string, unknown>, cwd: string): Verdict | undefined {
	if (toolName === "bash") return checkBash(String(input.command ?? ""), cwd);

	const rawPath = input.path ?? input.file_path;
	const pathArg = typeof rawPath === "string" ? rawPath : undefined;

	if (["read", "grep", "find", "ls"].includes(toolName)) {
		const abs = canonicalPath(pathArg ?? ".", cwd);
		if (isProtected(abs, cwd))
			return { block: true, rule: "protected-read", reason: `${abs} belongs to the experiment infrastructure and is off-limits. Work only inside the experiment repo.` };
		return undefined;
	}

	if (toolName === "edit" || toolName === "write") {
		if (!pathArg) return { block: true, rule: "write-path", reason: "Missing path." };
		const abs = canonicalPath(pathArg, cwd);
		const root = repoRoot(cwd);
		const writable = envList("AR_GUARD_WRITABLE", "train.py").map((f) => canonicalPath(f, root));
		const notAllowed: Verdict = {
			block: true,
			rule: "write-path",
			reason: `Only ${envList("AR_GUARD_WRITABLE", "train.py").join(", ")} in the repo may be modified with ${toolName} (scratch notes may go under ${envList("AR_GUARD_SCRATCH_DIRS", "/tmp").join(", ")}). ${abs} is not writable.`,
		};
		if (isProtected(abs, cwd)) return notAllowed;
		if (isInside(abs, root)) {
			if (!writable.includes(abs)) return notAllowed;
		} else if (!envList("AR_GUARD_SCRATCH_DIRS", "/tmp").some((d) => isInside(abs, canonicalPath(d, cwd)))) {
			return notAllowed;
		}
		return checkContent(contentOf(input), abs);
	}

	return undefined;
}

export default function (pi: ExtensionAPI) {
	// Project-local resources (.pi/ in the repo) are agent-writable: never trust them.
	pi.on("project_trust", () => ({ trusted: "no" }));

	pi.on("tool_call", async (event, ctx) => {
		const verdict = checkToolCall(event.toolName, event.input as Record<string, unknown>, ctx.cwd);
		if (!verdict) return undefined;
		const record = { ts: new Date().toISOString(), tool: event.toolName, toolCallId: event.toolCallId, rule: verdict.rule, input: event.input, cwd: ctx.cwd };
		try {
			pi.appendEntry("ar-guard", record);
		} catch {
			/* recording is best effort */
		}
		const log = process.env.AR_GUARD_LOG;
		if (log) {
			try {
				appendFileSync(log, JSON.stringify(record) + "\n");
			} catch {
				/* best effort */
			}
		}
		return { block: true, reason: `BLOCKED by experiment guard [${verdict.rule}]: ${verdict.reason}` };
	});
}
