"""Tool-free Claude completions in Modal containers, so a sweep never loads the local machine.

    uv run modal deploy -m bioprot.modal_claude

Credential: the Modal Secret `claude-auth` (CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY).
"""

from __future__ import annotations

import modal

app = modal.App("bioprot-claude")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates", "gnupg")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g @anthropic-ai/claude-code",
        "claude --version",
    )
    .env({"DISABLE_AUTOUPDATER": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"})
    .add_local_python_source("bioprot")
)


@app.function(image=image, secrets=[modal.Secret.from_name("claude-auth")], timeout=600,
              max_containers=16, cpu=1.0, memory=1024)
def complete(payload: dict) -> dict:
    from bioprot.backends import run_claude_cli

    return run_claude_cli(payload["messages"], payload["model"])


@app.local_entrypoint()
def main(model: str = "haiku"):
    print(complete.remote({"messages": [{"role": "user", "content": "Reply with exactly: OK"}], "model": model}))
