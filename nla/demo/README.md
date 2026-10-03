# NLA Explorer demo

Chat with Qwen3.8-27B (or paste raw text), click any token, and the layer-42 NLA explains the
activation there (streamed), with the reconstructor's cosine as a faithfulness score and the
model's own next-token prediction for context.

- URL: https://gereonelvers99--qwen38-nla-demo-web.modal.run/?key=<contents of .demo_key>
  (the key is remembered in the browser after the first visit)
- Modal app `qwen38-nla-demo`, volume `qwen38-nla-demo` — fully separate from training (`qwen38-nla`).
- A CPU `web` function serves the page; the GPU class `Demo` (1× B200) cold-starts in ~95 s on first
  request and scales to zero after 5 min idle. It uses 1 of the workspace's 10 GPUs while warm.
- Checkpoints load from `gereon/qwen3.8-27b-nla-L42`; when a newer revision is pushed, the top bar
  shows "New checkpoint · load" (hot swap, no restart). SFT/RL toggle appears once RL is uploaded.

```
modal run demo/demo_app.py::prefetch   # once: base model -> demo volume (CPU)
modal deploy demo/demo_app.py
modal app stop qwen38-nla-demo         # tear down
```
