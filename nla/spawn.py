"""Spawn a stage on the DEPLOYED app (survives local disconnects).
usage: python spawn.py <function> '<json kwargs>' [gpu]"""
import json
import sys

import modal

fn = modal.Function.from_name("qwen38-nla", sys.argv[1])
kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
if len(sys.argv) > 3:
    fn = fn.with_options(gpu=sys.argv[3])
call = fn.spawn(**kw)
print("spawned", sys.argv[1], kw, "call_id", call.object_id)
