#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Summarize a py-spy speedscope file: GIL share, GC share, top Python frames, GC triggers.

Example:
    py-spy record --gil --idle --native -r 100 -d 120 --format speedscope -o c.json -p <pid>
    python speedscope_summary.py c.json 100 120
"""

import collections
import json
import sys

GC_FRAMES = ("gc_collect", "collect_with_callback", "deduce_unreachable", "subtract_refs")

data = json.load(open(sys.argv[1]))
frames = data["shared"]["frames"]
names = [f"{f['name']} ({f.get('file', '').rsplit('/', 1)[-1]}:{f.get('line', '')})" for f in frames]
is_gc = [any(g in f["name"] for g in GC_FRAMES) for f in frames]
is_python = [f.get("file", "").endswith(".py") for f in frames]

total, gc_total = 0, 0
inclusive, gc_trigger = collections.Counter(), collections.Counter()
for profile in data["profiles"]:
    for stack, weight in zip(profile["samples"], profile["weights"]):
        total += weight
        for idx in set(stack):
            inclusive[idx] += weight
        gc_pos = next((i for i, idx in enumerate(stack) if is_gc[idx]), None)
        if gc_pos is not None:
            gc_total += weight
            # The innermost Python frame outside the GC frames is what allocated.
            py = [stack[i] for i in range(gc_pos) if is_python[stack[i]]]
            gc_trigger[names[py[-1]] if py else "<no python frame>"] += weight

if len(sys.argv) == 4:
    # With --gil, py-spy only samples threads holding the GIL.
    rate, seconds = float(sys.argv[2]), float(sys.argv[3])
    print(f"GIL held: {total / (rate * seconds):.0%}")
print(f"samples {total}; in GC {gc_total / max(total, 1):.1%}")
print("\n## top inclusive Python frames")
top = sorted((i for i in inclusive if is_python[i]), key=lambda i: -inclusive[i])[:30]
for idx in top:
    print(f"{inclusive[idx] / total:6.1%}  {names[idx]}")
print("\n## Python frame that triggered GC (share of GC samples)")
for name, weight in gc_trigger.most_common(15):
    print(f"{weight / max(gc_total, 1):6.1%}  {name}")
