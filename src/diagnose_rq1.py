"""Diagnose RQ1: why does FLEX sometimes end with only 'distance to goal'?

1. From results/rq1.json: for every FLEX run that ended with goal only, print its log.
2. From cache/: look at every MASK answer GPT gave, and check whether our parser
   (first 0/1 digit in the answer) agrees with the LAST 0/1 in the answer.
No API calls.
"""
import json
import re
from pathlib import Path

rows = json.loads(Path("results/rq1.json").read_text())
flex = [r for r in rows if r["method"] == "FLEX"]
bad = [r for r in flex if r["final"] == ["distance to goal"]]
print(f"FLEX runs ending with goal only: {len(bad)} / {len(flex)}\n")
for r in bad:
    print(f"{r['domain']:7s} | {r['scenario']:9s} | seed {r['seed']} | start {r['start']}")
    for step, msg in r["log"]:
        if not msg.startswith("intent"):
            print(f"      [{step}] {msg}")

print("\n=== MASK answers in the cache ===")
n = short = disagree = 0
examples = []
for f in Path("cache").glob("*.json"):
    d = json.loads(f.read_text(encoding="utf-8"))
    prompt = d["messages"][-1]["content"]
    if "We are evaluating whether" not in prompt or d["model_requested"] != "gpt-4o":
        continue
    n += 1
    ans = d["response"].strip()
    if ans in ("0", "1"):
        short += 1
        continue
    first = re.search(r"[01]", ans)
    last = re.findall(r"\b[01]\b", ans)
    first = first.group() if first else None
    last = last[-1] if last else None
    if first != last:
        disagree += 1
        if len(examples) < 3:
            feat = prompt.split("feature = ")[1].split(",")[0]
            examples.append((feat, first, last, ans[-300:]))
print(f"MASK answers: {n} | exactly '0' or '1': {short} | longer answers: {n - short} | "
      f"first digit != last 0/1: {disagree}")
for feat, first, last, tail in examples:
    print(f"\n--- feature {feat}: parser read {first}, final answer looks like {last}\n...{tail}")
