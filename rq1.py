"""RQ1 (paper Sec. 5.1, Table 1/2): are all three operations necessary?

7 scenarios x 2 worlds (KitchenWorld, LavaLand) x 5 seeds x 4 methods.
Each method gets a misaligned starting feature set + one correction, returns a new
feature set; we learn theta from simulated demos and report normalized reward.

Usage:
  python rq1.py                                  # everything
  python rq1.py --methods FLEX --seeds 0 --scenarios "All three"   # quick test
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import llm
from baselines import METHODS
from flex import hand_feature, names
from gridworld import make_env, all_trajectories, to_flat
from irl import true_rewards, sample_demos, learn_theta, plan, normalized_reward

# Per world: correction, and which object plays the role of stove / kettle / fridge.
WORLDS = {
    "kitchen": {"instruction": "Navigate to the goal.", "correction": "Don't burn yourself on the stove.",
                "target": "stove", "sibling": "kettle", "distractor": "fridge"},
    "lava":    {"instruction": "Navigate to the goal.", "correction": "Don't walk into the lava.",
                "target": "lava", "sibling": "swamp", "distractor": "meadow"},
}

# Table 2. "sibling" = the second object of the same category (kettle / swamp):
# it is only in the scene when GENERALIZE is needed.
# NOTE: the paper lists the Add+Generalize start as "goal, fridge" (same as All Three);
# we use "goal" so the scenario needs exactly Add+Generalize (DEVIATIONS.md).
SCENARIOS = {
    "Add only":   {"start": ["goal"],                           "sibling": False},
    "Gen only":   {"start": ["goal", "target"],                 "sibling": True},
    "Mask only":  {"start": ["goal", "target", "distractor"],   "sibling": False},
    "Add+Gen":    {"start": ["goal"],                           "sibling": True},
    "Add+Mask":   {"start": ["goal", "distractor"],             "sibling": False},
    "Mask+Gen":   {"start": ["goal", "target", "distractor"],   "sibling": True},
    "All three":  {"start": ["goal", "distractor"],             "sibling": True},
}

_phi_cache = {}


def feature_column(feature, env, trajs):
    """Feature values for all paths, cached by (feature code, environment)."""
    key = hashlib.sha256((feature["code"] + json.dumps(env, sort_keys=True, default=str)).encode()).hexdigest()
    if key not in _phi_cache:
        flat = [to_flat(t) for t in trajs]
        _phi_cache[key] = np.array([float(np.asarray(feature["fn"](env, tr)).reshape(-1)[0]) for tr in flat])
    return _phi_cache[key]


def score(features, env, trajs, true_r, seed):
    Phi = np.stack([feature_column(f, env, trajs) for f in features], axis=1)
    demos = sample_demos(true_r, seed=seed)
    theta, X = learn_theta(Phi, demos)
    return normalized_reward(true_r, plan(X, theta))


def start_features(scenario, world):
    out = []
    for role in SCENARIOS[scenario]["start"]:
        obj = "goal" if role == "goal" else world[role]
        out.append(hand_feature(f"distance to {obj}", obj, np.inf if obj == "goal" else 3.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--domains", nargs="+", default=list(WORLDS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--out", default="results/rq1")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(exist_ok=True)
    rows, t0 = [], time.time()
    for domain in args.domains:
        world = WORLDS[domain]
        for scenario in args.scenarios:
            exclude = () if SCENARIOS[scenario]["sibling"] else (world["sibling"],)
            for seed in args.seeds:
                env = make_env(domain, seed, exclude=exclude)
                trajs = all_trajectories(env)
                true_r = true_rewards(domain, env, trajs)
                start = start_features(scenario, world)
                before = score(start, env, trajs, true_r, seed)
                for method in args.methods:
                    try:
                        feats, log = METHODS[method](start, env, trajs[0], world["instruction"], [],
                                                     world["correction"])
                        rhat = score(feats, env, trajs, true_r, seed)
                    except Exception as e:  # keep the run going; record the failure
                        feats, log, rhat = start, [("ERROR", f"{type(e).__name__}: {e}")], float("nan")
                    rows.append({"domain": domain, "scenario": scenario, "seed": seed, "method": method,
                                 "before": before, "rhat": rhat, "start": names(start),
                                 "final": names(feats), "log": log})
                    print(f"{domain:7s} | {scenario:9s} | seed {seed} | {method:5s} | "
                          f"R_hat {before:.2f} -> {rhat:.2f} | {names(feats)}")
                Path(args.out + ".json").write_text(json.dumps(rows, indent=1, default=str))

    # ----- Table 1 style summary: mean ± SE over worlds x seeds
    lines = ["| Method | " + " | ".join(args.scenarios) + " |",
             "|---|" + "---|" * len(args.scenarios)]
    for method in args.methods:
        cells = []
        for sc in args.scenarios:
            v = np.array([r["rhat"] for r in rows if r["method"] == method and r["scenario"] == sc])
            v = v[~np.isnan(v)]
            se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            cells.append(f"{v.mean():.2f} ± {se:.2f}" if len(v) else "n/a")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    Path(args.out + "_table.md").write_text(table + "\n")
    print("\n" + table)
    print(f"\ntime: {(time.time() - t0) / 60:.1f} min")
    llm.print_usage()


if __name__ == "__main__":
    main()
