"""RQ1 transfer test: are the learned representations TRANSFERABLE? (paper: FLEX keeps
"compact, transferable representations").

For every RQ1 run (world x scenario x seed x method): the method edits the features in the
training scene (same as rq1.py; LLM answers come from the cache), theta is learned from that
scene's 5 demos, and the learned reward is then used, WITHOUT new demos, in 3 new scenes of
the same world that contain all objects (e.g. stove AND kettle AND fridge).

Planning in the new scenes uses a fixed random subset of 2,000 of the 12,870 paths (speed);
best/worst for the normalized reward are taken over the same subset.

Usage: python rq1_transfer.py [--methods FLEX ALGAE MFE USF FLEX+V] [--seeds ...]
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import llm
from baselines import METHODS
from flex import names
from gridworld import make_env, all_trajectories, to_flat
from irl import true_rewards, sample_demos, learn_theta, plan, normalized_reward, make_mask_verifier
from rq1 import WORLDS, SCENARIOS, start_features, feature_column

TEST_SEEDS = [1000, 1001, 1002]
N_SUBSET = 2000
_cache = {}


def column_subset(feature, env, paths, tag):
    key = hashlib.sha256((feature["code"] + json.dumps(env, sort_keys=True, default=str) + tag).encode()).hexdigest()
    if key not in _cache:
        vals = []
        with np.errstate(all="ignore"):
            for p in paths:
                try:
                    v = float(np.asarray(feature["fn"](env, to_flat(p))).reshape(-1)[0])
                except Exception:
                    v = 0.0
                vals.append(v if np.isfinite(v) else 0.0)
        _cache[key] = np.array(vals)
    return _cache[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["FLEX", "ALGAE", "MFE", "USF", "FLEX+V"])
    ap.add_argument("--domains", nargs="+", default=list(WORLDS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--out", default="results/rq1_transfer")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(exist_ok=True)

    # test scenes: all objects present, fixed path subset
    tests = {}
    for domain in args.domains:
        tests[domain] = []
        for ts in TEST_SEEDS:
            env_t = make_env(domain, ts)
            all_t = all_trajectories(env_t)
            idx = np.random.default_rng(ts).choice(len(all_t), size=N_SUBSET, replace=False)
            paths = all_t[idx]
            tests[domain].append((env_t, paths, true_rewards(domain, env_t, paths)))

    rows, t0 = [], time.time()
    for domain in args.domains:
        world = WORLDS[domain]
        for scenario in args.scenarios:
            exclude = () if SCENARIOS[scenario]["sibling"] else (world["sibling"],)
            for seed in args.seeds:
                env = make_env(domain, seed, exclude=exclude)
                trajs = all_trajectories(env)
                true_r = true_rewards(domain, env, trajs)
                demos = sample_demos(true_r, seed=seed)
                verifier = make_mask_verifier(feature_column, env, trajs, demos)
                start = start_features(scenario, world)
                for method in args.methods:
                    try:
                        feats, _ = METHODS[method](start, env, trajs[0], world["instruction"], [],
                                                   world["correction"], verifier=verifier)
                    except Exception as e:
                        feats = start
                    Phi = np.stack([feature_column(f, env, trajs) for f in feats], axis=1)
                    theta, X = learn_theta(Phi, demos)
                    mu, sd = Phi.mean(0), Phi.std(0) + 1e-8
                    in_scene = normalized_reward(true_r, plan(X, theta))
                    transfer = []
                    for k, (env_t, paths, r_t) in enumerate(tests[domain]):
                        Phi_t = np.stack([column_subset(f, env_t, paths, f"test{k}") for f in feats], axis=1)
                        transfer.append(normalized_reward(r_t, int(np.argmax(((Phi_t - mu) / sd) @ theta))))
                    rows.append({"domain": domain, "scenario": scenario, "seed": seed, "method": method,
                                 "features": names(feats), "n_features": len(feats),
                                 "in_scene": in_scene, "transfer": float(np.mean(transfer))})
                    print(f"{domain:7s} | {scenario:9s} | seed {seed} | {method:6s} | in-scene {in_scene:.2f} | "
                          f"new scenes {np.mean(transfer):.2f} | {len(feats)} features")
                Path(args.out + ".json").write_text(json.dumps(rows, indent=1))

    lines = ["| Method | #features | in-scene | new scenes | " + " | ".join(args.scenarios) + " |",
             "|---|---|---|---|" + "---|" * len(args.scenarios)]
    for m in args.methods:
        r = [x for x in rows if x["method"] == m]
        cells = [f"{np.mean([x['transfer'] for x in r if x['scenario'] == sc]):.2f}" for sc in args.scenarios]
        lines.append(f"| {m} | {np.mean([x['n_features'] for x in r]):.1f} | "
                     f"{np.mean([x['in_scene'] for x in r]):.2f} | {np.mean([x['transfer'] for x in r]):.2f} | "
                     + " | ".join(cells) + " |")
    table = "\n".join(lines)
    Path(args.out + "_table.md").write_text(table + "\n")
    print("\n(scenario columns = mean normalized reward in the NEW scenes)\n" + table)
    print(f"\ntime: {(time.time() - t0) / 60:.1f} min")
    llm.print_usage()


if __name__ == "__main__":
    main()
