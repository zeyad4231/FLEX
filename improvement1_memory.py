"""IMPROVEMENT 1: feature memory (continual learning) — a preference that goes away and comes back.

Stage 1  "Stay close to the fridge."                       truth: avoid heat + near fridge
Stage 2  "You don't need to go near the fridge anymore."    truth: avoid heat
Stage 3  "Stay close to the fridge again, like before."     truth: avoid heat + near fridge

FLEX+V    : MASK deletes the fridge feature; stage 3 rebuilds it from scratch; weights are
            relearned from zero at every stage (as in the paper).
FLEX+V+M  : MASK archives the feature (code + learned weight + which objects it measures); ADD
            reactivates it; the reactivated feature's remembered weight is the prior when relearning.

Measured at stage 3: normalized reward with 1, 2, 5 noisy demos (beta = 1, 10 draws each), LLM calls.

Usage: python improvement1_memory.py [--seeds 0 1 2 3 4]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import llm
from flex import hand_feature, names, flex_update
from irl import make_distance_feature, feature_matrix, sample_demos, learn_theta, plan, \
    normalized_reward, make_mask_verifier
from gridworld import all_trajectories
from rq2_grid import build_env, _band, feature_column

INSTRUCTION = "Navigate to the goal."
TRUE_FEATURES = [("goal", np.inf), ("heat source", 3.0), ("fridge", 3.0)]
STAGES = [
    {"correction": "Stay close to the fridge.",                     "theta": [-0.1, 1.0, -1.0]},
    {"correction": "You don't need to go near the fridge anymore.", "theta": [-0.1, 1.0, 0.0]},
    {"correction": "Stay close to the fridge again, like before.",  "theta": [-0.1, 1.0, -1.0]},
]
N_DEMOS = [1, 2, 5]
DRAWS = 10
PRIOR_STRENGTH = 1.0
STAGE3_BETA = 1.0   # the returning user gives few, sloppy demos


def stage_envs(seed):
    """A new random layout per stage (fridge, stove, kettle near the diagonal)."""
    rng = np.random.default_rng([seed, 7])
    start = [(0, 0), (0, 8), (8, 0), (8, 8)][rng.integers(4)]
    band = _band(start)
    envs = []
    for _ in STAGES:
        cells = [band[i] for i in rng.choice(len(band), size=3, replace=False)]
        envs.append(build_env(start, list(zip(["fridge", "stove", "kettle"], cells))))
    return envs


def true_r(env, trajs, theta):
    feats = [make_distance_feature(p, r) for p, r in TRUE_FEATURES]
    return feature_matrix(feats, env, trajs) @ np.array(theta)


def learn(features, env, trajs, demos, use_memory, store=True):
    """Learn theta. Arm M: a feature reactivated from the archive uses its remembered
    weight as a Gaussian prior (other features are learned from scratch, as in arm V)."""
    Phi = np.stack([feature_column(f, env, trajs) for f in features], axis=1)
    sd = Phi.std(0) + 1e-8
    mean = np.zeros(len(features))
    lam = np.full(len(features), 1e-3)
    if use_memory:
        for i, f in enumerate(features):
            if f.get("from_memory") and "theta_raw" in f:
                mean[i] = f["theta_raw"] * sd[i]  # raw weight -> standardised units of this scene
                lam[i] = PRIOR_STRENGTH
    theta, X = learn_theta(Phi, demos, prior_mean=mean, prior_strength=lam)
    if store:
        for f, t, s in zip(features, theta, sd):
            f["theta_raw"] = t / s  # remember the weight in raw units
    return plan(X, theta)


def calls():
    return llm.usage["api_calls"] + llm.usage["cache_hits"]


def run(seeds, out):
    Path(out).parent.mkdir(exist_ok=True)
    rows, t0 = [], time.time()
    for seed in seeds:
        envs = stage_envs(seed)
        data = []
        for env, st in zip(envs, STAGES):
            trajs = all_trajectories(env)
            data.append((env, trajs, true_r(env, trajs, st["theta"])))
        for arm in ["FLEX+V", "FLEX+V+M"]:
            use_mem = arm.endswith("+M")
            features = [hand_feature("distance to goal", "goal", np.inf),
                        hand_feature("distance to heat source", "heat source")]
            memory = [] if use_mem else None
            history, result = [], {"seed": seed, "arm": arm, "stages": []}
            for k, (st, (env, trajs, r)) in enumerate(zip(STAGES, data)):
                demos5 = sample_demos(r, n_demos=5, seed=seed)
                verifier = make_mask_verifier(feature_column, env, trajs, demos5)
                c0 = calls()
                features, log = flex_update(features, env, trajs[0], INSTRUCTION, list(history),
                                            st["correction"], verifier=verifier, memory=memory)
                n_calls = calls() - c0
                if not any("goal" in n.lower() for n in names(features)):  # task feature rule (RQ2)
                    features = [hand_feature("distance to goal", "goal", np.inf)] + features
                history.append(st["correction"])
                if k < len(STAGES) - 1:   # stages 1-2: learn with 5 demos (stores weights)
                    scores = {5: normalized_reward(r, learn(features, env, trajs, demos5, use_mem))}
                else:                      # stage 3: few-demo evaluation, many draws
                    scores = {}
                    for n in N_DEMOS:
                        vals = []
                        for d in range(DRAWS):
                            demos = sample_demos(r, n_demos=n, beta=STAGE3_BETA, seed=10_000 * seed + 100 * n + d)
                            vals.append(normalized_reward(r, learn(features, env, trajs, demos, use_mem,
                                                                   store=False)))
                        scores[n] = float(np.mean(vals))
                result["stages"].append({"stage": k + 1, "features": names(features), "llm_calls": n_calls,
                                         "scores": scores, "log": log})
                print(f"seed {seed} | {arm:8s} | stage {k + 1} | calls {n_calls:2d} | "
                      f"{ {n: round(v, 2) for n, v in scores.items()} } | {names(features)}")
                for step, msg in log:
                    if step in ("MEMORY", "MASK", "ADD", "MASK-CHECK", "GEN-CHECK") or \
                            (step == "GENERALIZE" and "->" in msg):
                        print(f"      [{step}] {msg}")
            rows.append(result)
            Path(out + ".json").write_text(json.dumps(rows, indent=1, default=str))

    print("\nStage 3 (preference returns): mean normalized reward by number of demos, and LLM calls")
    print(f"{'arm':10s} " + " ".join(f"{n} demo{'s' if n > 1 else ''}".rjust(9) for n in N_DEMOS) + "   LLM calls")
    for arm in ["FLEX+V", "FLEX+V+M"]:
        st3 = [r["stages"][2] for r in rows if r["arm"] == arm]
        means = [np.mean([s["scores"][n] for s in st3]) for n in N_DEMOS]
        print(f"{arm:10s} " + " ".join(f"{m:9.2f}" for m in means) +
              f"   {np.mean([s['llm_calls'] for s in st3]):9.1f}")
    print(f"\ntime: {(time.time() - t0) / 60:.1f} min")
    llm.print_usage()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--out", default="results/improvement1_memory")
    a = ap.parse_args()
    run(a.seeds, a.out)
