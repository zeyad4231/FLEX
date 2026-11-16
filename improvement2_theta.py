"""IMPROVEMENT 2 experiment: corrections that say HOW MUCH, not WHICH.

Scene: fridge right next to the stove. The robot already has the right features
[goal, heat source, fridge] and learned theta from 5 demos (avoid stove, stay near fridge).
The user then gives one 'how much' correction and the true preference changes. No new demos.

FLEX+V    : the paper's assumption -- language only edits features; theta relearned from the
            same (old) demos, so behaviour stays as before.
FLEX+V+th : the correction is classified; 'how much' scales the named feature's weight.

Also: the classifier is run on every earlier correction (RQ1, RQ2, improvement 1), which
should all be classified 'which' (false-positive check).

Usage: python improvement2_theta.py [--seeds 0 1 2 3 4]
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
from rq2_grid import build_env, _band, _neighbour, feature_column
from theta_language import classify, scale_theta

INSTRUCTION = "Navigate to the goal."
OLD_THETA = [-0.1, 1.0, -1.0]   # goal, heat source (avoid), fridge (stay near)
CORRECTIONS = [
    ("Stay much further away from the stove, even if you are farther from the fridge.", 3.0),
    ("Keep a bit more distance from the stove.", 2.0),
    ("You can get closer to the stove, it's not that hot.", 0.3),
    ("Getting a little closer to the stove is okay.", 0.5),
]
# False-positive check: every earlier 'which' correction, with the feature list the robot
# actually had just before it (RQ1 start sets, RQ2 FLEX+V seed-0 stages, improvement 1).
G, H, F = "distance to goal", "distance to heat source", "distance to fridge"
REALISTIC = [
    ([G], "Don't burn yourself on the stove."),
    ([G, "distance to stove"], "Don't burn yourself on the stove."),
    ([G, "distance to stove", F], "Don't burn yourself on the stove."),
    ([G, F], "Don't burn yourself on the stove."),
    ([G], "Don't walk into the lava."),
    ([G, "distance to lava"], "Don't walk into the lava."),
    ([G, "distance to lava", "distance to meadow"], "Don't walk into the lava."),
    ([G, "distance to meadow"], "Don't walk into the lava."),
    ([G], "Go near the fridge."),
    ([G, "distance to appliance"], "Don't burn yourself on the stove."),
    ([G, H], "Wait, ignore the fridge now."),
    ([G, H], "Good, don't burn yourself on the stove."),
    ([G, H], "Don't burn yourself on the kettle too."),
    ([G, H], "Stay close to the fridge."),
    ([G, H, "distance to light source"], "You don't need to go near the fridge anymore."),
    ([G, H], "Stay close to the fridge again, like before."),
]
# Hard case (reported separately): the related feature is ALREADY in the list.
HARD = ["Don't burn yourself on the stove.", "Go near the fridge.", "Good, don't burn yourself on the stove.",
        "Stay close to the fridge.", "Stay close to the fridge again, like before."]


def env_for(seed):
    rng = np.random.default_rng([seed, 11])
    start = [(0, 0), (0, 8), (8, 0), (8, 8)][rng.integers(4)]
    band = _band(start)
    stove = band[rng.integers(len(band))]
    fridge = _neighbour(rng, stove, {stove})
    return build_env(start, [("fridge", fridge), ("stove", stove)])


def run(seeds, out):
    Path(out).parent.mkdir(exist_ok=True)
    t0, rows = time.time(), []
    true_feats = [make_distance_feature("goal", np.inf), make_distance_feature("heat source", 3.0),
                  make_distance_feature("fridge", 3.0)]
    for seed in seeds:
        env = env_for(seed)
        trajs = all_trajectories(env)
        Phi_true = feature_matrix(true_feats, env, trajs)
        old_demos = sample_demos(Phi_true @ np.array(OLD_THETA), n_demos=5, seed=seed)
        start = [hand_feature("distance to goal", "goal", np.inf),
                 hand_feature("distance to heat source", "heat source"),
                 hand_feature("distance to fridge", "fridge")]
        verifier = make_mask_verifier(feature_column, env, trajs, old_demos)
        for correction, new_heat in CORRECTIONS:
            true_r = Phi_true @ np.array([OLD_THETA[0], new_heat, OLD_THETA[2]])
            # both arms: the usual FLEX+V feature update (should change nothing here)
            feats, log = flex_update(start, env, trajs[0], INSTRUCTION, [], correction, verifier=verifier)
            Phi = np.stack([feature_column(f, env, trajs) for f in feats], axis=1)
            theta, X = learn_theta(Phi, old_demos)
            r_plain = normalized_reward(true_r, plan(X, theta))
            # ours: classify the correction and scale the weight
            decision = classify(names(feats), [], correction)
            theta2, idx = scale_theta(theta, names(feats), decision)
            r_ours = normalized_reward(true_r, plan(X, theta2))
            rows.append({"seed": seed, "correction": correction, "true_heat": new_heat,
                         "features": names(feats), "decision": decision, "scaled": idx is not None,
                         "FLEX+V": r_plain, "FLEX+V+th": r_ours, "log": log})
            print(f"seed {seed} | {correction[:45]:45s} | {decision.get('type'):8s} "
                  f"{decision.get('direction')}/{decision.get('strength')} x{decision.get('factor') or 1:.2f} | "
                  f"FLEX+V {r_plain:.2f} -> FLEX+V+th {r_ours:.2f} | {names(feats)}")
        Path(out + ".json").write_text(json.dumps(rows, indent=1, default=str))

    print("\nMean normalized reward after a 'how much' correction (no new demos):")
    print(f"{'correction':55s} {'FLEX+V':>7s} {'FLEX+V+th':>10s}")
    for correction, _ in CORRECTIONS:
        r = [x for x in rows if x["correction"] == correction]
        print(f"{correction[:55]:55s} {np.mean([x['FLEX+V'] for x in r]):7.2f} "
              f"{np.mean([x['FLEX+V+th'] for x in r]):10.2f}")
    print(f"{'ALL':55s} {np.mean([x['FLEX+V'] for x in rows]):7.2f} {np.mean([x['FLEX+V+th'] for x in rows]):10.2f}")

    print("\nFalse-positive check (realistic): earlier 'which' corrections with the features the robot had")
    wrong = 0
    for feats, c in REALISTIC:
        d = classify(feats, [], c)
        wrong += d["type"] == "how_much"
        print(f"  {d['type']:8s} <- \"{c}\"  with {feats}")
    print(f"misclassified as how_much: {wrong} / {len(REALISTIC)}")
    print("\nHard case: related feature already present ([goal, heat source, fridge])")
    hard = 0
    for c in HARD:
        d = classify([G, H, F], [], c)
        hard += d["type"] == "how_much"
        print(f"  {d['type']:8s} <- \"{c}\"")
    print(f"classified as how_much: {hard} / {len(HARD)}")
    print(f"\ntime: {(time.time() - t0) / 60:.1f} min")
    llm.print_usage()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--out", default="results/improvement2_theta")
    a = ap.parse_args()
    run(a.seeds, a.out)
