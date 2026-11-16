"""RQ2 in the gridworld (paper Sec. 5.2, Fig. 3/4, App. 7.4).

A 5-stage sequence of corrections where edits stack: each stage starts from the
feature set produced by the previous one. It mirrors the paper's CoffeeRobot
sequence (bowl -> fridge, laptop -> stove, phone -> kettle). The user's true
preference throughout: keep away from heat sources.

Evaluation (Fig. 4): the feature set learned after stage k is tested on EVERY
stage j (earlier = forgetting, later = generalisation).

Usage:
  python rq2_grid.py                       # all methods, 5 seeds
  python rq2_grid.py --methods FLEX --seeds 0
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import llm
from baselines import METHODS, PAPER_METHODS
from flex import hand_feature, names
from gridworld import DOMAINS, GRID, all_trajectories, to_flat, render
from irl import true_rewards, sample_demos, learn_theta, plan, normalized_reward, make_mask_verifier

INSTRUCTION = "Navigate to the goal."

# (objects in the scene, correction) per stage — Fig. 3 / App. 7.4, translated to KitchenWorld
STAGES = [
    {"objects": ["fridge"],           "correction": "Go near the fridge."},                   # ADD (wrong feature)
    {"objects": ["fridge", "stove"],  "correction": "Don't burn yourself on the stove."},     # ADD
    {"objects": ["fridge*", "stove"], "correction": "Wait, ignore the fridge now."},          # MASK  (* = next to stove)
    {"objects": ["stove", "kettle*"], "correction": "Good, don't burn yourself on the stove."},  # no change
    {"objects": ["kettle"],           "correction": "Don't burn yourself on the kettle too."},   # GENERALIZE
]


# ------------------------------------------------------------------ layouts
def _band(start):
    """Interior cells near the start-goal diagonal, so objects lie where paths go."""
    sx, sy = start
    on_main = sx == sy
    return [(x, y) for x in range(1, 8) for y in range(1, 8)
            if (abs(x - y) <= 1 if on_main else abs(x + y - 8) <= 1)]


def _neighbour(rng, cell, taken):
    x, y = cell
    options = [(x + dx, y + dy) for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
               if 1 <= x + dx <= 7 and 1 <= y + dy <= 7 and (x + dx, y + dy) not in taken]
    return options[rng.integers(len(options))]


def build_env(start, objects):
    """Environment dict in the paper's format from a list of (type, (x, y))."""
    spec = {o["type"]: o for o in DOMAINS["kitchen"]["objects"]}
    goal = (GRID - 1 - start[0], GRID - 1 - start[1])
    env = {"type": "gridrobot", "X": GRID, "Y": GRID, "starts": [list(start)], "goals pos": [list(goal)],
           "goals color": "green", "goals type": "cross"}
    for k, (t, pos) in enumerate(objects, start=1):
        env[f"objects{k} pos"] = [list(pos)]
        env[f"objects{k} color"] = spec[t]["color"]
        env[f"objects{k} type"] = t
        env[f"objects{k} property"] = list(spec[t]["property"])
    return env


def stage_envs(seed):
    """The 5 stage environments for one seed (random positions, fixed structure)."""
    rng = np.random.default_rng([seed, 2])
    start = [(0, 0), (0, 8), (8, 0), (8, 8)][rng.integers(4)]
    band = _band(start)
    pick = lambda exclude: band[rng.choice([i for i, c in enumerate(band) if c not in exclude])]

    stove = pick([])
    fridge_far = pick([stove])
    fridge_near = _neighbour(rng, stove, {stove})
    kettle_near = _neighbour(rng, stove, {stove, fridge_near})
    return [
        build_env(start, [("fridge", fridge_far)]),
        build_env(start, [("fridge", fridge_far), ("stove", stove)]),
        build_env(start, [("fridge", fridge_near), ("stove", stove)]),
        build_env(start, [("stove", stove), ("kettle", kettle_near)]),
        build_env(start, [("kettle", kettle_near)]),
    ]


# --------------------------------------------------------------- evaluation
_phi_cache = {}


def feature_column(feature, env, trajs):
    """Feature values for all paths. A feature whose objects are absent in this
    stage (e.g. 'stove' after the stove is removed) may crash or return NaN:
    we treat that as 0 (the feature carries no information here)."""
    key = hashlib.sha256((feature["code"] + json.dumps(env, sort_keys=True, default=str)).encode()).hexdigest()
    if key not in _phi_cache:
        vals = []
        with np.errstate(all="ignore"):
            for tr in trajs:
                try:
                    v = float(np.asarray(feature["fn"](env, to_flat(tr))).reshape(-1)[0])
                except Exception:
                    v = 0.0
                vals.append(v if np.isfinite(v) else 0.0)
        _phi_cache[key] = np.array(vals)
    return _phi_cache[key]


def score(features, stage, seed):
    env, trajs, true_r = stage
    Phi = np.stack([feature_column(f, env, trajs) for f in features], axis=1)
    theta, X = learn_theta(Phi, sample_demos(true_r, seed=seed))
    return normalized_reward(true_r, plan(X, theta))


# ---------------------------------------------------------------------- run
def run(methods, seeds, out):
    Path(out).parent.mkdir(exist_ok=True)
    K = len(STAGES)
    rows, t0 = [], time.time()
    for seed in seeds:
        stages = []
        for env in stage_envs(seed):
            trajs = all_trajectories(env)
            stages.append((env, trajs, true_rewards("kitchen", env, trajs)))
        for method in methods:
            features = [hand_feature("distance to goal", "goal", np.inf)]   # Phi^0
            history = []
            snapshots = [names(features)]
            matrix = [[score(features, stages[j], seed) for j in range(K)]]  # row 0 = Phi^0
            for k, st in enumerate(STAGES):
                env, trajs, true_r = stages[k]
                verifier = make_mask_verifier(feature_column, env, trajs, sample_demos(true_r, seed=seed))
                try:
                    features, log = METHODS[method](features, env, trajs[0], INSTRUCTION, list(history),
                                                    st["correction"], verifier=verifier)
                except Exception as e:
                    log = [("ERROR", f"{type(e).__name__}: {e}")]
                # The goal term belongs to the task ("Navigate to the goal"), not to the
                # user's preferences: no correction is about it, and Table 2 never edits it.
                # If a method drops it, we put it back (same rule for every method; DEVIATIONS.md).
                if not any("goal" in n.lower() for n in names(features)):
                    features = [hand_feature("distance to goal", "goal", np.inf)] + features
                    log.append(("RULE", "goal feature restored (task feature, not edited)"))
                history.append(st["correction"])
                snapshots.append(names(features))
                matrix.append([score(features, stages[j], seed) for j in range(K)])
                print(f"seed {seed} | {method:5s} | stage {k + 1} | \"{st['correction']}\" -> {names(features)}")
                for step, msg in log:
                    if step != "GENERALIZE" or not msg.startswith("intent"):
                        print(f"      [{step}] {msg}")
            rows.append({"seed": seed, "method": method, "matrix": matrix, "features": snapshots})
            Path(out + ".json").write_text(json.dumps(rows, indent=1))

    # ----- summary: mean R_hat on each evaluation stage j after training stage k
    print("\nMean normalized reward. Rows: feature set after stage k. Columns: evaluated on stage j.")
    for method in methods:
        M = np.array([r["matrix"] for r in rows if r["method"] == method])  # (seeds, K+1, K)
        print(f"\n{method}")
        print("            " + "  ".join(f"eval S{j + 1}" for j in range(K)))
        for k in range(K + 1):
            print(f"  Phi^{k}    " + "  ".join(f"{M[:, k, j].mean():7.2f}" for j in range(K)))
    plot(rows, methods, out)
    print(f"\ntime: {(time.time() - t0) / 60:.1f} min")
    llm.print_usage()


def plot(rows, methods, out):
    """Figure-4 style: one panel per evaluation stage, x = training stage k."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    K = len(STAGES)
    fig, axes = plt.subplots(1, K, figsize=(3.2 * K, 3), sharey=True)
    for j, ax in enumerate(axes):
        for method in methods:
            M = np.array([r["matrix"] for r in rows if r["method"] == method])[:, :, j]
            mean = M.mean(0)
            se = M.std(0, ddof=1) / np.sqrt(len(M)) if len(M) > 1 else np.zeros_like(mean)
            x = np.arange(K + 1)
            ax.plot(x, mean, marker="o", label=method)
            ax.fill_between(x, mean - se, mean + se, alpha=0.2)
        ax.set_title(f"Evaluation on stage {j + 1}")
        ax.set_xlabel("feature set after stage k")
        ax.set_xticks(range(K + 1))
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("normalized reward")
    axes[-1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=150)
    print(f"plot saved to {out}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=PAPER_METHODS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--out", default="results/rq2_grid")
    ap.add_argument("--show-layouts", action="store_true", help="print the 5 stage grids and exit")
    a = ap.parse_args()
    if a.show_layouts:
        for s in a.seeds:
            for k, env in enumerate(stage_envs(s)):
                print(f"seed {s} stage {k + 1}: \"{STAGES[k]['correction']}\"\n{render(env)}\n")
    else:
        run(a.methods, a.seeds, a.out)
