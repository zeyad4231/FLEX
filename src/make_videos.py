"""Demo videos of the RQ2 correction sequence, replayed from the LLM cache.

Replays the 5-stage KitchenWorld sequence (rq2_grid.py) for FLEX and FLEX+V
on one seed and animates what the robot does after each correction.
Every GPT-4o answer comes from cache/, so this makes no API calls and costs nothing.

  video 1  rq2_sequence.mp4   all 5 corrections, FLEX vs FLEX+V side by side
  video 2  mask_failure.mp4   stage 2 step by step: ADD -> GENERALIZE -> MASK

Usage:
  python src/make_videos.py                 # both videos -> docs/videos/
  python src/make_videos.py --preview       # one PNG frame per video, to check the layout
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("FLEX_MODEL", "gpt-4o")  # the cache was built with gpt-4o

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyBboxPatch, Rectangle

import llm
import rq2_grid as R
from baselines import METHODS
from flex import hand_feature, names
from gridworld import GRID
from irl import true_rewards, sample_demos, learn_theta, plan, normalized_reward, make_mask_verifier

OUT = Path(__file__).resolve().parent.parent / "docs" / "videos"
FPS = 20

# ------------------------------------------------------------------ colours
INK, MUTED, FAINT = "#1f2328", "#57606a", "#d0d7de"
BG, CELL = "#ffffff", "#f6f8fa"
OBJ = {"fridge": "#3b82c4", "stove": "#d1495b", "kettle": "#e8871e"}
GOAL, START = "#2e9e5b", "#6e7781"
PATH = {"FLEX": "#8250df", "FLEX+V": "#0b7a75"}
GOOD, BAD = "#1a7f37", "#cf222e"


# ------------------------------------------------------------------ capture
def capture(seed, method):
    """Re-run the RQ2 sequence for one method; keep everything the video needs."""
    stages = []
    for env in R.stage_envs(seed):
        trajs = R.all_trajectories(env)
        stages.append((env, trajs, true_rewards("kitchen", env, trajs)))

    features = [hand_feature("distance to goal", "goal", np.inf)]
    history, out = [], []
    for k, st in enumerate(R.STAGES):
        env, trajs, true_r = stages[k]
        demos = sample_demos(true_r, seed=seed)
        before = names(features)
        verifier = make_mask_verifier(R.feature_column, env, trajs, demos)
        features, log = METHODS[method](features, env, trajs[0], R.INSTRUCTION, list(history),
                                        st["correction"], verifier=verifier)
        if not any("goal" in n.lower() for n in names(features)):   # same rule as rq2_grid.py
            features = [hand_feature("distance to goal", "goal", np.inf)] + features
            log = list(log) + [("RULE", "goal restored")]
        history.append(st["correction"])

        Phi = np.stack([R.feature_column(f, env, trajs) for f in features], axis=1)
        theta, X = learn_theta(Phi, demos)
        idx = plan(X, theta)
        out.append({
            "env": env, "correction": st["correction"], "before": before, "after": names(features),
            "log": log, "path": trajs[idx], "demos": [trajs[i] for i in demos],
            "r_hat": normalized_reward(true_r, idx),
        })
    return out


def summarize(log):
    """Short, readable lines for the ops that ran in one stage."""
    lines = []
    for step, msg in log:
        if step == "GENERALIZE" and msg.startswith("intent"):
            continue
        if step == "MASK-CHECK" and "allowed" in msg:
            continue
        if msg in ("no change", "no new feature"):
            continue
        if step == "MASK-CHECK":
            lr = msg.split("LR=")[-1].rstrip(")")
            feat = msg.split("'")[1]
            lines.append(("MASK-CHECK", f"refused to remove {short(feat)}", GOOD))
            lines.append(("", f"demos depend on it (LR = {lr})", GOOD))
        elif step == "MASK":
            lines.append(("MASK", "removed " + short(msg.lstrip("- ").strip()), BAD))
        elif step == "ADD":
            lines.append(("ADD", "+ " + short(msg.lstrip("+ ").strip()), INK))
        elif step == "RULE":
            lines.append(("RULE", "'goal' restored (task feature)", MUTED))
        elif step == "GENERALIZE":
            lines.append(("GENERALIZE", " → ".join(short(x.strip()) for x in msg.split("->")), INK))
    return lines or [("", "no change", MUTED)]


def short(name):
    """'distance to heat source' -> 'heat source' (all features here are distances)."""
    return "'" + name.lower().replace("distance to ", "") + "'"


# ------------------------------------------------------------------ drawing
def to_xy(p):
    """Grid (x, y) with y=0 at the top -> plot coordinates."""
    return p[0], GRID - 1 - p[1]


def draw_grid(ax, env):
    ax.set_xlim(-0.6, GRID - 0.4)
    ax.set_ylim(-0.6, GRID - 0.4)
    ax.set_aspect("equal")
    ax.axis("off")
    for x in range(GRID):
        for y in range(GRID):
            ax.add_patch(Rectangle((x - 0.46, y - 0.46), 0.92, 0.92, fc=CELL, ec="none", zorder=0))
    for k in range(1, 4):
        for pos in env.get(f"objects{k} pos", []):
            t = env[f"objects{k} type"]
            x, y = to_xy(pos)
            ax.add_patch(FancyBboxPatch((x - 0.4, y - 0.4), 0.8, 0.8, boxstyle="round,pad=0,rounding_size=0.12",
                                        fc=OBJ[t], ec="white", lw=2, zorder=3))
            ax.text(x, y - 0.68, t, ha="center", va="top", fontsize=9, color=INK, zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    sx, sy = to_xy(env["starts"][0])
    gx, gy = to_xy(env["goals pos"][0])
    ax.plot(sx, sy, "o", ms=13, color=START, zorder=4)
    ax.text(sx, sy, "S", ha="center", va="center", color="white", fontsize=8, weight="bold", zorder=5)
    ax.plot(gx, gy, "X", ms=15, color=GOAL, zorder=4)


def draw_demos(ax, demos):
    for d in demos:
        xy = np.array([to_xy(p) for p in d])
        ax.plot(xy[:, 0], xy[:, 1], color="#9aa4ae", lw=1.2, alpha=0.5, zorder=1, solid_capstyle="round")


def draw_path(ax, path, color, upto):
    """Draw the robot's path up to step `upto` (float, for smooth motion)."""
    xy = np.array([to_xy(p) for p in path], dtype=float)
    n = int(np.floor(upto))
    frac = upto - n
    pts = xy[: n + 1]
    if n + 1 < len(xy) and frac > 0:
        pts = np.vstack([pts, xy[n] + frac * (xy[n + 1] - xy[n])])
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=3, zorder=2, solid_capstyle="round", solid_joinstyle="round")
    ax.plot(pts[-1, 0], pts[-1, 1], "o", ms=12, color=color, mec="white", mew=2, zorder=7)


def feature_box(fig, x, y, w, title, feats, color):
    fig.text(x, y, title, fontsize=10, color=MUTED, ha="left", va="top")
    fig.text(x, y - 0.032, "Φ = [ " + ",  ".join(feats) + " ]", fontsize=11, color=color, ha="left", va="top",
             wrap=True)


def score_text(fig, x, y, r):
    col = GOOD if r >= 0.95 else (BAD if r < 0.8 else "#9a6700")
    fig.text(x, y, f"R̂ = {r:.2f}", fontsize=15, weight="bold", color=col, ha="right", va="top")


# ------------------------------------------------------------------ video 1
def video_sequence(runs, path_out, preview=False):
    """All 5 stages, FLEX (left) vs FLEX+V (right)."""
    K = len(R.STAGES)
    T = len(runs["FLEX"][0]["path"])
    hold_in, move, hold_out = int(1.5 * FPS), int(2.5 * FPS), int(3.0 * FPS)
    per = hold_in + move + hold_out
    intro, outro = int(4 * FPS), int(6 * FPS)
    total = intro + K * per + outro

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)

    def frame(i):
        fig.clf()
        if i < intro:
            return title_card(fig,
                              "FLEX reproduction: 5 language corrections in a row",
                              ["KitchenWorld, 9×9 grid. The user wants the robot to keep away from heat sources.",
                               "After each correction the robot updates its features Φ from language, learns θ from 5 demos",
                               "(grey lines) and plans a path to the goal.",
                               "",
                               "Left: FLEX as in the paper.   Right: FLEX+V, MASK is checked against the demos.",
                               "All GPT-4o answers are replayed from the cache: same run as the report (seed 0)."])
        if i >= intro + K * per:
            return title_card(fig, "Result after 5 corrections",
                              [f"FLEX: R̂ per stage  " + "  ".join(f"{s['r_hat']:.2f}" for s in runs['FLEX']),
                               f"FLEX+V: R̂ per stage  " + "  ".join(f"{s['r_hat']:.2f}" for s in runs['FLEX+V']),
                               "",
                               "FLEX deletes the heat-source feature at stages 2, 3 and 4: GENERALIZE creates it,",
                               "then MASK removes it, because the MASK prompt never sees the generalization intent.",
                               "FLEX+V refuses those removals (likelihood-ratio test, LR ≥ 3.84) and is right from stage 2."])
        k, t = divmod(i - intro, per)
        upto = 0 if t < hold_in else min(T - 1, (t - hold_in) / move * (T - 1))
        stage_frame(fig, k, runs, upto, show_path=t >= hold_in)

    def stage_frame(fig, k, runs, upto, show_path):
        fig.text(0.5, 0.955, f"Stage {k + 1} of {K}", ha="center", fontsize=12, color=MUTED)
        fig.text(0.5, 0.91, f"User: “{runs['FLEX'][k]['correction']}”", ha="center", fontsize=19,
                 color=INK, weight="bold")
        for col, m in enumerate(["FLEX", "FLEX+V"]):
            s = runs[m][k]
            x0 = 0.04 + col * 0.5
            ax = fig.add_axes([x0, 0.25, 0.25, 0.6])
            draw_grid(ax, s["env"])
            draw_demos(ax, s["demos"])
            if show_path:
                draw_path(ax, s["path"], PATH[m], upto)
            fig.text(x0 + 0.27, 0.82, m, fontsize=18, weight="bold", color=PATH[m])
            fig.text(x0 + 0.27, 0.775, "paper's method" if m == "FLEX" else "+ demo-verified MASK",
                     fontsize=10, color=MUTED)
            yy = 0.71
            for op, text, c in summarize(s["log"]):
                if not op:
                    yy += 0.045
                fig.text(x0 + 0.27, yy, op, fontsize=9, color=MUTED, weight="bold")
                fig.text(x0 + 0.27, yy - 0.03, text, fontsize=10.5, color=c)
                yy -= 0.075
            if show_path:
                score_text(fig, x0 + 0.44, 0.33, s["r_hat"])
            feature_box(fig, x0, 0.2, 0.45, "features after this correction", s["after"], INK)
        fig.add_artist(plt.Line2D([0.5, 0.5], [0.1, 0.86], color=FAINT, lw=1))
        fig.text(0.5, 0.03, "grey lines = the user's demonstrations     coloured line = robot's planned path"
                 "     R̂ = normalized reward (1 = best path)", ha="center", fontsize=9.5, color=MUTED)

    if preview:
        frame(intro + 1 * per + hold_in + move + 5)   # stage 2, path drawn
        fig.savefig(path_out.with_suffix(".png"))
        return
    anim = animation.FuncAnimation(fig, frame, frames=total)
    anim.save(path_out, writer=animation.FFMpegWriter(fps=FPS, bitrate=2500, codec="libx264",
                                                       extra_args=["-pix_fmt", "yuv420p", "-preset", "veryfast"]))


# ------------------------------------------------------------------ video 2
def video_mask(runs, path_out, preview=False):
    """Stage 2 only, one operation at a time, to show where FLEX goes wrong."""
    k = 1
    f, v = runs["FLEX"][k], runs["FLEX+V"][k]
    lr = next(m.split("LR=")[-1].rstrip(")") for s, m in v["log"] if s == "MASK-CHECK")
    steps = [
        ("Before the correction", "Φ = [ " + ",  ".join(f["before"]) + " ]",
         "At stage 1, 'Go near the fridge' was generalized to 'distance to appliance'.", INK, None),
        ("1. ADD", "no new feature", "ADD finds no missing feature for this correction.", INK, None),
        ("2. GENERALIZE", "distance to appliance → distance to heat source",
         "GENERALIZE rewrites the feature around the property behind the correction: heat.", INK, None),
        ("3. MASK (FLEX)", "− distance to heat source",
         "The MASK prompt does not see the generalization intent, so it judges\n'heat source' irrelevant and deletes the feature GENERALIZE just created.", BAD, "FLEX"),
        ("3. MASK (FLEX+V)", f"refused: removing it makes the demos much less likely (LR = {lr} ≥ 3.84)",
         "Before deleting, FLEX+V refits θ with and without the feature.\nThe user's demos clearly avoid the stove, so the feature stays.", GOOD, "FLEX+V"),
    ]
    ops = [(st, m) for st, m in f["log"] if not m.startswith("intent")]
    assert ops == [("ADD", "no new feature"), ("GENERALIZE", "distance to appliance -> distance to heat source"),
                   ("MASK", "- distance to heat source")], f"stage-2 log changed: {ops}"
    T = len(f["path"])
    per, move = int(4.5 * FPS), int(2.5 * FPS)
    intro, outro = int(4 * FPS), int(5 * FPS)
    total = intro + len(steps) * per + outro
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)

    def frame(i):
        fig.clf()
        if i < intro:
            return title_card(fig, "Why faithful FLEX scores 0.92 instead of 0.99",
                              ["One correction, one operation at a time (stage 2 of the sequence, seed 0).",
                               "",
                               "FLEX edits the feature set with three LLM calls: ADD, then GENERALIZE, then MASK.",
                               "In 21 of 70 RQ1 runs, MASK deleted the feature GENERALIZE had just created."])
        if i >= intro + len(steps) * per:
            return title_card(fig, "The fix: check MASK against the demonstrations",
                              ["Keep a feature if removing it lowers the demos' log-likelihood significantly",
                               "(likelihood-ratio test, χ²(1), 95%: LR ≥ 3.84).",
                               "",
                               "RQ1 average normalized reward:  FLEX 0.92  →  FLEX+V 0.98   (paper: 0.99)",
                               "RQ2: FLEX+V is correct after 2 corrections instead of 5."])
        j, t = divmod(i - intro, per)
        title, change, why, col, which = steps[j]
        fig.text(0.5, 0.93, "User: “Don't burn yourself on the stove.”", ha="center", fontsize=18,
                 weight="bold", color=INK)
        ax = fig.add_axes([0.05, 0.12, 0.38, 0.7])
        draw_grid(ax, f["env"])
        draw_demos(ax, f["demos"])
        if which:
            s = runs[which][k]
            upto = min(T - 1, t / move * (T - 1))
            draw_path(ax, s["path"], PATH[which], upto)
            if t >= move:
                fig.text(0.43, 0.85, f"R̂ = {s['r_hat']:.2f}", ha="right", fontsize=15, weight="bold",
                         color=GOOD if s["r_hat"] >= 0.95 else BAD)
        y = 0.8
        for jj, (tt, cc, _, c2, _) in enumerate(steps[: j + 1]):
            if j == 4 and jj == 3:
                continue   # show the FLEX+V branch in place of the FLEX one
            active = jj == j
            fig.text(0.5, y, tt, fontsize=13 if active else 11, weight="bold", color=INK if active else MUTED)
            fig.text(0.5, y - 0.045, cc, fontsize=12 if active else 10.5, color=c2 if active else MUTED)
            y -= 0.12
        fig.text(0.5, 0.2, why, fontsize=11.5, color=INK, va="top", linespacing=1.5)

    if preview:
        frame(intro + 3 * per + move + 5)
        fig.savefig(path_out.with_suffix(".png"))
        return
    anim = animation.FuncAnimation(fig, frame, frames=total)
    anim.save(path_out, writer=animation.FFMpegWriter(fps=FPS, bitrate=2500, codec="libx264",
                                                       extra_args=["-pix_fmt", "yuv420p", "-preset", "veryfast"]))


def title_card(fig, title, lines):
    fig.text(0.5, 0.68, title, ha="center", fontsize=24, weight="bold", color=INK)
    for n, line in enumerate(lines):
        fig.text(0.5, 0.56 - n * 0.055, line, ha="center", fontsize=13, color=MUTED if line else INK)


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {m: capture(a.seed, m) for m in ["FLEX", "FLEX+V"]}
    print("API calls:", llm.usage["api_calls"], "| cache hits:", llm.usage["cache_hits"])
    video_sequence(runs, OUT / "rq2_sequence.mp4", a.preview)
    video_mask(runs, OUT / "mask_failure.mp4", a.preview)
    print("saved to", OUT)
