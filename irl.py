"""Reward learning (paper Sec. 3, Eq. 1-3) + ground truth + evaluation metric.

Pipeline for one environment:
  features Phi  ->  feature matrix for all 12,870 paths
  true reward   ->  simulated expert demos (Eq. 2, noisily rational)
  demos         ->  learned weights theta (Eq. 3, MaxEnt IRL)
  theta         ->  robot's path (argmax) -> normalized reward R_hat
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from gridworld import get_objects_with_property, to_flat


# ---------------------------------------------------------------- features
def make_distance_feature(prop, r=3.0):
    """Feature = sum over the path of (distance to objects with `prop`, capped at r),
    averaged over matching objects. Same form as the paper's example (App. p. 20).
    Higher value = the path stayed farther away."""
    def feature(environment, trajectory):
        positions = get_objects_with_property(prop, environment)
        if not positions:
            return np.zeros(1)
        value = 0.0
        for i in range(0, len(trajectory) - 1, 2):
            for p in positions:
                d = np.hypot(trajectory[i] - p[0], trajectory[i + 1] - p[1])
                value += min(d, r)
        return np.array([value / len(positions)])
    feature.__name__ = f"distance_to_{prop.replace(' ', '_')}"
    return feature


def feature_matrix(features, env, trajs):
    """Phi for every path: array (N_paths, N_features)."""
    flat = [to_flat(t) for t in trajs]
    cols = []
    for f in features:
        cols.append([float(np.asarray(f(env, tr)).reshape(-1)[0]) for tr in flat])
    return np.array(cols).T


# ------------------------------------------------------------ ground truth
# The user's true preference per domain: reach the goal (Phi_goal, uncapped) and
# keep away from the dangerous category. Weights are our choice (see DEVIATIONS.md).
GROUND_TRUTH = {
    "kitchen": {"features": [("goal", np.inf), ("heat source", 3.0)], "theta": np.array([-0.1, 1.0])},
    "lava":    {"features": [("goal", np.inf), ("dangerous", 3.0)],   "theta": np.array([-0.1, 1.0])},
}


def true_rewards(domain, env, trajs):
    gt = GROUND_TRUTH[domain]
    feats = [make_distance_feature(p, r) for p, r in gt["features"]]
    return feature_matrix(feats, env, trajs) @ gt["theta"]


# ------------------------------------------------------------------ demos
def sample_demos(true_r, n_demos=5, beta=5.0, seed=0):
    """Eq. 2: the expert picks path tau with probability ∝ exp(beta * R*(tau))."""
    rng = np.random.default_rng(seed)
    logits = beta * true_r
    p = np.exp(logits - logsumexp(logits))
    return rng.choice(len(true_r), size=n_demos, p=p)


# -------------------------------------------------------------------- IRL
def learn_theta(Phi, demo_idx, l2=1e-3):
    """Eq. 3 with a uniform prior = maximum likelihood (MaxEnt IRL).

    The normaliser sums over ALL 12,870 paths, so it is exact.
    Features are standardised for stable optimisation; a tiny L2 term keeps
    theta finite when demos are (near-)optimal.
    """
    mu, sd = Phi.mean(0), Phi.std(0) + 1e-8
    X = (Phi - mu) / sd
    D = X[demo_idx]
    n = len(D)

    def neg_log_lik(theta):
        s = X @ theta
        lse = logsumexp(s)
        p = np.exp(s - lse)
        value = -(D @ theta).sum() + n * lse + l2 * theta @ theta
        grad = -D.sum(0) + n * (p @ X) + 2 * l2 * theta
        return value, grad

    res = minimize(neg_log_lik, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B")
    return res.x, X


def plan(X, theta):
    """Robot picks the highest-reward path (argmax)."""
    return int(np.argmax(X @ theta))


def normalized_reward(true_r, idx):
    """R_hat = (R - R_worst) / (R_best - R_worst); 1 = optimal, 0 = worst."""
    lo, hi = true_r.min(), true_r.max()
    return float((true_r[idx] - lo) / (hi - lo)) if hi > lo else 1.0


def evaluate(features, domain, env, trajs, true_r, n_demos=5, seed=0):
    """Full loop for one feature set: demos -> theta -> path -> R_hat."""
    demos = sample_demos(true_r, n_demos=n_demos, seed=seed)
    Phi = feature_matrix(features, env, trajs)
    theta, X = learn_theta(Phi, demos)
    idx = plan(X, theta)
    return normalized_reward(true_r, idx), idx, theta


# ------------------------------------------------------------- checkpoint
if __name__ == "__main__":
    import time
    from gridworld import make_env, all_trajectories, render

    tests = {
        "kitchen": {
            "CORRECT  goal + heat source": [("goal", np.inf), ("heat source", 3.0)],
            "wrong    goal + stove       ": [("goal", np.inf), ("stove", 3.0)],
            "wrong    goal + fridge      ": [("goal", np.inf), ("fridge", 3.0)],
        },
        "lava": {
            "CORRECT  goal + dangerous   ": [("goal", np.inf), ("dangerous", 3.0)],
            "wrong    goal + lava        ": [("goal", np.inf), ("lava", 3.0)],
            "wrong    goal + meadow      ": [("goal", np.inf), ("meadow", 3.0)],
        },
    }
    t0 = time.time()
    for domain, sets in tests.items():
        print(f"\n=== {domain}: normalized reward over 5 seeds ===")
        scores = {name: [] for name in sets}
        for seed in range(5):
            env = make_env(domain, seed)
            trajs = all_trajectories(env)
            true_r = true_rewards(domain, env, trajs)
            for name, spec in sets.items():
                feats = [make_distance_feature(p, r) for p, r in spec]
                rhat, idx, _ = evaluate(feats, domain, env, trajs, true_r, seed=seed)
                scores[name].append(rhat)
                if seed == 0 and name.startswith("CORRECT"):
                    print("seed 0, path learned with the CORRECT features:")
                    print(render(env, trajs[idx]))
        for name, s in scores.items():
            s = np.array(s)
            print(f"{name}: mean {s.mean():.2f} ± {s.std(ddof=1) / np.sqrt(len(s)):.2f} (SE) | per seed {np.round(s, 2)}")
    print(f"\ntime: {time.time() - t0:.1f}s")