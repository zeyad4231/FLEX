"""2D navigation domain (paper Sec. 5 + Appendix 7.1).

9x9 grid, (0,0) = top-left, (8,8) = bottom-right. The agent moves from a start
corner to the opposite corner in 16 steps (17 positions = the paper's
length-34 trajectory vector). Objects carry text properties (e.g. "heat source")
that the GENERALIZE operation uses.
"""
from itertools import combinations
from math import comb

import numpy as np

GRID = 9
T = 17  # timesteps per trajectory (paper: 34 numbers = 17 (x, y) points)

# Object definitions. KitchenWorld properties are copied from Appendix 7.1.
# LavaLand properties are NOT given in the paper -> our own choice (see DEVIATIONS.md).
DOMAINS = {
    "kitchen": {
        "goal": {"color": "green", "type": "cross"},
        "objects": [
            {"type": "fridge", "color": "blue",
             "property": ["fridge", "light source", "icy", "stationary", "appliance"]},
            {"type": "stove", "color": "red",
             "property": ["stove", "heat source", "metal", "stationary", "appliance"]},
            {"type": "kettle", "color": "orange",
             "property": ["kettle", "portable", "heat source", "appliance"]},
        ],
    },
    "lava": {
        "goal": {"color": "green", "type": "cross"},
        "objects": [
            {"type": "meadow", "color": "green",
             "property": ["meadow", "grassy", "safe", "terrain"]},
            {"type": "lava", "color": "red",
             "property": ["lava", "dangerous", "hot", "terrain"]},
            {"type": "swamp", "color": "brown",
             "property": ["swamp", "dangerous", "wet", "terrain"]},
        ],
    },
}


def make_env(domain, seed):
    """Random environment config in the paper's format (Appendix 7.1).

    Start = a random corner, goal = the opposite corner.
    Objects = distinct random interior cells.
    """
    rng = np.random.default_rng(seed)
    spec = DOMAINS[domain]
    corners = [(0, 0), (0, 8), (8, 0), (8, 8)]
    start = corners[rng.integers(4)]
    goal = (8 - start[0], 8 - start[1])

    interior = [(x, y) for x in range(1, 8) for y in range(1, 8)]
    picks = rng.choice(len(interior), size=len(spec["objects"]), replace=False)

    env = {
        "type": "gridrobot", "X": GRID, "Y": GRID,
        "starts": [list(start)],
        "goals pos": [list(goal)],
        "goals color": spec["goal"]["color"],
        "goals type": spec["goal"]["type"],
    }
    for k, (obj, p) in enumerate(zip(spec["objects"], picks), start=1):
        env[f"objects{k} pos"] = [list(interior[p])]
        env[f"objects{k} color"] = obj["color"]
        env[f"objects{k} type"] = obj["type"]
        env[f"objects{k} property"] = list(obj["property"])
    return env


def _match(query, word):
    """Case-insensitive match that tolerates a plural 's' (e.g. 'heat sources')."""
    q, w = query.strip().lower(), word.strip().lower()
    return q == w or q == w + "s" or w == q + "s"


def get_objects_with_property(prop, environment, robot_env=None):
    """Positions of all objects whose type, color or property list matches `prop`.

    Name and signature follow Appendix 7.1 (LM-generated feature code calls this).
    """
    positions = []
    for k in range(1, 4):
        if not environment.get(f"objects{k} pos"):
            continue
        words = [environment[f"objects{k} type"], environment[f"objects{k} color"]]
        words += environment.get(f"objects{k} property", [])
        if any(_match(prop, w) for w in words):
            positions += [list(p) for p in environment[f"objects{k} pos"]]
    if _match(prop, "goal"):
        positions += [list(p) for p in environment["goals pos"]]
    return positions


def all_trajectories(env):
    """Every shortest path from start to goal, as an array (N, T, 2).

    Corner-to-opposite-corner = 16 moves = T-1, so every valid 17-step path is a
    shortest path. There are C(16, 8) = 12,870 of them: we enumerate them all,
    so planning (argmax) and the best/worst reward are exact, not sampled.
    """
    sx, sy = env["starts"][0]
    gx, gy = env["goals pos"][0]
    nx, ny = abs(gx - sx), abs(gy - sy)
    n = nx + ny
    assert n == T - 1, "start and goal must be opposite corners"
    dx, dy = (1 if gx > sx else -1), (1 if gy > sy else -1)

    out = np.zeros((comb(n, nx), T, 2))
    for k, x_steps in enumerate(combinations(range(n), nx)):
        x, y = sx, sy
        out[k, 0] = (x, y)
        x_steps = set(x_steps)
        for t in range(n):
            if t in x_steps:
                x += dx
            else:
                y += dy
            out[k, t + 1] = (x, y)
    return out


def to_flat(traj):
    """(T, 2) -> flat list [x0, y0, x1, y1, ...] (the paper's trajectory format)."""
    return np.asarray(traj).reshape(-1).tolist()


def render(env, traj=None):
    """ASCII picture: S start, G goal, 1-3 objects, * path."""
    grid = [["." for _ in range(GRID)] for _ in range(GRID)]
    if traj is not None:
        for x, y in np.asarray(traj).astype(int):
            grid[y][x] = "*"
    sx, sy = env["starts"][0]
    gx, gy = env["goals pos"][0]
    grid[sy][sx], grid[gy][gx] = "S", "G"
    legend = []
    for k in range(1, 4):
        for x, y in env.get(f"objects{k} pos", []):
            grid[y][x] = str(k)
        if env.get(f"objects{k} pos"):
            legend.append(f"{k}={env[f'objects{k} type']}")
    return "\n".join(" ".join(row) for row in grid) + "\n" + "  ".join(legend)


if __name__ == "__main__":
    for domain in ["kitchen", "lava"]:
        env = make_env(domain, seed=0)
        trajs = all_trajectories(env)
        print(f"=== {domain} (seed 0) ===")
        print(f"start={env['starts'][0]} goal={env['goals pos'][0]} | trajectories: {trajs.shape}")
        print(render(env, trajs[np.random.default_rng(0).integers(len(trajs))]))
        prop = "heat source" if domain == "kitchen" else "dangerous"
        print(f"objects with property '{prop}': {get_objects_with_property(prop, env)}\n")