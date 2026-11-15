"""FLEX: language-guided feature editing (paper Sec. 4).

A feature is a dict {"name": str, "code": str, "fn": callable(environment, trajectory)}.
flex_update() applies ADD -> GENERALIZE -> MASK to a feature list, given one
language correction. Each step logs what it did so experiments can be inspected.
"""
import re
import sys
import types

import numpy as np

import llm
import prompts as P
from gridworld import get_objects_with_property, to_flat
from irl import make_distance_feature

# --------------------------------------------------------------- sandbox
# GPT's code does `from flexible_features.utils.env_utils import get_objects_with_property`
# (the paper's internal package). We register a fake module with that name so the
# import works and points to our own function.
_shim = types.ModuleType("flexible_features.utils.env_utils")
_shim.get_objects_with_property = get_objects_with_property
sys.modules.setdefault("flexible_features", types.ModuleType("flexible_features"))
sys.modules.setdefault("flexible_features.utils", types.ModuleType("flexible_features.utils"))
sys.modules["flexible_features.utils.env_utils"] = _shim


def compile_feature(code, env, test_traj):
    """Run GPT's code, find its function, and test it on one trajectory.
    Returns the function, or raises an error with a readable message."""
    namespace = {"np": np, "numpy": np, "get_objects_with_property": get_objects_with_property}
    exec(code, namespace)
    names = re.findall(r"^def\s+(\w+)\s*\(", code, flags=re.M)
    if not names:
        raise ValueError("no function definition found in the code")
    fn = namespace[names[-1]]
    value = float(np.asarray(fn(env, to_flat(test_traj))).reshape(-1)[0])
    if not np.isfinite(value):
        raise ValueError(f"feature returned a non-finite value: {value}")
    return fn


def _parse_code_answer(text):
    name = re.search(r'Feature name:\s*"?([^"\n]+)"?', text)
    code = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
    return (name.group(1).strip() if name else None), (code.group(1) if code else None)


def generate_feature_code(feature_name, env, test_traj, max_tries=3):
    """Ask GPT to write the code for `feature_name`; retry with the error message
    if the code crashes. Returns a feature dict."""
    base = P.GENERATE_CODE_2D.format(env=env_text(env), feature_name=feature_name,
                                     example=P.EXAMPLE_DISTANCE_2D)
    prompt, error = base, None
    for _ in range(max_tries):
        answer = llm.chat(prompt, system=P.SYSTEM_2D)
        _, code = _parse_code_answer(answer)
        try:
            if code is None:
                raise ValueError("no ```python code block found in the answer")
            fn = compile_feature(code, env, test_traj)
            return {"name": feature_name, "code": code, "fn": fn}
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            prompt = base + P.CODE_RETRY.format(error=error)
    raise RuntimeError(f"could not generate working code for '{feature_name}': {error}")


# --------------------------------------------------------------- helpers
def hand_feature(name, prop, r=3.0):
    """A hand-written starting feature (used for the initial, misaligned feature sets)."""
    return {"name": name, "code": f"make_distance_feature({prop!r}, r={r})",
            "fn": make_distance_feature(prop, r)}


def env_text(env):
    """Environment config as text, in the style of Appendix 7.1."""
    lines = ["env:"]
    for k, v in env.items():
        lines.append(f"{k}: {v!r}" if isinstance(v, str) else f"{k}: {v}")
    return "\n".join(lines)


def names(features):
    return [f["name"] for f in features]


def _same(a, b):
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
    return norm(a) == norm(b)


# -------------------------------------------------------------------- ADD
def op_add(features, env, test_traj, instruction, history, correction, log, verifier=None):
    prompt = P.ADD_GENERATE_FEATURE.format(env=env_text(env), instruction=instruction,
                                           history=history, correction=correction,
                                           features=names(features))
    answer = llm.chat(prompt, system=P.SYSTEM_2D)
    if "NO NEW FEATURE" in answer.upper():
        log.append(("ADD", "no new feature"))
        return features
    first_line = next(l for l in answer.strip().splitlines() if l.strip())
    new_name = re.sub(r'^(feature( name)?:)?\s*["*`]*|["*`.]*$', "", first_line, flags=re.I).strip()
    if any(_same(new_name, n) for n in names(features)):
        log.append(("ADD", f"proposed '{new_name}' already exists -> skipped"))
        return features
    try:
        feat = generate_feature_code(new_name, env, test_traj)
    except RuntimeError as e:
        log.append(("ADD", f"code generation failed for '{new_name}' -> skipped ({e})"))
        return features
    log.append(("ADD", f"+ {new_name}"))
    return features + [feat]


# ------------------------------------------------------------- GENERALIZE
def op_generalize(features, env, test_traj, instruction, history, correction, log, verifier=None):
    common = dict(env=env_text(env), instruction=instruction, history=history, correction=correction)
    intent = llm.chat(P.GEN_EXTRACT_INTENT.format(**common), system=P.SYSTEM_2D).strip()
    log.append(("GENERALIZE", f"intent: {intent}"))

    answer = llm.chat(P.GEN_FEATURE_NAMES.format(intent=intent, features=names(features), **common),
                      system=P.SYSTEM_2D)
    renames = {}
    for line in answer.splitlines():
        if "->" not in line:
            continue
        old, new = [s.strip().strip('"*`-').strip() for s in line.split("->", 1)]
        if "NO CHANGE" in new.upper() or _same(old, new):
            continue
        match = next((n for n in names(features) if _same(n, old)), None)
        if match:
            renames[match] = new

    out, seen = [], set()
    for f in features:
        old_name, new_name = f["name"], renames.get(f["name"])
        if new_name:
            try:
                f = generate_feature_code(new_name, env, test_traj)
                log.append(("GENERALIZE", f"{old_name} -> {new_name}"))
            except RuntimeError as e:
                log.append(("GENERALIZE", f"code generation failed for '{new_name}' -> kept '{old_name}'"))
        key = re.sub(r"[^a-z0-9]", "", f["name"].lower())
        if key in seen:  # two features generalized to the same name -> keep one (DEVIATIONS.md)
            log.append(("GENERALIZE", f"merged duplicate '{f['name']}'"))
            continue
        seen.add(key)
        out.append(f)
    if not renames:
        log.append(("GENERALIZE", "no change"))
    return out


# -------------------------------------------------------------------- MASK
def op_mask(features, env, test_traj, instruction, history, correction, log, verifier=None):
    keep = []
    for f in features:
        prompt = P.MASK_ELEMENT.format(env=env_text(env), instruction=instruction, history=history,
                                       correction=correction, features=names(features), feature=f["name"])
        answer = llm.chat(prompt, system=P.SYSTEM_2D).strip()
        digit = re.search(r"[01]", answer)
        relevant = digit is None or digit.group() == "1"  # unparseable -> keep (safe default)
        if not relevant and verifier is not None:  # IMPROVEMENT 3: demos must agree
            allowed, lr = verifier(features, f)
            if not allowed:
                log.append(("MASK-CHECK", f"refused removing '{f['name']}': demos depend on it (LR={lr:.1f})"))
                relevant = True
            else:
                log.append(("MASK-CHECK", f"allowed removing '{f['name']}' (LR={lr:.1f})"))
        if relevant:
            keep.append(f)
        else:
            log.append(("MASK", f"- {f['name']}"))
    if not keep:  # never return an empty feature set
        log.append(("MASK", "would remove everything -> kept original set"))
        return features
    if len(keep) == len(features):
        log.append(("MASK", "no change"))
    return keep


# ------------------------------------------------------------------ FLEX
OPS = {"add": op_add, "generalize": op_generalize, "mask": op_mask}


def flex_update(features, env, test_traj, instruction, history, correction,
                ops=("add", "generalize", "mask"), verifier=None):
    """One FLEX round: apply the chosen operations in order (paper: ADD -> GENERALIZE -> MASK).
    ops=("add",) gives the ALGAE-style add-only baseline.
    verifier: optional demo check for MASK (improvement 3); None = the paper's FLEX."""
    log = []
    for op in ops:
        features = OPS[op](features, env, test_traj, instruction, history, correction, log, verifier=verifier)
    return features, log


# ------------------------------------------------------------- quick demo
if __name__ == "__main__":
    from gridworld import make_env, all_trajectories, render
    from irl import true_rewards, evaluate

    # RQ1 "All Three" scenario (Table 2): start with [goal, fridge]
    domain, seed = "kitchen", 0
    env = make_env(domain, seed)
    trajs = all_trajectories(env)
    true_r = true_rewards(domain, env, trajs)

    features = [hand_feature("distance to goal", "goal", np.inf), hand_feature("distance to fridge", "fridge")]
    instruction = "Navigate to the goal."
    correction = "Don't burn yourself on the stove."

    rhat, idx, _ = evaluate([f["fn"] for f in features], domain, env, trajs, true_r, seed=seed)
    print(f"BEFORE: {names(features)} -> R_hat = {rhat:.2f}")

    features, log = flex_update(features, env, trajs[0], instruction, [], correction)
    for step, msg in log:
        print(f"  [{step}] {msg}")

    rhat, idx, _ = evaluate([f["fn"] for f in features], domain, env, trajs, true_r, seed=seed)
    print(f"AFTER:  {names(features)} -> R_hat = {rhat:.2f}")
    print(render(env, trajs[idx]))
    for f in features:
        print(f"\n--- code for '{f['name']}' ---\n{f['code']}")
    llm.print_usage()
