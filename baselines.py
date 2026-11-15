"""Baselines from the paper (Sec. 5, Appendix 7.2).

- ALGAE (add only):  flex.flex_update(..., ops=("add",))   [approximation, see DEVIATIONS.md]
- MFE  (Mask-From-Exhaustive): LM lists every plausible feature, then the FLEX MASK step removes irrelevant ones
- USF  (Unscaffolded): one LM call returns the whole new feature list
"""
import re

import llm
import prompts as P
from flex import env_text, names, generate_feature_code, op_mask, flex_update, _same


GENERIC = {"computation", "confidence", "reasoning", "formula", "explanation", "description",
           "rationale", "calculation", "method", "justification", "note", "notes", "example"}


def parse_feature_list(text, max_features=8):
    """Pull feature names out of a free-text list such as
    '1. **Distance to stove** (confidence: 0.9)' while ignoring indented
    sub-bullets like '   - Computation: ...' or '   - Confidence: 0.9'."""
    items = []
    for line in text.splitlines():
        m = re.match(r"^(\s*)(\d+[.)]|[-*•])\s+(.*)$", line)
        if m:
            items.append((len(m.group(1).expandtabs(4)), m.group(2)[0].isdigit(), m.group(3)))
    if not items:
        return []
    # keep only top-level items: numbered ones if the list is numbered, else the least-indented bullets
    if any(numbered for _, numbered, _ in items):
        items = [it for it in items if it[1] and it[0] < 2]
    else:
        min_indent = min(ind for ind, _, _ in items)
        items = [it for it in items if it[0] == min_indent]
    out = []
    for _, _, name in items:
        name = re.sub(r"[*`\"]", "", name)
        name = re.split(r"\s*(?:\(|:|\s-\s|–|—|,\s*confidence)", name, maxsplit=1)[0].strip().rstrip(".")
        if 3 <= len(name) <= 80 and name.lower() not in GENERIC and not any(_same(name, n) for n in out):
            out.append(name)
    return out[:max_features]


def _build(feature_names, existing, env, test_traj, log, tag):
    """Reuse existing features with the same name; generate code for new ones."""
    out = []
    for n in feature_names:
        old = next((f for f in existing if _same(f["name"], n)), None)
        if old:
            out.append(old)
            continue
        try:
            out.append(generate_feature_code(n, env, test_traj))
            log.append((tag, f"+ {n}"))
        except RuntimeError:
            log.append((tag, f"code generation failed for '{n}' -> skipped"))
    return out


def algae_update(features, env, test_traj, instruction, history, correction):
    return flex_update(features, env, test_traj, instruction, history, correction, ops=("add",))


def mfe_update(features, env, test_traj, instruction, history, correction):
    log = []
    answer = llm.chat(P.MFE_EXHAUSTIVE.format(env=env_text(env), features=names(features)), system=P.SYSTEM_2D)
    proposed = [n for n in parse_feature_list(answer) if not any(_same(n, f["name"]) for f in features)]
    exhaustive = features + _build(proposed, features, env, test_traj, log, "MFE-LIST")
    kept = op_mask(exhaustive, env, test_traj, instruction, history, correction, log)
    return kept, log


def usf_update(features, env, test_traj, instruction, history, correction):
    log = []
    answer = llm.chat(P.USF_UPDATE.format(env=env_text(env), instruction=instruction, history=history,
                                          correction=correction), system=P.SYSTEM_2D)
    new_names = parse_feature_list(answer)
    new = _build(new_names, features, env, test_traj, log, "USF")
    if not new:
        log.append(("USF", "no usable features in the answer -> kept original set"))
        return features, log
    log.append(("USF", f"new set: {names(new)}"))
    return new, log


def flex_full(features, env, test_traj, instruction, history, correction):
    return flex_update(features, env, test_traj, instruction, history, correction)


METHODS = {"FLEX": flex_full, "ALGAE": algae_update, "MFE": mfe_update, "USF": usf_update}
