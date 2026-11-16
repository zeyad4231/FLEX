"""IMPROVEMENT 2: let language change HOW MUCH a feature matters (theta), not only WHICH
features exist. The paper assumes corrections never carry information about theta
(Sec. 4: l is conditionally independent of theta), so "stay even further away" is ignored.

One extra LLM call classifies the correction:
  "which"    -> handled by FLEX as usual (ADD / GENERALIZE / MASK)
  "how_much" -> feature + direction (more / less) + strength -> scale that feature's weight
"""
import json
import re

import llm

ROUTE = """A robot's reward is a weighted sum of these features: {features}.

Earlier corrections from the user: {history}
The user just said: "{correction}"

Decide what kind of correction this is:
- "how_much": the user asks the robot to care MORE or LESS about something, using a word of degree or comparison: further, closer, more, less, even, a bit, a little, much, too much, not that, so close, etc. Example: "Stay even further away from the laptop" -> how_much (more). Example: "You can get a bit closer to the vase" -> how_much (less).
- "which": anything else. A plain instruction to avoid, approach, include or ignore something is "which", even if a related feature is already in the list. Example: "Don't spill on my laptop" -> which. Example: "Stay close to the table" -> which. Example: "Ignore the bowl" -> which. Example: "Don't spill on my phone too" -> which.

Answer in JSON only, with no other text:
{{"type": "which" or "how_much", "feature": "<exact feature name from the list, or null>", "direction": "more" or "less" or null, "strength": "slightly" or "moderately" or "strongly" or null}}"""

FACTORS = {("more", "slightly"): 1.5, ("more", "moderately"): 2.0, ("more", "strongly"): 3.0,
           ("less", "slightly"): 1 / 1.5, ("less", "moderately"): 0.5, ("less", "strongly"): 1 / 3}


def classify(feature_names, history, correction):
    """Returns a dict {"type", "feature", "direction", "strength", "factor"}."""
    answer = llm.chat(ROUTE.format(features=feature_names, history=history, correction=correction))
    m = re.search(r"\{.*\}", answer, flags=re.S)
    try:
        d = json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        d = {}
    d.setdefault("type", "which")
    d["factor"] = FACTORS.get((d.get("direction"), d.get("strength")))
    return d


def scale_theta(theta, feature_names, decision):
    """Multiply the weight of the named feature by the factor (sign kept: 'more' = care more)."""
    if decision.get("type") != "how_much" or decision.get("factor") is None:
        return theta, None
    norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
    idx = next((i for i, n in enumerate(feature_names) if norm(n) == norm(decision.get("feature"))), None)
    if idx is None:
        return theta, None
    new = theta.copy()
    new[idx] *= decision["factor"]
    return new, idx
