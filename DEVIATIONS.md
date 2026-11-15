# Deviations from the paper / unspecified choices

1. **No official code.** github.com/MIT-CLEAR-Lab/FLEX contains only a README (checked 12 Nov 2026). Everything is reimplemented from the paper and its Appendix prompts.
2. **LavaLand object properties** are not given in the paper. Ours: lava = [dangerous, hot, terrain], swamp = [dangerous, wet, terrain], meadow = [grassy, safe, terrain].
3. **Gridworld start/goal:** opposite corners (random which pair, per seed); objects at random interior cells. The paper says positions are randomized but not how.
4. **Candidate trajectories:** with 17 timesteps on a 9x9 grid, corner-to-corner paths are exactly the 12,870 shortest paths, so we enumerate all of them (exact planning and exact best/worst reward).
5. **`get_objects_with_property`** matches type, color or property, case-insensitive, tolerating a plural "s".
6. **Ground-truth reward** is not specified in the paper. Ours: θ* = [-0.1 · distance-to-goal (uncapped), +1.0 · distance-to-category (capped at 3)], category = heat source (KitchenWorld) / dangerous (LavaLand).
7. **Simulated expert demos:** 5 demos per seed, sampled from the Boltzmann model (Eq. 2) with rationality β = 5. With β = 1 even the correct features only reached R̂ = 0.96, so demos were too noisy to match the paper's 1.00.
8. **IRL details:** maximum likelihood over the exact set of 12,870 paths; features standardised; tiny L2 (1e-3) to keep θ finite.
9. **Layout seeding:** the random generator is seeded with (seed, domain), so KitchenWorld and LavaLand get different layouts for the same seed.
10. **Single-turn LLM calls.** The paper's code prompt says "the feature identified above" (a follow-up message in one conversation). Our calls are single-turn, so the feature name is inserted into the code prompt explicitly.
11. **2D code-generation template:** the appendix's generic template shows the Franka signature (environment, trajs, robot_env); for 2D we use the 2D inputs/example from the same appendix, with get_objects_with_property returning [x, y].
12. **GENERALIZE duplicates:** when two features generalize to the same name (e.g. laptop and phone → electronics), we keep one. The paper does not say how this is handled.
13. **Safety defaults:** an unparseable MASK answer keeps the feature; MASK is never allowed to remove every feature; generated code that crashes is sent back to the LLM with the error (max 3 tries).
## Observations

- **Model strength matters for MASK (15 Nov).** KitchenWorld "All Three", seed 0. Both gpt-4o-mini and gpt-4o did ADD (+stove) and GENERALIZE (stove → heat source) correctly. gpt-4o-mini then masked "distance to heat source" (R̂ 0.95, goal-only, correct path only by luck of the layout); gpt-4o kept it (R̂ 1.00). The MASK prompt asks whether a feature "might lead the agent to treat multiple objects the same way" and never sees the GENERALIZE intent, so it pushes against generalized features; a weaker model follows that literally.
- **Cost:** one full FLEX update ≈ 8 calls ≈ $0.024 on gpt-4o (≈ $0.0015 on gpt-4o-mini).
18. **Model and decoding:** gpt-4o served as gpt-4o-2024-08-06 (from the API response); temperature 0 for all calls. The paper names "GPT-4o" but gives no snapshot or temperature.