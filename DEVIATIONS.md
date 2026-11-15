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