# Deviations from the paper / unspecified choices

1. **No official code.** github.com/MIT-CLEAR-Lab/FLEX contains only a README (checked 12 Nov 2026). Everything is reimplemented from the paper and its Appendix prompts.
2. **LavaLand object properties** are not given in the paper. Ours: lava = [dangerous, hot, terrain], swamp = [dangerous, wet, terrain], meadow = [grassy, safe, terrain].
3. **Gridworld start/goal:** opposite corners (random which pair, per seed); objects at random interior cells. The paper says positions are randomized but not how.
4. **Candidate trajectories:** with 17 timesteps on a 9x9 grid, corner-to-corner paths are exactly the 12,870 shortest paths, so we enumerate all of them (exact planning and exact best/worst reward).
5. **`get_objects_with_property`** matches type, color or property, case-insensitive, tolerating a plural "s".