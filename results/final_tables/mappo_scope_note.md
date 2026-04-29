# MAPPO Scope Note

- MAPPO is citation-relevant, but it is not a lowest-cost submission item in the current repository state.
- The current codebase does not define a stable multi-agent decomposition for the 96-edge task.
- Per-agent observation spaces are not specified.
- A joint action space or shared-policy coordination interface is not specified.
- A cooperative reward function for multiple agents is not specified.
- No compatible multi-agent SUMO wrapper was found; the current live chain is single-policy / single-route oriented.
- Therefore MAPPO is deferred in this batch instead of being implemented speculatively.

Minimal next step if revisited later:

- First freeze an agentization scheme over intersections or edge clusters.
- Then define per-agent observations, shared/global state, cooperative reward, and a PettingZoo-style or equivalent SUMO wrapper before model work starts.
