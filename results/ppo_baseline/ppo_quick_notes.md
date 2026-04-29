# PPO quick baseline notes

- PPO (quick baseline, untuned)
- Training steps: 20000
- Evaluation episodes: 30
- Device: `cpu`
- Environment reuse: reused the existing SUMO road topology from `code/baselines.py` and the scenario weight definitions from `code/evaluate.py`, then wrapped them in a minimal Gymnasium environment instead of adding a new simulator stack.
- State definition: concatenation of current-node one-hot, goal-node one-hot, normalized current/goal/delta coordinates, Manhattan distance, 4 local outgoing edge weights, 4 local validity-mask bits, and the normalized global directed-edge weight vector for the current scenario.
- Action definition: `Discrete(4)` over `N / E / S / W` moves on the existing 5x6 SUMO-aligned road graph; invalid moves stay in place and receive a penalty.
- Reward: per-step negative travel time `-(1 + weight / 5)`, `+20` on reaching the goal, `-2` for invalid moves, `-10` on timeout.
- Training budget: single run, fixed seed `20260409`, `max_episode_steps=16`, no sweep, no deep tuning.
- Metric definition: `Travel Time (s)` is averaged over successful evaluation episodes; `Planning Time (s)` is average PPO action-selection wall time per episode.
- Quick result snapshot: success rate=1.0, travel time=11.25, average reward=8.75, timeout rate=0.0.
- This baseline is only for RL category coverage and does not represent optimal PPO performance.
