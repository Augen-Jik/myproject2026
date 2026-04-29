# stable-baselines3 install notes

- Initial environment check in this session: `stable-baselines3`, `gymnasium`, and `gym` were not installed.
- Install command used:
```bash
python -m pip install stable-baselines3 gymnasium
```
- Installed version: `stable_baselines3==2.8.0`
- Installed version: `gymnasium==1.2.3`
- `gym` is still unavailable after setup.
- SUMO-side dependencies already present in the environment: `sumolib`, `traci`, and `/usr/bin/sumo`.
- Re-run command for the baseline:
```bash
python /root/autodl-tmp/rl_baseline/run_ppo_quick_baseline.py
```
