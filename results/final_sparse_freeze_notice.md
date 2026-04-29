# Final Sparse Freeze Notice

- `upstream_sparse_baseline = /root/autodl-tmp/model_merged_sparse_v2_stage4_fix`
- `patching_status = stopped`
- `exploratory_only_models:`
- `/root/autodl-tmp/model_merged_sparse_v2_stage4b_fix`
- `/root/autodl-tmp/model_merged_sparse_v2_stage4c_fix`
- `/root/autodl-tmp/model_merged_sparse_v2_stage4d_micro`
- `final_decision = FREEZE_STAGE4_FIX_AND_MOVE_ON`

## Scope
- `stage4_fix` 现在是正式冻结的 sparse 上游基线，供主实验、默认推理、GAT-v2 训练/验证、演示入口继续使用。
- `stage4b_fix`、`stage4c_fix`、`stage4d_micro` 保留为探索性负结果材料，不再作为默认上游、不再进入主链路继续 patch。

## Main-Chain Reset
- 默认运行配置已切回 `stage4_fix`：
- `/root/autodl-tmp/code/config.py`
- `/root/autodl-tmp/code/app_fixed.py`
- `/root/autodl-tmp/validate_sparse_v2.py`
- `/root/autodl-tmp/validate_sparse_v2_strict.py`
- `/root/autodl-tmp/train_gat_v2.py`
- `/root/autodl-tmp/validate_gat_v2.py`
- `/root/autodl-tmp/audit_sparse_v2_consistency.py`

## Notes
- 负结果归档见 `/root/autodl-tmp/results/final_patch_negative_results.md`。
- 模型路径审计见 `/root/autodl-tmp/results/final_model_path_audit.csv`。
