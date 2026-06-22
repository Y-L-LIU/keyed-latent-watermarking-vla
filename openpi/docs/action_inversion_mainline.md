# Action Inversion Mainline README

这份文档只覆盖当前主线:

- `full reverse`
- `partial channel` (`--fm-channel-inverse`)

不覆盖 telemetry-only 分支。
脚本里仍然保留 `--fm-latent-map` 和 `--fm-latent-posterior` 这类实验路径, 但这份 README 不展开。

## 1. 主线脚本

在线 rollout + 反演 + 检测:

- `scripts/eval_libero_action_inversion.py`

读取已保存 rollout 做离线重打分:

- `scripts/eval_saved_libero_action_inversion.py`

跨多个 suite 的轻量聚合汇总:

- `scripts/eval_partial_saved_libero_action_inversion.py`

## 2. 两条主线怎么选

### `full reverse`

默认路径, 不加任何 `fm-*` flag。

流程:

1. 策略输出完整 `raw_actions [T, 32]`
2. 直接用 old reverse `_recover_noise_from_actions(...)`
3. 在 noise domain 做 watermark detector scoring

适合:

- 有完整 action chunk
- 想跑最稳的基线
- 想和历史结果直接对齐

### `partial channel`

打开 `--fm-channel-inverse`。

流程:

1. 只观测 env-visible 前 `7` 个动作通道
2. 用 channel-only FM solver 先补全成完整 `raw_actions [T, 32]`
3. 再接现有 old reverse
4. 再做 detector scoring

适合:

- 环境侧只能拿到 `a_env[:, :, :7]`
- 想保持 old reverse 和 detector 不变

注意:

- 这条路径里 solver 的内部状态始终是 `[B, T, 32]`
- 只是 observation loss 只比较前 `7` 个通道

## 3. 最常用调用

下面示例都假设你已经在 `openpi` 目录里。

### 3.1 Full reverse smoke run

```bash
python scripts/eval_libero_action_inversion.py \
  --checkpoint-dir /path/to/checkpoint \
  --task-suite-name libero_goal \
  --task-offset 0 \
  --num-tasks 1 \
  --num-trials-per-task 1 \
  --save-rollout-dir /tmp/libero_full_reverse_rollouts \
  --save-report-dir /tmp/libero_full_reverse_reports \
  --beta 0.2 \
  --reference-mode gaussian \
  --chunk-selection-strategy stateful_online \
  --chunk-selection-count 5
```

### 3.2 Partial channel smoke run

```bash
python scripts/eval_libero_action_inversion.py \
  --checkpoint-dir /path/to/checkpoint \
  --task-suite-name libero_goal \
  --task-offset 0 \
  --num-tasks 1 \
  --num-trials-per-task 1 \
  --save-rollout-dir /tmp/libero_partial_channel_rollouts \
  --save-report-dir /tmp/libero_partial_channel_reports \
  --beta 0.2 \
  --reference-mode gaussian \
  --chunk-selection-strategy stateful_online \
  --chunk-selection-count 5 \
  --fm-channel-inverse \
  --obs-sigma 1e-4 \
  --fm-guide-scale 0.5 \
  --fm-guide-schedule linear_decay
```

### 3.3 对已保存 rollout 做离线重打分

```bash
python scripts/eval_saved_libero_action_inversion.py \
  --rollout-dir /tmp/libero_full_reverse_rollouts/task_rollout \
  --output-dir /tmp/libero_full_reverse_rescore \
  --false-key-count 31 \
  --group-sizes 1 2 4 5 8 12 \
  --inversion-steps 1 2 3 4 5 6 7 8
```

### 3.4 对多个 suite 做轻量聚合汇总

```bash
python scripts/eval_partial_saved_libero_action_inversion.py \
  --suite goal=/path/to/goal_rollouts/task_rollout \
  --suite object=/path/to/object_rollouts/task_rollout \
  --suite spatial=/path/to/spatial_rollouts/task_rollout \
  --output-path /tmp/libero_partial_summary.json \
  --false-key-count 15 \
  --group-sizes 1 2 4 5 8 12 \
  --step-count 8
```

## 4. 关键参数怎么理解

### 4.1 模型和任务范围

- `--checkpoint-dir`
  模型 checkpoint 路径或 `gs://...` 路径。
- `--config-name`
  默认是 `pi05_libero`。
- `--task-suite-name`
  常见取值: `libero_spatial`, `libero_object`, `libero_goal`。
- `--task-offset`
  从 suite 的第几个 task 开始跑。
- `--num-tasks`
  连续跑多少个 task。
- `--num-trials-per-task`
  每个 task 跑多少次 episode。

### 4.2 rollout 控制

- `--replan-steps`
  每次 replanning 实际执行多少步。默认 `5`。
- `--max-rollout-steps`
  限制整条 episode 最多执行多少步。
- `--num-steps-wait`
  rollout 前等待步数。
- `--resize-size`
  图像 resize 尺寸。默认 `224`。
- `--eval-mode`
  `task_rollout` 或 `probe_verification`。主线一般用 `task_rollout`。

### 4.3 watermark / reference

- `--secret-key`
  真 key。
- `--beta`
  watermark 强度。脚本默认是 `0.02`，但很多实验会显式用 `0.2`。
- `--freq-min-hz`, `--freq-max-hz`
  reference 频带范围。
- `--n-tones`
  keyed reference 的 tone 数。
- `--reference-mode`
  `bandpass` 或 `gaussian`。

经验上:

- 如果 `score-step-scope=executed`，而且每个 selected window 很短，优先用 `gaussian`
- `bandpass` 更适合完整 chunk 或更长窗口

### 4.4 selected windows

- `--chunk-selection-strategy`
  `periodic`, `fixed_slots`, `stateful_online`
- `--chunk-selection-period`
  `periodic` 模式的周期
- `--chunk-selection-count`
  每个周期选几个, 或在线总预算
- `--chunk-selection-total-slots`
  `fixed_slots` 模式总 slot 数

经验上:

- `periodic + period=1 + count=1` 等价于全选
- 如果你想每条 episode 只挑少量 window, 常用:
  `--chunk-selection-strategy stateful_online --chunk-selection-count 5`

### 4.5 detector / scoring

- `--detector`
  `cosine`, `dot`, `mse`, `coherence`, `wmf`, `ace`
- `--score-step-scope`
  `executed` 或 `full_chunk`
- `--window-aggregator`
  `sum` 或 `mean`
- `--max-score-windows`
  最多用多少个 selected windows 去打分
- `--null-decoy-count`
  false-key normalization 时的 decoy 个数
- `--subspace-rank`
  `wmf` / `ace` 的子空间秩
- `--target-fpr`
  summary 里 `tpr@fpr` 的目标 FPR
- `--threshold`
  手动阈值

### 4.6 old reverse

- `--num-inversion-steps`
  reverse ODE 的步数
- `--save-recovered-noise-cache-steps`
  额外保存不同步数下的 recovered noise, 供离线分析
- `--inversion-method`
  `reverse` 或 `reverse_refine`
- `--refinement-steps`
  refine 优化步数
- `--refinement-learning-rate`
  refine 学习率
- `--refinement-latent-l2`
  latent 正则
- `--refinement-init-l2`
  离初始解的 trust-region 正则

## 5. Partial channel 专用参数

只在 `--fm-channel-inverse` 打开时有意义。

- `--obs-sigma`
  channel observation loss 的噪声尺度
- `--fm-guide-scale`
  每步梯度引导强度
- `--fm-guide-schedule`
  `const` 或 `linear_decay`

当前实现假设:

- 观测是 `a_env[:, :, :7]`
- 内部状态是 `a_raw [B, T, 32]`
- solver 先补全完整动作, 再走 old reverse

## 6. 离线重打分脚本参数

### `eval_saved_libero_action_inversion.py`

- `--rollout-dir`
  在线 rollout 保存的 `.npz` 所在目录
- `--output-dir`
  离线重打分输出目录
- `--candidate-key`
  只看某个候选 key, 不给就会自动构造 true key + false keys
- `--false-key-count`
  false keys 数量
- `--group-sizes`
  多轨迹聚合时的 group size
- `--group-samples`
  每个 group size 抽样多少次
- `--inversion-steps`
  用缓存的哪些 inversion step 去重打分

### `eval_partial_saved_libero_action_inversion.py`

- `--suite`
  形式是 `name=/abs/rollout_dir`
- `--output-path`
  汇总 JSON 输出路径
- `--false-key-count`
  false keys 数量
- `--group-sizes`
  聚合 group size
- `--group-samples`
  每个 size 采样多少组
- `--step-count`
  使用哪个 cached inversion step

## 7. 推荐起手配置

如果只是先看主线能不能跑通, 推荐：

```bash
--beta 0.2
--reference-mode gaussian
--chunk-selection-strategy stateful_online
--chunk-selection-count 5
```

然后二选一：

- full reverse: 什么 `fm-*` 都不加
- partial channel: 加 `--fm-channel-inverse --obs-sigma 1e-4 --fm-guide-scale 0.5 --fm-guide-schedule linear_decay`

## 8. 结果怎么看

在线 summary 常看这些值：

- `watermarked_scores mean`
- `plain_scores mean`
- `wrong_key_scores mean`
- `roc_auc`
- `pairwise_wm_gt_plain_accuracy`
- `tpr_at_1pct_fpr`
- `plain_recovery_rms_mean`
- `watermarked_recovery_rms_mean`

粗略判断:

- `watermarked > plain > wrong_key` 是健康方向
- `roc_auc` 越高越好
- `recovery_rms` 越小通常越稳

## 9. 远端运行建议

这个仓库是 remote-first。

如果环境敏感, 优先用仓库根目录的远端工作流：

- `bin/remote-sync-check`
- `bin/remote-exec`
- `make remote-*`

文档里的命令为了可读性都写成了直接 `python ...`。实际跑大任务时，建议包在远端执行层里。
