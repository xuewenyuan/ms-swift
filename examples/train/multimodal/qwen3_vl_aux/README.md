# Qwen3-VL Multi-Task Auxiliary Skeleton

这个目录提供一个基于 `ms-swift release/3.12` 的非侵入多任务骨架，目标结构如下：

- 主输入：`text prompt + image/video`
- 主文本监督：`y_action`, `y_traj`
- 辅助监督：`y_bev`, `y_cog`, `y_god`
- 从 Qwen3-VL 的 LLM 视觉 hidden states 取 4 层：`[6, 13, 20, 27]`
- 每个辅助任务各自一套 `VGGT-style DPTHead + AuxHead + AuxLoss`
- 通过 `external_plugins` / `custom_register_path` 接入，不修改 `ms-swift` 核心训练框架

## 当前实现

- `plugin.py`
  - 注册 `train_type=qwen3vl_aux`
  - 注册 `loss_type=qwen3vl_aux`
  - 注册 `optimizer=qwen3vl_aux`
  - patch `Qwen3-VL` forward，强制返回 `hidden_states`
  - 从四层 LLM hidden states 中抽取视觉 token
  - 将视觉 token 重排为 `VGGT-style DPTHead` 需要的 patch grid
  - 构建三个分支：
    - `Bev DPTHead -> BevAuxHead -> BevAuxLoss`
    - `Cog DPTHead -> CogAuxHead -> CogAuxLoss`
    - `God DPTHead -> GodAuxHead -> GodAuxLoss`
- `dataset.py`
  - 保留 `y_bev`, `y_cog`, `y_god`
  - 支持将 `bev_label/cog_label/god_label` 映射到 `y_*`
  - 支持 `image -> images`, `video -> videos`
- `train.sh`
  - 给出最小训练命令样板

## 默认张量流

对于每个辅助任务，当前默认结构是：

```text
selected hidden states [6,13,20,27]
    -> task-specific DPTHead
    -> feature map
    -> task-specific AuxHead
    -> logits
    -> CrossEntropy loss
```

主任务仍然走 `ms-swift` 标准的 LM loss，也就是 assistant 文本里包含的 `action + trajectory` token 监督。

总 loss 形式：

```text
L = L_lm
  + λ_bev * L_bev
  + λ_cog * L_cog
  + λ_god * L_god
```

## 推荐数据格式

最直接的格式如下：

```jsonl
{"messages": [{"role": "user", "content": "<image>请输出 action 和 trajectory"}, {"role": "assistant", "content": "action: ... trajectory: ..."}], "images": ["/abs/path/a.jpg"], "y_bev": 1, "y_cog": 0, "y_god": 2}
{"messages": [{"role": "user", "content": "<video>请输出 action 和 trajectory"}, {"role": "assistant", "content": "action: ... trajectory: ..."}], "videos": ["/abs/path/a.mp4"], "y_bev": 0, "y_cog": 1, "y_god": 1}
```

如果你的列名不是 `y_bev/y_cog/y_god`，也可以先用：

```text
bev_label -> y_bev
cog_label -> y_cog
god_label -> y_god
```

## 关键参数

- `--remove_unused_columns false`
  - 必须开启，否则 `y_bev / y_cog / y_god` 不会进入模型 forward
- `QWEN3VL_AUX_HIDDEN_LAYERS`
  - 默认 `6,13,20,27`
- `QWEN3VL_AUX_DPT_PATCH_SIZE`
  - 默认 `32`，按 Qwen3-VL merged visual token 的有效 patch stride 来设
- `QWEN3VL_AUX_DPT_FEATURES`
  - 默认 `256`
- `QWEN3VL_AUX_DPT_OUT_CHANNELS`
  - 默认 `256,512,1024,1024`
- `QWEN3VL_BEV_NUM_CLASSES`
- `QWEN3VL_COG_NUM_CLASSES`
- `QWEN3VL_GOD_NUM_CLASSES`
- `QWEN3VL_BEV_LOSS_WEIGHT`
- `QWEN3VL_COG_LOSS_WEIGHT`
- `QWEN3VL_GOD_LOSS_WEIGHT`
- `QWEN3VL_AUX_HEAD_LR`
  - 默认辅助分支学习率

## 当前边界

- 当前 `AuxHead` / `AuxLoss` 先用最简单的分类头 + `CrossEntropy`
- 当前三条辅助分支各自有一套 DPTHead，不共享上采样权重
- 当前实现优先保证单图 / 单视频 / 多媒体顺序输入可跑通
- 更复杂的任务输出形式，例如 dense map、box、trajectory field、multi-label / regression，可以继续替换：
  - `BevAuxHead / CogAuxHead / GodAuxHead`
  - `BevAuxLoss / CogAuxLoss / GodAuxLoss`
