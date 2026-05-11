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
  - 调用公共 `feature_extractor`
  - 将视觉特征和任务配置转交给 `AuxHead/AuxLoss`
- `common/`
  - `visual_feature_extractor.py`
    - 负责从 Qwen3-VL 输出中抽取视觉 hidden states
    - 负责按 image/video token span 重组媒体级输入
  - `vggt_upsampler.py`
    - 提供 `VGGTUpsampler` 接口
    - 后续由 `AuxSeparateHeads` 内部调用
  - `config.py`
    - 负责统一加载辅助任务参数
- `aux_head/`
  - `BevAuxHead / CogAuxHead / GodAuxHead` 只保留接口定义
  - `AuxSeparateHeads` 统一管理三类 head
- `aux_loss/`
  - `BevAuxLoss / CogAuxLoss / GodAuxLoss` 只保留接口定义
  - `AuxSeparateLosses` 统一管理三类 loss
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
    -> feature extractor
    -> task-specific AuxHead (internally calls VGGTUpsampler)
    -> predictions
    -> task-specific AuxLoss
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
- `QWEN3VL_AUX_CONFIG_PATH`
  - 推荐通过一个 JSON 文件集中传辅助任务参数
- `QWEN3VL_AUX_CONFIG`
  - 也支持直接传 JSON 字符串，更适合临时调试

推荐把辅助任务参数分成两层：

```json
{
  "shared": {
    "layer_indices": [6, 13, 20, 27],
    "merge_size": 2,
    "lm_loss_weight": 1.0,
    "upsampler_cfg": {
      "patch_size": 32,
      "features": 256,
      "out_channels": [256, 512, 1024, 1024]
    }
  },
  "tasks": {
    "bev": {
      "label_key": "y_bev",
      "loss_weight": 1.0,
      "head": {},
      "loss": {}
    },
    "cog": {
      "label_key": "y_cog",
      "loss_weight": 1.0,
      "head": {},
      "loss": {}
    },
    "god": {
      "label_key": "y_god",
      "loss_weight": 1.0,
      "head": {},
      "loss": {}
    }
  }
}
```

这种传法更合适，因为：

- shared 参数只写一次，比如 `layer_indices / merge_size / upsampler_cfg`
- 每个任务只维护自己的 `label_key / loss_weight / head / loss`
- plugin 不需要为了新任务参数继续扩展新的环境变量

## 当前边界

- 当前 `AuxHead` / `AuxLoss` 已经改成接口骨架，本仓库里不包含你的具体任务实现
- 当前 `VGGTUpsampler` 也是接口骨架，具体实现由 `AuxSeparateHeads` 内部接入
- 当前实现优先保证单图 / 单视频 / 多媒体顺序输入可跑通
- 你后续只需要补齐这几个接口的具体实现：
  - `BevAuxHead / CogAuxHead / GodAuxHead`
  - `BevAuxLoss / CogAuxLoss / GodAuxLoss`
