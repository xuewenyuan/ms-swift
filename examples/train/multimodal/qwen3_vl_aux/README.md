# Qwen3-VL Multi-Task Auxiliary Skeleton

这个目录提供一个基于 `ms-swift release/3.12` 的非侵入多任务骨架，目标结构如下：

- 主输入：`text prompt + image/video`
- 主文本监督：`y_action`, `y_traj`
- 辅助监督：`y_bev`, `y_cog`, `y_god`
- 辅助标签可以直接放对象，也可以放一个地址字段再懒加载
- 从 Qwen3-VL 的 LLM 视觉 hidden states 取 4 层：`[6, 13, 20, 27]`
- 每个辅助任务各自一套 `VGGT-style DPTHead + AuxHead + AuxLoss`
- 通过 `external_plugins` 接入，不修改 `ms-swift` 核心训练框架

## 当前实现

- `plugin.py`
  - 注册 `train_type=qwen3vl_aux`
  - 注册 `loss_type=qwen3vl_aux`
  - 注册 `optimizer=qwen3vl_aux`
  - patch `Qwen3-VL` forward，强制返回 `hidden_states`
  - 调用公共 `label_loader`
  - 一次性调用 `AuxSeparateHeads`
  - 再按 task 拆分 `bev/god/cog` 的 loss
  - aux 训练默认关闭 LM 自回归 CE，只优化辅助 loss
  - 保存 LoRA adapter、ViT/aligner 全参增量和 aux head；纯 Aux warmup 可关闭 LoRA，只保存 aux head
- `common/`
  - `label_loader.py`
    - 负责把 `labels_path` 或 task-specific 路径懒加载成实际 label 对象
    - 负责把 numpy / dict / list 递归整理成 batch 级结构
  - `token_feature_adapter.py`
    - 负责从 `hidden_states` 拆分 `pv / bev / navi` token
    - 负责把 BEV token 还原成后续 upsampler 需要的融合特征
  - `vggt_upsampler.py`
    - 提供 `VGGTUpsampler` 接口
    - 由 `AuxSeparateHeads` 在 task head 前统一调用
  - `config.py`
    - 负责统一加载辅助任务参数
- `aux_heads.py`
  - 和 `plugin.py` 同层
  - 保留原始的多任务总控式 `AuxSeparateHeads`
  - 内部统一完成 token 拆分、upsampler、各 task head 前向
- `aux_losses.py`
  - 和 `plugin.py` 同层
  - 统一管理三类 loss 的注册和 task-specific payload 包装
- `aux_task/`
  - 按任务拆分 `bev / cog / god`
  - 每个任务目录下放自己的 `AuxHead` 和 `AuxLoss`
- `train.sh`
  - 给出最小辅助任务训练命令样板
- `merge_aux_to_lm.sh`
  - 将辅助阶段学到的 LoRA 和 ViT/aligner 全参合入 base model，输出可继续训练 LM 的普通模型目录

## 默认张量流

对于每个辅助任务，当前默认结构是：

```text
Qwen3-VL hidden_states
    -> token_feature_adapter
    -> AuxSeparateHeads
    -> VGGTUpsampler
    -> task-specific AuxHead
    -> predictions
    -> label loader
    -> task-specific AuxLoss
```

辅助任务训练默认不走 `ms-swift` 标准的 LM loss，也就是 assistant 文本里的 `action + trajectory` token 不参与自回归 CE。

总 loss 形式：

```text
L = λ_lm * L_lm
  + λ_bev * L_bev
  + λ_cog * L_cog
  + λ_god * L_god
```

其中 `λ_lm` 默认是 `0.0`。如果需要辅助任务和 LM 同训，可以把 `shared.lm_loss_weight` 或环境变量 `QWEN3VL_AUX_LM_WEIGHT` 设成非 0。任务级 `λ_bev / λ_cog / λ_god` 推荐写在 `tasks.<task>.train_weight`；旧配置里的外层 `tasks.<task>.loss_weight` 仍然兼容。`tasks.<task>.loss.loss_weight` 属于任务内部 loss 聚合，不建议用它控制任务是否进入最终训练目标。

当 `average_tokens_across_devices=True` 时，SWIFT trainer 会在 custom loss 返回后再乘以进程数；插件会对返回给 trainer 的 loss 做反向补偿，日志里的 `loss` 应该与 `objective_loss` 同量级，`loss_scale_factor` 只用于确认这个补偿因子。

推荐分两阶段训练：

```bash
# 阶段 1：只训练 Aux Head
QWEN3VL_AUX_DISABLE_LORA=1 \
QWEN3VL_AUX_LM_WEIGHT=0 \
swift sft ...

# 阶段 2：重新初始化 LoRA，并加载阶段 1 的 Aux Head
QWEN3VL_AUX_INIT_HEADS_FROM=output/qwen3_vl_aux/checkpoint-100 \
QWEN3VL_AUX_LM_WEIGHT=1 \
swift sft ...
```

## 推荐数据格式

最直接的格式如下：

```jsonl
{"messages": [{"role": "user", "content": "<image>请输出 action 和 trajectory"}, {"role": "assistant", "content": "action: ... trajectory: ..."}], "images": ["/abs/path/a.jpg"], "y_bev": 1, "y_cog": 0, "y_god": 2}
{"messages": [{"role": "user", "content": "<video>请输出 action 和 trajectory"}, {"role": "assistant", "content": "action: ... trajectory: ..."}], "videos": ["/abs/path/a.mp4"], "y_bev": 0, "y_cog": 1, "y_god": 1}
```

如果你的辅助标签在外部文件里，更推荐这样：

```jsonl
{"messages": [{"role": "user", "content": "<image>请输出 action 和 trajectory"}, {"role": "assistant", "content": "action: ... trajectory: ..."}], "images": ["/abs/path/a.jpg"], "labels_path": "obs://.../labels.zstd.mspack"}
```

这时训练时会：

- 在 `plugin.py` 的 `forward` 中调用 `common/label_loader.py`
- 默认按 `labels[2][0]` 取出 `src_label`
- 再把 batched label 直接传给 `AuxSeparateHeads` 和各 task loss

## 关键参数

- `--remove_unused_columns false`
  - 必须开启，否则 `y_bev / y_cog / y_god` 不会进入模型 forward
- `QWEN3VL_AUX_HIDDEN_LAYERS`
  - 默认 `6,13,20,27`
- `QWEN3VL_AUX_LM_WEIGHT`
  - 默认 `0`，表示训练辅助任务时关闭 LM 自回归监督
- `QWEN3VL_AUX_DISABLE_LORA`
  - 默认 `0`。设为 `1` 时不注入任何 LoRA，适合第一阶段只训练 Aux Head
- `QWEN3VL_AUX_INIT_HEADS_FROM`
  - 可选。指向第一阶段 checkpoint 目录，直接加载其中的 `aux_head_config.json` 和 `aux_trainables.safetensors`，适合第二阶段从 Aux Head warmup 结果开始联合训练
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
- `--vit_lr` / `--aligner_lr`
  - 给 ViT / aligner 参数组设置独立学习率；`--freeze_vit false` / `--freeze_aligner false` 会让对应模块进入 LoRA target，`QWEN3VL_AUX_TRAIN_VISION=1` / `QWEN3VL_AUX_TRAIN_ALIGNER=1` 才会强制全参训练
- `QWEN3VL_AUX_LOAD_HEADS`
  - 加载 aux checkpoint 时是否加载 aux head。训练恢复默认 `true`，export/merge 默认 `false`
- `QWEN3VL_AUX_CONFIG_PATH`
  - 推荐通过一个 JSON 文件集中传辅助任务参数
- `QWEN3VL_AUX_CONFIG`
  - 也支持直接传 JSON 字符串，更适合临时调试
- 显式设置的环境变量会覆盖 JSON 配置，适合临时调整单个实验参数
- `QWEN3VL_AUX_ENABLED_TASKS`
  - 可选，逗号分隔，比如 `bev,god`
  - 不传时默认按 `tasks.<name>.enabled` 配置生效
- `QWEN3VL_AUX_LABEL_PATH_KEY`
  - 默认 `labels_path`
- `QWEN3VL_AUX_LABEL_FORMAT`
  - 默认 `mspack`
- `QWEN3VL_AUX_LABEL_CACHE_SIZE`
  - 默认 `256`

推荐把辅助任务参数分成两层：

```json
{
  "shared": {
    "enabled_tasks": ["bev", "god"],
    "layer_indices": [6, 13, 20, 27],
    "merge_size": 2,
    "lm_loss_weight": 0.0,
    "label_loader": {
      "path_key": "labels_path",
      "format": "mspack",
      "src_label_path": [2, 0],
      "cache_size": 256
    },
    "upsampler_cfg": {
      "type": "updeconv",
      "patch_size": 32,
      "features": 256,
      "out_channels": [256, 512, 1024, 1024]
    }
  },
  "tasks": {
    "bev": {
      "enabled": true,
      "label_key": "y_bev",
      "label_keys": [],
      "train_weight": 1.0,
      "head": {},
      "loss": {}
    },
    "cog": {
      "enabled": false,
      "label_key": "y_cog",
      "label_keys": [],
      "train_weight": 1.0,
      "head": {},
      "loss": {}
    },
    "god": {
      "enabled": true,
      "label_key": "y_god",
      "label_keys": [],
      "train_weight": 1.0,
      "head": {},
      "loss": {}
    }
  }
}
```

这种传法更合适，因为：

- shared 参数只写一次，比如 `layer_indices / merge_size / upsampler_cfg`
- 可以在 `shared.enabled_tasks` 或 `QWEN3VL_AUX_ENABLED_TASKS` 里统一指定本次启用哪些任务
- 每个任务只维护自己的 `label_key / label_keys / train_weight / head / loss`
- 可以用 `enabled=false` 临时关闭还没实现完的任务，比如当前先跳过 `cog`
- plugin 不需要为了新任务参数继续扩展新的环境变量

## 当前边界

- 当前 `AuxHead` / `AuxLoss` 已经改成接口骨架，本仓库里不包含你的具体任务实现
- 当前 `VGGTUpsampler` 仍然是接口骨架，但构造方式已经兼容原始 `AuxSeparateHeads`
- 当前实现优先保证单图 / 单视频 / 多媒体顺序输入可跑通
- aux 阶段 checkpoint 中：
  - 开启 LoRA 时，标准 adapter 文件保存 LoRA
  - `QWEN3VL_AUX_DISABLE_LORA=1` 时不生成 adapter 文件
  - `vit.safetensors` 保存 ViT/aligner 全参训练结果
  - `aux_trainables.safetensors` 只保存 aux head / aux loss 相关权重
- Aux Head warmup 后做 LM + Aux 联合训练时，不需要 merge。第二阶段用基础模型正常初始化 LoRA，并设置 `QWEN3VL_AUX_INIT_HEADS_FROM=/path/to/stage1/checkpoint` 加载第一阶段 aux 权重
- 第一阶段与第二阶段的可训练参数拓扑不同，因此第二阶段不要把第一阶段目录传给 `--resume_from_checkpoint`
- 只有需要输出不带 aux head 的普通 LM 模型时，才运行 `merge_aux_to_lm.sh`
- 你后续只需要补齐这几个接口的具体实现：
  - `aux_task/bev/bev_aux_head.py`, `aux_task/bev/bev_aux_loss.py`
  - `aux_task/cog/cog_aux_head.py`, `aux_task/cog/cog_aux_loss.py`
  - `aux_task/god/god_aux_head.py`, `aux_task/god/god_aux_loss.py`
