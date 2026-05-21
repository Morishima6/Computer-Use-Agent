# Phase 3 Builder

先运行 CLI：

```bash
$env:SILICONFLOW_API_KEY="sk-ytwgqxhcywsszqqhuzuulsmcyslmoycivfdzlsctevuqruzc"
python -m Core-Agent.trajectory.cu_generator.builder.phase3_builder --users user_1
```

如果需要显式指定输入和输出目录，可以使用：

```bash
python -m trajectory.cu_generator.builder.phase3_builder \
  --segments-root trajectory/cu_generator/segments \
  --output-dir trajectory/cu_base \
  --users user_1 \
  --similarity-threshold 0.82 \
  --embedding-model Qwen/Qwen3-Embedding-8B
```

构建过程中现在会输出阶段日志和长循环进度，便于观察当前走到哪一步。

在 `cluster` 阶段，builder 现在会把每个 `abstract_intent` 的 embedding 缓存到
`segments/<user>/abstract_intent_embeddings.json`。
后续再次运行 Phase 3 时，会优先复用这些缓存，只补缺失或失效的条目，从而减少重复的远端 embedding 请求。

如果构建中断，可以从输出目录下的 checkpoint 恢复：

```bash
python -m Core-Agent.trajectory.cu_generator.builder.phase3_builder \
  --users user_1 \
  --resume
```

也可以显式指定从某个阶段继续：

```bash
python -m Core-Agent.trajectory.cu_generator.builder.phase3_builder `
  --users user_1 `
  --start-from build-cu
```

## 目录作用

这个目录负责 WANDER 文档里 Phase 3 的离线构建流程，也就是把 Phase 2 产出的 `parameterized units` 聚合成 Canonical Units，并写回 `trajectory/cu_base`。

当前主入口在：

- `phase3_builder.py`

当前产物包括：

- `cu_base.json`
- `transitions.json`
- `mappings.json`
- `config.json`
- `faiss_intent.index`：只有在 `faiss` 和 embedding 都可用时才会生成
- `.phase3_checkpoints/*.json`：阶段 checkpoint，供 `--resume` / `--start-from` 使用

## 当前流程

Phase 3 builder 的执行顺序如下：

1. `units_loader.py`
   从 `segments/**/seg_*.json` 读取 Phase 2 的 `"parameterized units"`，并标准化成内部结构。
2. `clusterer.py`
   先按 `app_name` 做硬分组，再按文本、参数结构、action path 做聚类。
3. `cu_builder.py`
   把 cluster 转成 Canonical Unit，构建 `unit_tree`、`parameter_defs`、`path_stats`。
4. `transition_builder.py`
   根据原始 unit 顺序和 `instance_to_cu` 统计 CU 之间的转移频次。
5. `faiss_builder.py`
   为 CU 的 `intent` 构建 FAISS 索引；如果依赖不可用，会优雅降级。
6. `persist.py`
   把所有 Phase 3 结果写入 `trajectory/cu_base`。

## 输入与输出

输入目录默认是：

- `trajectory/cu_generator/segments`

输出目录默认是：

- `trajectory/cu_base`

输入数据当前依赖 Phase 2 segment 文件中的这些字段：

- `segment_id`
- `app`
- `env`
- `steps`
- `"parameterized units"`

## 常用文件说明

- `schemas.py`：定义 Phase 3 内部使用的数据结构
- `units_loader.py`：负责读取和标准化 Phase 2 产物
- `clusterer.py`：负责聚类
- `cu_builder.py`：负责生成 Canonical Unit
- `transition_builder.py`：负责转移表
- `faiss_builder.py`：负责 FAISS 索引
- `persist.py`：负责写盘
- `phase3_builder.py`：负责统一编排

## 当前已知说明

- 当前默认是离线构建流程，不做在线更新。
- 如果环境里没有 `faiss` 或 `numpy`，Phase 3 仍然可以完成，但不会生成 `faiss_intent.index`。
- 聚类效果和 embedding 可用性强相关；在 embedding 不可用时，会退回本地规则相似度。
