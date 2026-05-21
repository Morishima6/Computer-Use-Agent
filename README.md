# Core-Agent

`Core-Agent` 是一个面向 **OSWorld** 桌面环境的 GUI Agent 项目快照。它把多模态大模型、桌面截图感知、基于描述的控件定位、动作执行、反思控制，以及一套离线构建的 **Canonical Unit（CU）知识库** 组合起来，用于执行真实桌面软件中的多步任务。

当前这个目录里的实现重点有两条主线：

1. `gui_agents/`
   运行时在线代理。负责读取截图、规划下一步动作、把自然语言动作落到具体 UI 控件，并在桌面环境里执行。
2. `trajectory/`
   轨迹与知识库侧。负责存放 Phase 2 产物、Phase 3 的 CU 构建逻辑、检索逻辑，以及最终的 `cu_base` 知识库。

如果你把它放到一句话里理解，可以把这个项目看成：

> 一个运行在 OSWorld/Linux 桌面里的 GUI Agent，并额外接入了一套从历史操作轨迹中提炼出的可复用操作单元库，用来帮助代理做更稳定的下一步规划。

## 1. 项目定位

这个仓库目录不是一个“只有在线推理”的 Agent，也不是一个“只有数据构建”的离线工具，而是两者结合：

- 在线执行层：从当前屏幕出发，生成单步 GUI 动作。
- 离线知识层：从历史 `segment -> units -> parameterized units` 中聚合出 Canonical Units。
- 运行时检索层：在执行时按“当前状态相似度 + 上一步转移关系”双通道检索候选 CU，作为规划提示。

从代码现状看，这个目录里 **真正落地的核心代码** 主要是：

- GUI Agent 运行时
- CU 检索运行时
- Phase 3 Canonical Unit 构建器
- 已生成的 `trajectory/cu_base` 知识库

而 `trajectory/WANDER_Engineering_Guide_v4.md` 中描述的 **Phase 1/Phase 2** 更多体现为方法设计文档与样例产物说明；当前目录里没有完整的 Phase 1/2 采集与标注流水线实现文件。

## 2. 这个项目和 OSWorld 的关系

这个项目默认面向 **OSWorld 风格的 Linux 桌面任务环境**：

- 动作执行依赖 `pyautogui`
- 文档中默认操作系统是 Linux
- Prompt 中直接假设可以使用 `sudo`
- 默认密码写死为 `osworld-public-evaluation`
- 一些动作逻辑直接假设系统中有 `wmctrl`、LibreOffice、浏览器等桌面应用

因此，项目的运行语义不是“网页自动化”，而是“**真实桌面 GUI 自动化**”。

## 3. 总体架构

### 3.1 在线执行链路

```text
用户任务
  -> gui_agents/cli_app.py
  -> AgentS3
  -> Worker
  -> Reflection / UI state extraction / CU retrieval
  -> Planner 生成单步 grounded action
  -> OSWorldACI 把动作描述落到屏幕坐标或具体键鼠操作
  -> pyautogui 执行
```

### 3.2 离线知识构建链路

```text
Phase 2 segment 文件
  -> trajectory/cu_generator/segments/**/*.json
  -> units_loader.py 读取 parameterized units
  -> clusterer.py 聚类
  -> cu_builder.py 构建 Canonical Units / Unit Tree / Path Stats
  -> transition_builder.py 构建 CU 转移表
  -> faiss_builder.py 构建 intent 向量索引
  -> persist.py 写入 trajectory/cu_base
```

### 3.3 运行时 CU 检索链路

```text
当前任务 + 当前屏幕
  -> ui_state_extractor.py 生成状态文本
  -> CUStore 读取 cu_base
  -> CURetriever 双通道检索
     - state similarity
     - transition history
  -> build_cu_retrieval_prompt() 生成规划提示
  -> Worker 在本轮规划时决定是否选用某个 CU/path
```

## 4. 目录结构

```text
Core-Agent/
├── README.md
├── requirements.txt
├── setup.py
├── gui_agents/
│   ├── cli_app.py
│   ├── agents/
│   │   ├── agent_s.py
│   │   ├── worker.py
│   │   ├── grounding.py
│   │   ├── code_agent.py
│   │   └── worker_components/
│   ├── core/
│   ├── memory/
│   └── utils/
└── trajectory/
    ├── WANDER_Engineering_Guide_v4.md
    ├── cu_generator/
    │   ├── segments/
    │   └── builder/
    ├── cu_base/
    └── retrieval/
```

### 目录说明

- `gui_agents/cli_app.py`
  命令行入口，负责初始化模型参数、截图循环、运行 Agent。
- `gui_agents/agents/agent_s.py`
  定义 `AgentS3`，是当前主执行代理。
- `gui_agents/agents/worker.py`
  代理核心调度器。负责编排规划、反思、CU 检索、动作解析和执行。
- `gui_agents/agents/grounding.py`
  `OSWorldACI` 动作接口层。把“点击某个元素”转成坐标定位与 `pyautogui` 代码。
- `gui_agents/agents/code_agent.py`
  可选代码代理，用于用 Python/Bash 处理更适合编程完成的子任务。
- `trajectory/cu_generator/segments/`
  Phase 2 产物数据目录，按用户和 trace 保存 `seg_*.json`。
- `trajectory/cu_generator/builder/`
  Phase 3 构建器，把 parameterized units 聚合成 CU。
- `trajectory/cu_base/`
  当前仓库携带的 Canonical Unit 知识库。
- `trajectory/retrieval/`
  运行时检索代码，包括 embedding、FAISS、转移表检索和 prompt 组装。

## 5. 运行时核心模块

### 5.1 `cli_app.py`

这是最直接的运行入口，主要做四件事：

- 解析模型与 grounding 参数
- 获取当前屏幕大小并缩放截图
- 初始化 `OSWorldACI`
- 初始化 `AgentS3` 并进入交互式任务循环

默认每个任务最多执行 15 步，每一步都会：

1. 截图
2. 调用代理生成下一步
3. 把返回的单步 action 编译成可执行代码
4. `exec()` 执行该代码

### 5.2 `AgentS3`

`AgentS3` 本身很薄，主要是把具体工作委托给 `Worker`。它的价值在于把“一个 UI Agent”的接口稳定下来：

- `reset()`
- `predict(instruction, observation)`

### 5.3 `Worker`

`Worker` 是整个运行时的核心。

它内部维护了：

- 规划模型 `generator_agent`
- 反思模型 `reflection_agent`
- 截图历史
- 当前 CU 执行状态
- 最近一次反思结果
- 最近一次 code agent 结果

一轮 `generate_next_action()` 中，`Worker` 会按这个顺序工作：

1. 把截图交给 `grounding_agent`
2. 如果启用了反思，先评估上一轮动作是否有效
3. 如果启用了 CU 检索，决定是否触发新一轮检索
4. 生成 CU 提示或 Active CU Prompt
5. 调用规划模型输出严格格式化的响应
6. 解析 `(CU Selection)` 段
7. 从 code block 中提取 `agent.xxx(...)`
8. 转成可执行 `pyautogui` 代码并返回

### 5.4 `OSWorldACI`

`OSWorldACI` 是动作落地层，提供了一组受控 API 给大模型调用，例如：

- `agent.click(...)`
- `agent.type(...)`
- `agent.scroll(...)`
- `agent.drag_and_drop(...)`
- `agent.hotkey(...)`
- `agent.switch_applications(...)`
- `agent.open(...)`
- `agent.wait(...)`
- `agent.done()`
- `agent.fail()`

这些 API 最终都会返回一段可执行的 Python/`pyautogui` 代码。

其中最关键的能力是：

- 用 grounding model 根据自然语言描述定位 UI 元素坐标
- 用 OCR 和文本 span 匹配实现文本选区定位
- 在 Linux / macOS / Windows 上生成不同的应用切换与打开逻辑

虽然接口支持多平台，但项目的 prompt 和运行假设明显更偏向 **Linux + OSWorld**。

### 5.5 Reflection 控制

反思模块不是为了生成新计划，而是为了控制当前 CU 的状态机。

当前支持的控制状态包括：

- `continue_current_cu`
- `cu_blocked`
- `cu_completed`
- `cu_failed`
- `task_completed`

这一层的作用是：

- 判断上一动作是否真的推进了任务
- 判断当前 CU 是否还能继续
- 决定何时回退、何时重检索、何时结束一个 CU

### 5.6 Code Agent

当任务更适合通过代码而不是 GUI 操作完成时，运行时可以调用 `agent.call_code_agent()`：

- 让单独的 `CodeAgent` 用 Python/Bash 分步执行
- 支持有限预算的代码执行
- 记录执行历史、输出、错误与总结
- 最终把结果再回传给主 Worker，用 GUI 继续验证

这对表格处理、批量编辑、文件处理类任务特别有帮助。

## 6. CU 检索与知识库

### 6.1 什么是 Canonical Unit

在这个项目里，Canonical Unit 可以理解为：

> 从多条历史轨迹中聚合出来的、可复用的“操作子目标 + 操作路径模板”。

每个 CU 不只是一个名字，它还包含：

- `intent`
- `abstract_state_before`
- `abstract_state_after`
- `unit_tree`
- `parameter_defs`
- `path_stats`
- `source_instance_ids`
- `app_context`

也就是说，它同时携带：

- 这个操作“想做什么”
- 它通常发生在什么界面状态下
- 常见的动作路径是什么
- 可参数化的字段有哪些

### 6.2 `trajectory/cu_base/` 中有哪些文件

- `cu_base.json`
  Canonical Units 主体数据。
- `transitions.json`
  CU 到 CU 的历史转移频次表。
- `mappings.json`
  `instance_to_cu` 和 `faiss_to_cu` 映射。
- `config.json`
  构建时的超参数快照。
- `faiss_intent.index`
  用于状态相似度检索的 FAISS 向量索引。

### 6.3 检索策略

当前检索采用双通道合并：

1. `state_similarity`
   基于任务文本 / 当前界面状态文本做 embedding，相似检索最匹配的 CU。
2. `transition`
   基于上一个 CU 的历史后继关系，猜测下一个可能的 CU。

然后将两类候选合并排序：

- 同时被两条通道命中的优先
- 再按相似度、转移次数、执行次数、成功率排序

### 6.4 Active CU 模式

一旦规划器在某轮选择了某个 `CU + path`，系统就进入 Active CU 模式：

- 后续轮次优先沿这个 path 继续执行
- 仍然只输出一个原子动作
- 若反思判定“局部偏航但可恢复”，会给出 recovery guidance
- 若反思判定 CU 已失效，则回到 `need_retrieve`

## 7. `trajectory/cu_generator/segments/` 数据格式

仓库中的 `segments` 是 Phase 2 的输入产物样例，每个 `seg_*.json` 通常包含：

- `segment_id`
- `app`
- `env`
- `steps`
- `units`
- `parameterized units`

其中：

- `steps`
  保存逐步操作、截图路径、时间戳、前后状态描述、动作元数据。
- `units`
  第一阶段切分出的语义单元。
- `parameterized units`
  第二阶段抽象参数化后的单元，是 Phase 3 的直接输入。

示例里常见字段包括：

- `unit_intent`
- `unit_type`
- `abstract_intent`
- `parameters`
- `parameterized_action_sequence`
- `phase1_status`
- `phase2_status`

## 8. Phase 3 构建器

`trajectory/cu_generator/builder/` 是当前仓库最完整的离线数据流水线实现。

### 8.1 构建流程

1. `units_loader.py`
   读取所有 `parameterized units`，标准化为 `ParameterizedUnitRecord`
2. `clusterer.py`
   按 app 分组后，用 embedding 相似度做聚类
3. `cu_builder.py`
   构建 Canonical Unit、Unit Tree、Path Stats、参数定义
4. `transition_builder.py`
   从原始顺序中统计 CU 之间的转移关系
5. `faiss_builder.py`
   为 CU intent 构建 FAISS 向量索引
6. `persist.py`
   将结果写入 `trajectory/cu_base/`

### 8.2 构建入口

主入口文件是：

- `trajectory/cu_generator/builder/phase3_builder.py`

可直接运行：

```bash
python -m trajectory.cu_generator.builder.phase3_builder --users user_1
```

也可以显式指定输入输出目录：

```bash
python -m trajectory.cu_generator.builder.phase3_builder \
  --segments-root trajectory/cu_generator/segments \
  --output-dir trajectory/cu_base \
  --users user_1 \
  --similarity-threshold 0.82 \
  --embedding-model Qwen/Qwen3-Embedding-8B
```

### 8.3 当前构建依赖

Phase 3 依赖以下能力：

- `numpy`
- `scikit-learn`
- `faiss-cpu`
- embedding API

如果缺少 `faiss` 或 embedding 能力，部分功能会优雅降级，但无法生成完整的向量索引。

## 9. 模型与后端支持

运行时模型封装在 `gui_agents/core/engine.py`，当前支持：

- `openai`
- `anthropic`
- `azure`
- `gemini`
- `open_router`
- `vllm`
- `huggingface`
- `parasail`

另外，项目还额外使用了两个“辅助模型位”：

- grounding model
  用于根据描述在截图中定位坐标
- state text model
  当前默认是 `qwen3-vl-flash`，用于把截图转成简短的状态文本供 CU 检索

embedding 默认模型名是：

- `Qwen/Qwen3-Embedding-8B`

## 10. 安装与运行

### 10.1 安装依赖

建议先进入该目录：

```bash
cd Core-Agent
```

然后安装依赖：

```bash
pip install -r requirements.txt
pip install -e .
```

安装完成后会注册一个命令行入口：

```bash
cua
```

### 10.2 常用环境变量

按你选择的后端设置，不需要全部都配：

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
AZURE_OPENAI_API_KEY=...
QWEN_API_KEY=...
DASHSCOPE_API_KEY=...
SILICONFLOW_API_KEY=...
```

### 10.3 启动在线代理

最直接的方式是运行：

```bash
python -m gui_agents.cli_app \
  --provider openai \
  --model <your-main-model> \
  --model_url <your-main-base-url> \
  --ground_provider openai \
  --ground_url <your-grounding-base-url> \
  --ground_model <your-grounding-model> \
  --grounding_width 1280 \
  --grounding_height 720
```

如果希望启用 CU 检索：

```bash
python -m gui_agents.cli_app \
  --provider openai \
  --model <your-main-model> \
  --model_url <your-main-base-url> \
  --ground_provider openai \
  --ground_url <your-grounding-base-url> \
  --ground_model <your-grounding-model> \
  --grounding_width 1280 \
  --grounding_height 720 \
  --enable_cu_retrieval
```

### 10.4 启用本地代码环境

如果你希望 `CodeAgent` 真正执行本地 Python/Bash，可以加：

```bash
--enable_local_env
```

但这意味着模型可能执行任意代码，只适合可信环境。

## 11. 关键文件导读

- [gui_agents/cli_app.py](./gui_agents/cli_app.py)
  命令行入口与交互式执行循环。
- [gui_agents/agents/worker.py](./gui_agents/agents/worker.py)
  主调度器，最值得优先阅读。
- [gui_agents/agents/grounding.py](./gui_agents/agents/grounding.py)
  动作 API、坐标定位、代码代理调用都在这里。
- [gui_agents/memory/procedural_memory.py](./gui_agents/memory/procedural_memory.py)
  Worker、Reflection、Code Agent 的系统提示词定义。
- [trajectory/retrieval/cu_retrieval/cu_store.py](./trajectory/retrieval/cu_retrieval/cu_store.py)
  读取 `cu_base` 并执行检索。
- [trajectory/retrieval/cu_retrieval/cu_matcher.py](./trajectory/retrieval/cu_retrieval/cu_matcher.py)
  双通道候选合并逻辑。
- [trajectory/cu_generator/builder/phase3_builder.py](./trajectory/cu_generator/builder/phase3_builder.py)
  Phase 3 主入口。
- [trajectory/WANDER_Engineering_Guide_v4.md](./trajectory/WANDER_Engineering_Guide_v4.md)
  设计文档，说明整个 WANDER 方法的完整概念。

## 12. 当前仓库快照的特点

从当前 `trajectory/cu_base/` 的内容可以看出，这个仓库已经携带了一份可直接用于运行时检索的知识库快照，而不是只有代码骨架。

同时也要注意：

- 当前目录更偏“研究原型/工程实验版”，不是彻底产品化的 SDK
- prompt 和环境假设对 OSWorld/Linux 绑定较强
- 运行时中有较多 `print`、日志文件和 `exec()` 风格的实验性实现
- Phase 1/Phase 2 的概念完整，但代码落地主要集中在 Phase 3 和运行时

## 13. 适合怎样继续扩展

如果你准备继续做这个项目，通常有三条路径：

1. 强化运行时
   优化 grounding、反思、动作格式约束、多步成功率。
2. 强化知识库
   增加更多 `segments` 数据、提升 Phase 2 抽象质量、重建更稳定的 CU。
3. 强化评估与环境
   将其更紧密地接到 OSWorld 任务集、自动评测脚本、结果追踪面板。

## 14. 一句话总结

`Core-Agent` 当前是一个面向 OSWorld 的桌面 GUI Agent 工程快照：上层用多模态模型做逐步 GUI 操作，下层用 Canonical Unit 知识库为规划提供历史经验和结构化先验，中间再用反思机制做运行时控制。
