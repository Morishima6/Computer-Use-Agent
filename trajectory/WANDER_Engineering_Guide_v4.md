# WANDER 完整工程实施指南 v4.0
> **WANDER**: Bridging Human **W**orkflow and **A**utomation Computer Usage via **N**aturalistic **D**emonstrations and **E**nhanced **R**etrieval

---

## 目录

1. [总体架构概览](#1-总体架构概览)
2. [Phase 1：采集器部署与原始数据收集](#2-phase-1采集器部署与原始数据收集)
3. [Phase 2：原始数据 → 结构化 Unit 序列](#3-phase-2原始数据--结构化-unit-序列)
4. [Phase 3：Canonical Unit 聚合 + Transition Table](#4-phase-3canonical-unit-聚合--transition-table)
5. [Phase 4：Agent 执行时的双通道检索与自主选择](#5-phase-4agent-执行时的双通道检索与自主选择)
6. [文件组织与目录结构](#6-文件组织与目录结构)
7. [关键超参数汇总与调参建议](#7-关键超参数汇总与调参建议)

---

## 1. 总体架构概览

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              整体 Pipeline                                       │
│                                                                                 │
│  用户日常操作 ──► 采集器 ──► 原始轨迹 ──► Grounding + Annotation + 噪声过滤           │
│                                                 │                               │
│                                                 ▼                               │
│                           规则信号分段 → Unit 切分+标注                            │
│                                                 │                               │
│                                                 ▼                               │
│                                      参数化/抽象化 Unit 序列                       │
│                                                 │                               │
│                                    ┌────────────┴────────────┐                  │
│                                    ▼                         ▼                  │
│                          Canonical Unit 聚合          Transition Table 构建      │
│                          (聚类 → Unit Tree            (原始轨迹 → CU 间           │
│                           + 统计 + 参数)               转移频次邻接表)              │
│                                    │                         │                  │
│                                    └────────────┬────────────┘                  │
│                                                 ▼                               │
│                                  Canonical Unit Base                            │
│                               ┌─────────────────────┐                           │
│                               │  CU 列表 + Unit Tree │  cu_Base.json             │
│                               │  FAISS intent index │  + faiss_intent.index    │
│                               │  Transition Table   │  + transitions.json      │
│                               └─────────┬───────────┘                           │
│                                         │                                       │
│  用户指令 ──► 双通道检索 ──► 候选列表 ──► Agent 自主选择 ──► 执行 ──► 反馈            │
│              (通道1: 状态相似度                                                   │
│               通道2: 转移表下一跳)                                                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 核心设计原则

- **Canonical Unit**：检索、复用、统计全围绕 Canonical Unit 展开。每个 CU 是从多条轨迹中聚合而成的抽象操作单元。
- **Unit 而非 Step**：多个语义不可分割的原子操作由 LLM 联合切分为一个 Unit，作为最小操作单元。
- **抽象先行**：每个 Unit 在 Phase 2 即携带抽象层描述，使不同实例（如"Slide 1 标题"和"Slide 2 标题"）在 Phase 3 聚合时自然归入同一 Canonical Unit。
- **全维度参数化**：不仅 TYPING 文本输入，所有操作维度（点击目标、选择项、UI 元素实例）均进行参数化抽象，使 CU 天然具备泛化能力。
- **线性生产，聚类融合**：Phase 2 负责逐段独立地生产线性 unit（原料），Phase 3 负责把来自不同轨迹的线性原料按语义聚类合并为 Canonical Unit（去重 + 多路径聚合）。
- **拓扑是信号而非约束**：Transition Table 记录历史上 CU 之间的转移频次，但仅作为检索的辅助通道。Agent 看到候选列表后自主判断——有用的拓扑信号被自然采纳，spurious 的拓扑被 Agent 的判断力过滤。
---

## 2. Phase 1：采集器部署与原始数据收集 [已完成✅]

### 2.1 采集器需要记录什么

采集器在后台静默运行，记录以下四类原始信号：

| 信号类型 | 具体内容 | 采集方式 |
|---------|---------|---------|
| 截图 | 每次操作前后各一张全屏截图 | 操作触发截图 |
| 操作事件 | 鼠标点击/滚动/拖拽、键盘输入/快捷键 | 系统级 hook |
| 窗口元信息 | 当前活跃窗口的 app_name、window_title、URL（如果是浏览器） | wmctrl / xdotool / D-Bus |
| 时间戳 | 每个事件的精确时间 | time.time() |

### 2.2 原始数据的存储格式

用户随时产生一个目录：

```
raw_data/
├── user_1/
│   ├── Trace1
│   │   ├── report.jsonl          # 操作事件流（events）
│   │   └── screenshots/
│   │       ├── s1_before.png
│   │       ├── s1_after.png
│   │       └── ...
│   ├── Trace2/
│   │   └── ...
```

### 2.3 噪声过滤-原始数据层面

规则1：连续滚动合并，只记录第一次滚动的before和最后一次滚动的after，将中间的滚动过程合并

规则2: 连续输入合并


### 2.4 原始数据的grouding + Annotation

对于鼠标操作，把（x，y）的坐标，转为 NL_explanation。
对 Step-level 的state描述、precondition和effect描述。


## 3. Phase 2：原始数据 → 结构化 Unit 序列

这一步把连续的原始事件流经过 **基于规则信号的切分 → 噪声过滤 → LLM 两阶段处理**，产出可直接用于 Canonical Unit 聚合的参数化/抽象化 Unit。

> **关键设计**：将 Phase 2 中的语义理解拆成两个紧密衔接的 LLM 子阶段。原因：
> 1. Unit Segmentation 与 concrete annotation 强耦合，本质上都在回答"这一段 step 共同完成了什么子目标"
> 2. Abstraction / parameterization 依赖 unit 边界和 concrete intent 已经稳定，更适合作为后置 enrich 步骤
> 3. 两阶段后输出更短、错误隔离更强，且第二阶段可以按 unit 单独重试

### 3.1 第一步：规则信号分段 [已完成✅]

用规则信号将连续的原始轨迹切为较短的片段（segment），目的是为后续 LLM 处理提供合理长度的输入，避免单次送入过长的事件序列。分段本身不赋予语义含义——它只是物理上的切割，不代表"一个任务"或"一个目标"。

1. 分段信号

对按时间排序的事件列表，逐对扫描相邻事件（prev, curr），用多信号加权打分判断是否为分段边界：

```
对每一对相邻事件 (prev, curr)，累计 boundary_score：

  信号1（最强）：时间间隔
    curr.timestamp - prev.timestamp > θ_T（时间阈值，建议 30-120 秒，需实验标定）
    → score += 0.5

  信号2：应用切换
    prev 操作后的 app ≠ curr 操作前的 app
    → score += 0.25

  信号3：窗口标题大幅变化（同一 app 内切换了完全不同的内容）
    text_similarity(prev 操作后的 window_title, curr 操作前的 window_title) < 0.3
    → score += 0.15

  综合判定：score ≥ θ_boundary → 标记为分段边界（阈值需实验标定）
```

2. 输出

按照 boundary 切分为物理上的独立文件（而非仅保留索引），每个 segment 一个文件，便于下游并行处理、断点续传和错误隔离：
```
raw/user_1/Trace1/report.jsonl          # 原始完整文件（保留不动，作为 ground truth）
         │
         ├── 分段 ──►  segments/user_1/
         │               ├── seg_001.jsonl    # events[0 : boundary_0]
         │               ├── seg_002.jsonl    # events[boundary_0 : boundary_1]
         │               └── ...
         └── 截图文件同样复制一份
```

### 3.2 第二步：噪声过滤-语义层面 [已完成✅]

Naturalistic 数据中有大量无意义操作，需要过滤。对每个 segment 文件内的事件逐条扫描，应用以下三条规则（更多噪声类型参见附录 C）：

```
输入：segment_events（一个 segment 内的事件序列，实际文件为 seg_n.jsonl ）
输出：filtered_events（过滤后的事件序列）

逐条扫描每个 event：

  规则1：无效点击丢弃
    事件为 CLICK，且 CLIP(screenshot_before) 与 CLIP(screenshot_after) 的余弦相似度 > 0.98
    → 界面几乎没变化，大概率是误点击 → 丢弃该事件

  规则2：快速撤销序列抵消
    事件为 Ctrl+Z
    → 说明上一步是误操作 → 同时丢弃上一步和本次 Ctrl+Z，两者互相抵消

  以上均不命中 → 保留该事件
```

### 3.3 第三步：LLM 两阶段处理（unit切分 + 标注 → 抽象参数化）

#### 3.3.1 总体思路

segment_events（来自 seg_n.jsonl 文件） → Pass 1: Unit 切分 + Concrete Annotation → Res：concrete units → Pass 2: Abstraction + Parameterization → Res：parameterized units

这样拆的原因不是单纯为了降低复杂度，而是为了让 unit 边界和语义边界对齐：
- 第一阶段回答的是哪些 step 应被视为一个 unit，以及这个 unit 具体在做什么。这两个问题在语义层面强耦合，应该一起判断。
- 第二阶段回答的是这个已经明确的 unit 应如何抽象，以及哪些值应被参数化。它依赖第一阶段已经给出稳定的 unit 边界和 concrete intent。

超长 segment 的单独方案：极少数超长 segment（如 50+ step，每个 step 两张截图，图片 token 可能逼近上下文上限）做长度检查，第一阶段降级到滑动窗口方案（见附录 B）；第二阶段仍然按 unit 独立执行。

#### 3.3.2 第一阶段：Unit切分 + Concrete Annotation

1. 输入

第一阶段的输入是一个 segment 的连续 steps（包含过滤后的 segment JSON + 截图）

2. 核心思路

这一阶段要求模型在同一次理解过程中同时完成两件事：
- Segmentation：标注哪些连续 steps 应该合并为一个 unit，输出每个 unit 的 `step_indices`。
- Concrete Annotation：对每个 unit 生成具体层结构化描述（界面状态、操作语义、意图等）。

这两件事必须联合完成，因为它们强耦合：决定"s7、s8、s9 应该合并"和理解"这三步共同完成了内容替换"是同一个判断过程。如果先切分再标注，切分时缺乏语义理解，容易犯边界错误；如果先标注再切分，每个 step 独立标注时看不到它和前后 step 的语义关联。

3. 切分与合并unit的判断准则

- 过渡态合并：中间状态如果只是过渡态（如 Ctrl+A 选中后的状态），而不是用户真正关心的稳定状态，应并入同一个 unit。
- 共现操作合并：多个操作如果总是共同出现，拆开后单独执行没有独立意义（如"全选→删除→输入"三步构成一个完整的"替换内容"操作），应并入同一个 unit。
- 独立操作保留：单步即可表达完整语义的操作（如"点击 Proceed to checkout"，切换幻灯片），应保留为独立 unit。
- **intent 是子目标而非动作转写**：`intent` 描述的是用户想达成的深层子目标（如"更新收货地址"），而不是表层动作本身（如"在输入框里打字"）。


4. 输出

对每个 unit，LLM 需要输出以下字段（仅限需要语义理解的部分，机械性字段由后处理补齐）：

| 字段 | 说明 | 示例 |
|------|------|------|
| `step_indices` | 该 unit 包含哪些 step 的序号 | `[7, 8, 9]` |
|------|------|------|
| `unit_before_state` | 操作前界面的具体自然语言描述 | "Firefox 显示 Amazon 地址表单，Ship to 输入框包含旧地址" |
| `unit_precondition` | 到达该状态需要满足的前提条件 | `["已进入地址填写页面", "输入框处于激活状态"]` |
| `unit_after_state` | 操作后界面的具体自然语言描述 | "Firefox 显示 Amazon 地址表单，Ship to 输入框包含新地址 '123 Main St'" |
| `unit_effect` | 操作产生的可观测效果及验证信号 | `["新地址已输入到 Ship to 输入框"]` |
|------|------|------|
| `unit.type` | 对该 unit 操作的语义类型归纳（不是逐 step 的原始事件类型） | `"REPLACE_CONTENT"` |
| `unit_intent` | 该 unit 的用户深层子目标 | `"更新收货地址为新地址"` |


5. 处理流程

```
输入：一个 segment 的全部 steps（含截图）

1. 将全部 steps 按格式组装为 prompt，附上所有截图
2. 调用多模态 LLM，要求同时输出切分方案和每个 unit 的 concrete annotation
3. 解析 LLM 的结构化输出（JSON 格式），得到 concrete_units 列表
4. 对输出进行第一阶段自动校验（见 Section 3.4）
5. 校验通过 → 将 concrete_units 写回 segment JSON 的 units 字段，每个 unit 标记 phase1_status = "done"
6. 校验失败 → 标记 phase1_status = "failed"，记录失败原因，整个 segment 重跑第一阶段
```

#### 3.3.3 第二阶段：Unit-level 抽象参数化

> 理由："给 Slide 1 设红色"和"给 Slide 2 设红色"的 abstract_state 都是"Impress 中目标元素被选中，可进行格式操作"，在 Phase 3 聚合时自然归入同一个 Canonical Unit。不同实例的参数绑定记录在 CU 的 `source_instances` 中。

1. 整体流程
单个 `concrete unit` → 送入第二阶段模型 → 输出抽象状态、参数定义和参数化动作序列 → 回填为最终 `parameterized unit`


这一阶段逐个处理上一阶段输出的 unit
每个 unit 的输入包括：
- 该 unit 包含的原始 steps
- 第一阶段产出的 concrete annotation
- unit 的 before / after screenshot

第二阶段的目标是把"已知含义明确的 concrete unit"转换成"可泛化、可复用"的抽象单元。它主要补充以下字段：

- `abstract_unit_before_state`
- `abstract_unit_after_state`
- `parameters`
- `parameterized_action_sequence`

2. 关键

关键是识别：
- 哪些信息是实例特定值，应该提取成参数
- 哪些信息属于稳定语义，应保留在抽象状态里
- 参数名应该如何命名，才能支持后续跨实例复用

例如：
- 具体的"点击 Slide 1 标题"会被泛化成"点击 `{{target_element}}`"，并绑定 `target_element = "Slide 1 标题"`
- 具体的"输入 `#FF0000`"会被泛化成"输入 `{{color_value}}`"，并绑定 `color_value = "#FF0000"`
- 具体的"Firefox 显示 Amazon 购物车页面"会被泛化成"浏览器显示电商购物车页面"




#### 3.3.5 Unit示例

以 Chrome 改名轨迹为例，两阶段处理后的输出：

| LLM 切分结果 | 包含的原子 Steps | 抽象描述 |
|-------------|-----------------|---------|
| u1: 点击 profile 按钮 (single) | s1 | 具体: "点击右上角 profile 头像" / 抽象: "打开用户配置入口" |
| u2: 点击 Manage profiles (single) | s2 | 具体: "点击 Manage profiles 链接" / 抽象: "进入配置管理页面" |
| u3: 点击三点菜单 (single) | s3 | 具体: "点击 profile 卡片的三点菜单" / 抽象: "展开 {{target_element}} 的操作菜单" |
| u4: 点击 Edit (single) | s4 | 具体: "点击 Edit 选项" / 抽象: "进入编辑模式" |
| u5: 点击 name 输入框 (single) | s5 | 具体: "点击 profile name 输入框" / 抽象: "聚焦到 {{target_field}} 输入框" |
| u6: 替换内容 (merged) | s7+s8+s9+s10 | 具体: "全选并替换为 Thomas,保存" / 抽象: "清空并输入 {{profile_name}}后保存", param: profile_name="Thomas" |


### 3.4 质量控制

**第一阶段自动校验**：
- 每个 unit 的 `step_indices` 必须连续、互不交叉，并且完整覆盖该 segment 的所有 step
- `screen_description` 必须与 `app_context.app_name` 基本一致，否则说明描述可能跑偏
- `intent` 不能退化成动作转写，例如只写"点击按钮""输入文字"；如果出现这类表述，应重新生成

**第二阶段自动校验**：
- `bound_params` 中的每个具体值都必须能在原始 `action_sequence` 或 concrete annotation 中找到来源
- `parameterized_action_sequence` 中的占位符必须能一一映射到 `parameters`
- `abstract_state_before` / `abstract_state_after` 不应继续包含实例特定值（如具体 slide 编号、具体文件名等）；如果仍包含，标记为需要人工审查

**人工抽检**：
- 随机抽取 5% 的第一阶段结果，重点看 unit 边界和 intent 是否合理
- 再随机抽取 5% 的第二阶段结果，重点看抽象是否过度或不足
- 记录错误类型和错误率，作为论文中数据质量的 evidence

[!] 重要：所有以上的描述只是一个方案，具体哪些应该是LLM的输入输出，哪些应该使用规则解析，自行决定！

---



## 4. Phase 3：Canonical Unit 聚合 + Transition Table

### 4.1 总流程

```
Phase 2 产出的所有 parameterized units（来自所有轨迹的所有 segment）
        │
        ▼
Step 1: 按 (app_context, unit.type, intent_embedding) 聚类
        │
        ▼
Step 2: 每个 cluster → 生成一个 Canonical Unit（含 Unit Tree + 统计 + 参数）
        │
        ▼
Step 3: 建立 unit_instance → canonical_unit 的映射表
        │
        ▼
Step 4: 回溯原始轨迹，统计 Canonical Unit 之间的转移频次 → Transition Table
        │
        ▼
Step 5: 构建 FAISS 索引 + 持久化
```

### 4.2 Canonical Unit 数据结构

```python
class CanonicalUnit:
    cu_id: str                   
    
    # --- 语义标识 ---
    intent: str                         # 聚合后的代表性 intent 描述
    intent_embedding_id: int            # intent 的 embedding ID，用于 FAISS 检索
    unit_type: str                      # 共享的 unit.type，如 "REPLACE_CONTENT"
    abstract_state_before: str          # 代表性的抽象前状态描述
    abstract_state_after: str           # 代表性的抽象后状态描述
    
    # --- Unit Tree ---
    unit_tree: dict                     # 树状结构，组织同一目标的不同操作路径
        unit_tree 的例子：
        {
        "root": "a1",
        "nodes": {
            "a1": {
            "type": "CLICK",
            "description": "选中目标文字",
            "params": {"target": "{{target_element}}"},
            "children": ["a2"],
            "source_count": 23
            },
            "a2": {
            "type": "CLICK",
            "description": "打开调色盘",
            "params": {},
            "children": ["a3", "a4"],
            "source_count": 23
            },
            "a3": {
            "type": "CLICK",
            "description": "直接点击预设颜色",
            "params": {"color": "{{color_preset}}"},
            "children": [],
            "source_count": 15,
            "can_terminate": true
            },
            "a4": {
            "type": "CLICK",
            "description": "点击自定义颜色按钮",
            "params": {},
            "children": ["a5"],
            "source_count": 8
            },
            "a5": {
            "type": "TYPING",
            "description": "输入颜色编码",
            "params": {"color_value": "{{color_value}}"},
            "children": ["a6"],
            "source_count": 8
            },
            "a6": {
            "type": "CLICK",
            "description": "点击确认",
            "params": {},
            "children": [],
            "source_count": 8,
            "can_terminate": true
            }
        }
        }
    
    # --- 参数定义 ---
    parameter_defs: list[dict]          # 从所有实例汇总的参数定义
    # [{"param_name": "color_value", "param_type": "string", 
    #   "description": "颜色编码", "observed_values": ["#FF0000", "#0000FF", ...]}]
    
    # --- 统计信息 ---
    execution_count: int                # 所有实例的总执行次数
    success_count: int                  # 成功次数
    success_rate: float                 # success_count / execution_count
    path_stats: dict                    # 每条路径的独立统计
    # {
    #   "preset_color":  {"count": 15, "success": 14, "success_rate": 0.93},
    #   "custom_color":  {"count": 8,  "success": 7,  "success_rate": 0.88}
    # }
    
    # --- 来源信息 ---
    source_users: list[str]             # 来自哪些用户
    source_instance_ids: list[str]      # 聚合了哪些原始 unit 实例
    app_context: dict                   # {"app_name": "Chrome", "url_pattern": "...", ...}
    
    # --- 元信息 ---
    category: str                       # "Daily" / "Office" / ...
    first_seen: str
    last_seen: str
```

### 4.3 聚类合并算法

#### 4.3.1 整体思路

将所有 parameterized unit 按语义相似度聚类。同一目标的不同操作路径（如菜单栏 vs 右键菜单）会被聚合到同一个 Canonical Unit 中，不同路径在 Unit Tree 中形成分支。

**伪代码流程：**

```
build_canonical_units(使用所有 parameterized units, 也叫 abstract units):

  维护数据结构：
    canonical_units = {}          # cu_id → CanonicalUnit
    instance_to_cu = {}           # unit_instance_id → cu_id（映射表，供后续 Transition Table 使用）
    cu_counter = 0                # CU 编号计数器

  ═══ Step 1: 硬过滤分组 ═══
  按 app_context.app_name 分组 → app_groups
  （可选）在每个 app_group 内，再按 unit.type 分组

  ═══ Step 2: 组内语义聚类 ═══
  FOR 每个分组:
    对组内所有 unit 的 intent 做 text embedding
    使用层次聚类算法（agglomerative clustering）：
      距离度量 = 1 - cosine_similarity
      linkage = average
      距离阈值 = 1 - τ_cluster（即 cosine 相似度 ≥ τ_cluster 的归为同一 cluster）
    按聚类结果将 units 分组

  ═══ Step 3: 每个 cluster → Canonical Unit ═══
  FOR 每个 cluster:
    分配 cu_id = "cu_{cu_counter:05d}"

    ── 选取代表性 intent ──
    取 cluster 中 intent 文本最长（最详细）的那个作为代表

    ── 选取代表性 abstract states ──
    abstract_state_before: 取 cluster 中该字段文本最长的
    abstract_state_after:  取 cluster 中该字段文本最长的

    ── 合并操作序列 ──
    合并所有实例的 action_sequence → Unit Tree（见 4.3.2）

    ── 合并参数定义 ──
    合并所有实例的 parameters → parameter_defs（见 4.3.3）

    ── 聚合统计信息 ──
    execution_count = cluster 中实例数量
    success_count = cluster 中 outcome == "success" 的数量（若无 outcome 字段则默认成功）
    success_rate = success_count / execution_count
    path_stats: 在 Unit Tree 构建过程中填充（每条路径的独立统计）

    ── 构建实例映射 ──                  
    FOR cluster 中的每个 unit:
      instance_to_cu[unit.unit_id] = cu_id
      收集 unit_id 到 source_instance_ids 列表

    ── 收集来源与元信息 ──
    source_users = cluster 中所有 source_user 去重
    app_context = 取 cluster 中第一个实例的 app_context
    unit_type = 取 cluster 中第一个实例的 unit_type（缺省 "UNKNOWN"）
    first_seen = cluster 中所有实例 timestamp 的最小值
    last_seen = cluster 中所有实例 timestamp 的最大值

    → 组装以上字段，产出一个 CanonicalUnit
    cu_counter += 1

  ═══ Step 4: 构建 FAISS 索引 ═══
  为所有 Canonical Unit 的 intent 构建 FAISS 索引（详见 4.5）
  用于 Agent 运行时的语义检索
```

#### 4.3.2 Unit Tree 构建（将多条线性路径合并为树）

```
  初始化 tree = 空树

  FOR cluster 中的每个 unit:
    取出其 parameterized_action_sequence 作为一条线性路径 new_path
    
    若 tree 为空:
      将 new_path 直接转化为一条单链树（每个 action 是一个节点，依次串联）
      标记末尾节点 can_terminate = true
      继续下一个 unit

    从 tree 的 root 开始，逐步沿树向下与 new_path 的每一步比较：
    
      当前树节点 current_node, 当前匹配深度 matched_depth = 0
      
      对 new_path 中的每一步 new_step：

        ── 语义级别比较 ──
        若 current_node.type == new_step.type
           且 semantic_sim(current_node.description, new_step.description) ≥ τ_action_match：
        
          → 匹配成功
            current_node.source_count += 1
            matched_depth += 1
            
            查看 current_node 的 children：
              ├─ 1 个 child → 继续沿唯一子节点向下
              ├─ 多个 children（已有分叉点）→ 在 children 中找与下一步最匹配的
              │   ├─ 找到 → 沿该 child 继续
              │   └─ 未找到 → 在此处挂载 new_path 剩余步骤为新分支 → 结束
              └─ 无 children（叶子）
                  ├─ new_path 还有后续 → 从此叶子长出新分支 → 结束
                  └─ new_path 也结束 → 纯统计更新 → 结束

        否则（不匹配）：
          → 在上一个匹配节点处创建分叉
            原子树保留为一个分支
            new_path 从当前位置起的剩余步骤作为另一个分支
            → 结束

  若 new_path 所有步骤都匹配完（是已有路径的前缀）：
    标记最后匹配到的节点 can_terminate = true
```

> **边界情况**：两条路径完全不共享前缀（如菜单栏 vs 右键菜单）。此时 tree 的 root 变成一个虚拟分叉点，两条路径从一开始就分开。这不是 bug——Unit Tree 天然兼容零前缀共享的情况，它退化为一个 variant list。


#### 4.3.3 参数定义合并

```
  all_params = {}
  FOR 每个 unit:
    FOR unit.parameters 中的每个参数:
      ├─ 若 all_params 中已有同名参数
      │   → 将该实例的 concrete_value 追加到 observed_values
      └─ 若为全新参数名
          → 新建参数定义 {param_name, param_type, description, observed_values: [该值]}
  返回 all_params 的值列表
```

### 4.4 Transition Table 构建

> **设计动机**：Transition Table 的目的是捕捉"用户做完操作 A 后通常接着做什么"这个信号。它不是因果依赖的硬编码——用户先改颜色后改字体只是偶然顺序，但用户"打开 profile 后接着点 manage"则是强因果序列。两者都会被记录在 Transition Table 中，区分交给 Agent 在运行时判断。

**输入**：所有原始轨迹（每条轨迹是按时间排列的 unit 实例序列）+ 上一步构建好的 instance_to_cu 映射表。

**前置条件**：必须在 CU Base 构建完成之后执行，因为依赖 instance_to_cu 映射表将原始 unit 实例映射到 Canonical Unit。

**核心思路**：遍历每条原始轨迹，将其中的 unit 实例序列"翻译"为 CU 序列，然后统计所有相邻 CU 对的出现频次，最终形成一张邻接频次表。

```
build_transition_table:

  ═══ 第一步：逐条轨迹翻译为 CU 序列 ═══

  对每条原始轨迹（按时间排列的 unit 实例序列）：

    原始轨迹:  [unit_38, unit_12, unit_55, unit_07, unit_22, ...]
                  │         │         │        │        │
                  ▼         ▼         ▼        ▼        ▼
    查映射表:  instance_to_cu[unit_id] → cu_id
                  │         │         │        │        │
                  ▼         ▼         ▼        ▼        ▼
    CU 序列:   [CU_3,    CU_7,    CU_1,    CU_7,   CU_12,  ...]

    注意：若某个 unit_id 在映射表中找不到（可能是被过滤掉的噪声），则跳过该实例。

  ═══ 第二步：滑动窗口统计相邻 CU 对 ═══

  对每条 CU 序列，用大小为 2 的滑动窗口扫描所有相邻对：

    CU 序列:  CU_3 → CU_7 → CU_1 → CU_7 → CU_12
              ├────┤
              │ (CU_3, CU_7) → transitions[CU_3][CU_7] += 1
              └────┤
                   ├────┤
                   │ (CU_7, CU_1) → transitions[CU_7][CU_1] += 1
                   └────┤
                        ├────┤
                        │ (CU_1, CU_7) → transitions[CU_1][CU_7] += 1
                        └────┤
                             ├─────┤
                             │ (CU_7, CU_12) → transitions[CU_7][CU_12] += 1
                             └─────┘

    特殊处理：若 cu_prev == cu_next（自环），跳过不计——连续执行同一个 CU 不构成有意义的转移信号。

  ═══ 输出 ═══

  transitions 是一张邻接频次表，结构为：
    {源 CU: {目标 CU: 转移次数, ...}, ...}

  示例：
    transitions[CU_3][CU_7] = 12   表示历史上有 12 次"做完 CU_3 后紧接着做 CU_7"
    transitions[CU_3][CU_1] = 3    表示历史上有 3 次"做完 CU_3 后紧接着做 CU_1"
```

### 4.5 FAISS 索引构建

为所有 Canonical Unit 的 intent 构建向量索引，用于 Agent 运行时的语义检索。

```
遍历所有 CU，逐个生成 intent 的 text embedding
      │
      ▼
将所有 embedding 汇总为矩阵，写入 FAISS 内积索引（IndexFlatIP）
      │
      ▼
��时维护一张 id_mapping 表：FAISS 内部序号 → cu_id
（检索时 FAISS 返回的是内部序号，需要通过此表映射回 cu_id）
```

### 4.6 持久化

将构建完成的 CU Base 写入磁盘，共四个文件：

```
① cu_Base.json
   ├─ metadata：CU ���数、覆盖的原始实例数、构建时间戳、构建配置
   └─ canonical_units：所有 CU 的完整数据（intent、Unit Tree、统计、参数等）

② transitions.json
   └─ CU 间转移频次邻接表（即 Section 4.4 构建的 Transition Table）

③ faiss_intent.index
   └─ FAISS 向量索引的二进制文件（即 Section 4.5 构建的索引）

④ mappings.json
   ├─ faiss_to_cu：FAISS 内部序号 → cu_id
   └─ instance_to_cu：unit_instance_id → cu_id（供 Transition Table 更新使用）
```

### 4.7 存储架构

```
cu_Base/
├── cu_Base.json         # Canonical Unit 列表（含 Unit Tree、统计、参数）
├── transitions.json        # CU 间转移频次邻接表
├── faiss_intent.index      # intent embedding 的 FAISS 索引
├── mappings.json           # faiss_idx → cu_id + unit_instance_id → cu_id
└── config.json             # 构建时的所有超参数
```

---

## 5. Phase 4：Agent 执行时的双通道检索与自主选择

### 5.1 总流程

```
Agent 收到用户指令
      │
      ▼
初始化阶段（一次性）
      │  加载 CU Base + Transition Table + FAISS 索引
      │
      ▼
每一步的执行循环：
      │
      ├─ ① 获取当前截图
      │
      ├─ ② 双通道检索候选 CU
      │     通道 1（状态相似度）：当前截图 → 生成描述 → 与 CU 的 before_state 匹配 → top-K
      │     通道 2（下一跳）：查 transitions[上一步 CU] → 历史转移的 CU 列表（带频次）
      │     合并去重 → 候选列表
      │
      ├─ ③ 构建 prompt，将候选列表呈现给 Agent
      │     两个通道的结果分开展示，Agent 自主选择
      │
      ├─ ④ Agent 选择并执行
      │     ├─ 选择了某个 CU → 查看其 Unit Tree，选择路径，填充参数，执行
      │     └─ 候选均不适用 → Agent 完全自主决策（free exploration）
      │
      ├─ ⑤ 记录执行日志
      │
      └─ ⑥ 检查任务是否完成 → 完成则 break
```

### 5.2 双通道检索

每一步执行前，通过两个独立通道并行检索候选 CU，然后合并去重。两个通道回答的问题不同：通道 1 问"当前状态下有哪些已知操作可做"，通道 2 问"上一步之后历史���通常接什么"。

```
═══ 通道 1：状态 + 意图 语义相似度检索 ═══

当前截图 ──► VLM 生成当前界面的抽象描述
                │
                ▼
将「用户指令」与「当前界面描述」拼接为查询文本
                │
                ▼
查询文本 ──► text embedding ──► 在 FAISS 索引中检索 top-K 最相似的 CU
                                      │
                                      ▼
                              每个命中的 CU 记录：
                                similarity_score + 来源标记 = "state_similarity"


═══ 通道 2：Transition Table 下一跳 ═══

前提：上一步 Agent 选择并执行了某个 CU（记为 last_cu_id）
      若上一步未选择任何 CU，则此通道不激活

last_cu_id ──► 查 transitions[last_cu_id]
                    │
                    ▼
              取出所有历史上紧接 last_cu_id 之后执行过的 CU 及其频次
              按频次降序排列，取 top-K
                    │
                    ▼
              每个命中的 CU 记录：
                transition_count + 来源标记 = "transition"


═══ 合并去重 ═══

将两个通道的结果按 cu_id 合并：
  ├─ 仅通道 1 命中 → 来源 = "state_similarity"
  ├─ 仅通道 2 命中 → 来源 = "transition"
  └─ 两个通道都命中同一个 CU → 来源 = "both"（双通道命中，可信度更高）

→ 输出：候选列表（每个候选附带 CU 信息、分数/频次、来源标记）
```

### 5.3 候选列表的 Prompt 构建

**关键设计**：两个通道的结果**分开呈现**给 Agent。通道 1 回答的是"当前界面能做什么"，通道 2 回答的是"别人做完你刚做的事之后通常接着做什么"。这两个问题性质不同，分开呈现比混在一起更有利于 Agent 判断。

Prompt 由三个部分拼接而成：

```
┌─────────────────────────────────────────────────────────────────┐
│ Part 1：基础上下文                                                │
│   "你是一个 GUI 自动化 Agent。                                    │
│    用户的指令是 X。当前屏幕截图已提供。"                            │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Part 2：候选操作列表（按来源分两个板块呈现）                         │
│                                                                 │
│  板块 A —— "当前状态匹配到的操作"                                  │
│  （来源 = state_similarity 或 both 的候选，按匹配度降序排列）        │
│  每个候选展示：                                                   │
│    · intent 描述 + 匹配度分数                                     │
│    · 历史成功率 + 执行次数                                        │
│    · 若有多路径 → 列出各路径及各自统计                              │
│    · 若有参数 → 列出参数名 + 历史值示例                            │
│                                                                 │
│  板块 B —— "你上一步操作之后通常会接的操作"                         │
│  （来源 = transition 或 both 的候选，按历史频次降序排列）             │
│  每个候选展示：                                                   │
│    · intent 描述 + 历史转移频次 + 历史成功率                       │
│                                                                 │
│  注意：若某个通道无结果，对应板块整个省略                            │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Part 3：决策指引                                                  │
│   "请从以上候选中选择最合适的操作并执行。                           │
│    如果候选均不适用，请自主决定下一步。                              │
│    如果选择的候选有多条路径，请指明你选择的路径。"                    │
└─────────────────────────────────────────────────────────────────┘
```

**Prompt 示例**：

```
你是一个 GUI 自动化 Agent。用户的指令是"把标题文字改成红色"。当前屏幕截图已提供。

## 当前状态匹配到的操作：
  1. "修改字体颜色"（匹配度 0.91，历史成功率 93%，共执行 23 次）
     有两种路径：
       路径 A「预设颜色」（15 次，成功率 93%）: 选中文字 → 打开调色盘 → 直接点击预设颜色
       路径 B「自定义颜色」（8 次，成功率 88%）: 选中文字 → 打开调色盘 → 自定义 → 输入色值 → 确认
     需要填充的参数：
       - color_preset (enum): 预设颜色选项，历史值: ["红色", "蓝色", "黑色"]
       - color_value (string): 自定义颜色编码，历史值: ["#FF0000", "#0000FF"]
  2. "修改字号"（匹配度 0.72，历史成功率 97%，共执行 15 次）

## 你上一步操作之后通常会接的操作：
  3. "保存文档"（历史频次 12 次，成功率 100%）
  4. "修改字体样式"（历史频次 5 次，成功率 95%）

请根据用户指令和当前界面，从以上候选中选择最合适的操作并执行。
如果以上候选均不适用，请根据当前界面自主决定下一步操作。
如果你选择了某个候选操作且它有多条路径，请指明你选择的路径。
```

---

## 7. 关键超参数汇总与调参建议

| 超参数 | 所在阶段 | 含义 | 建议初始值 | 调参方向 |
|--------|---------|------|-----------|---------|
| `T_pause` | 规则信号分段 | 时间停顿阈值（秒） | 60 | 30-120，用人工标注的 F1 标定 |
| `boundary_score_threshold` | 规则信号分段 | 多信号融合的综合分段阈值 | 0.4 | 0.3-0.6 |
| `cu_cluster` | CU 聚类 | intent 语义相似度聚类阈值（决定哪些 unit 实例归入同一 CU） | 0.82 | 0.75-0.90（高→CU 多/粒度细，低→CU 少/粒度粗） |
| `cu_action_match` | Unit Tree 合并 | 逐步匹配时的语义相似度阈值（决定两步操作是否视为同一步） | 0.85 | 0.75-0.95（高→分支多/树宽，低→合并激进/树窄） |
| `top_k_retrieval` | 双通道检索 | 每个通道返回的候选数量 | 5 | 3-10 |



---

## 附录 A：在线更新（可选）

> 在线更新是可选的加分项，不是核心方法。如果实验中不做在线更新，在 paper 中作为 future work 提即可。

### A.1 Agent 执行结果反馈到 CU Base

```
update_cu_Base(navigator, 执行记录列表, 任务是否成功):

  FOR 每条执行记录:
    若 selected_cu_id 存在（Agent 采纳了某个 CU 的建议）:
      找到对应的 CU
      CU.execution_count += 1
      ├─ 任务成功 → CU.success_count += 1
      └─ 任务失败 → （不增加 success_count）
      重算 CU.success_rate
      
      若记录了 selected_path:
        在 path_stats 中更新对应路径的统计
```

### A.2 注意事项

- 如果要做在线更新，需要注意避免"agent 的错误执行污染 CU Base"的问题。一种方式是只有在 task_success=True 时才更新。
- 在线更新的实验需要多轮执行：第 1 轮跑完 → 更新 CU Base → 第 2 轮用更新后的库 → 再更新 → ...，观察跨轮次的 success rate 变化。

---

## 附录 B：滑动窗口方案（超长 Segment 降级方案）

> **说明**：默认方案为第一阶段尽量整个 segment 一次性处理（见 3.3.1），第二阶段始终按 unit 独立执行。当 segment 过长（如 50+ step，图片 token 逼近上下文上限）时，仅第一阶段降级到本方案。

### B.1 滑动窗口设计

以 segment 边界为硬边界，在 segment 内的 steps 上使用滑动窗口送入 VLM：

```
输入：segment_events（经过噪声过滤后的事件序列）
参数：window_size = 每次送入 LLM 的 step 数量（建议 5-8）
      overlap = 相邻窗口的重叠 step 数（建议 1-2，用于处理边界 case）
输出：先得到第一阶段的 concrete Unit 序列，再逐个进入第二阶段得到参数化 Unit 序列

处理流程：

  segment_events: [ s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, ... ]
                |←── window_1 ──→|
                            |←── window_2 ──→|        （overlap=2，窗口间共享 2 个 step）
                                        |←── window_3 ──→|

  对每个窗口：
    取 window_size 个 step ──► 送入第一阶段 LLM ──► 得到该窗口内的 concrete units

    若非第一个窗口 → 解决重叠区冲突：
      重叠区内的 step 可能被前后两个窗口各自划入不同 unit
      → 比较两个 unit 的覆盖范围，保留覆盖更广的那个

    将本窗口的 units 追加到总结果

  窗口滑动：每次前进 window_size - overlap 个 step，直到处理完所有 step

  所有窗口处理完毕后：
    对合并后的 concrete units 逐个执行第二阶段抽象参数化
```

### B.2 窗口重叠处理

> **伪代码流程：**
>
> ```
> _resolve_overlap(已有 units, 新窗口 units, overlap):
>
>   取已有 units 的最后一个 unit（last_existing）
>   last_existing_end = 其覆盖的最大 step 编号
>
>   FOR 新窗口中的每个 unit:
>     unit 起始 step ≤ last_existing_end?（即有重叠）
>     ├─ YES → 比较两个 unit 的覆盖范围（step 数量）
>     │   ├─ 新 unit 更广 → 删除旧 unit，保留新 unit
>     │   └─ 旧 unit 更广 → 跳过新 unit
>     └─ NO  → 无重叠，直接保留新 unit
> ```

```python
def _resolve_overlap(existing_units, new_units, overlap):
    """
    处理相邻窗口重叠区域的 unit 归属问题。
    
    策略：如果新窗口的第一个 unit 的 step_indices 与上一轮最后一个 unit 有交集，
    以覆盖范围更大（包含更多 step）的那个为准。
    """
    if not existing_units or not new_units:
        return new_units
    
    last_existing = existing_units[-1]
    last_existing_end = max(last_existing["step_indices"])
    
    resolved = []
    for unit in new_units:
        unit_start = min(unit["step_indices"])
        if unit_start <= last_existing_end:
            # 有重叠：比较覆盖范围
            if len(unit["step_indices"]) > len(last_existing["step_indices"]):
                # 新 unit 覆盖更广，替换旧的
                existing_units.pop()
                resolved.append(unit)
            # 否则跳过新 unit，保留旧的
        else:
            resolved.append(unit)
    
    return resolved
```

### B.3 超参数

| 超参数 | 含义 | 建议初始值 | 调参方向 |
|--------|------|-----------|---------|
| `window_size` | 每次送入 LLM 的 step 数量 | 8 | 5-10（大→上下文更充分，小→LLM 处理更快） |
| `overlap` | 相邻窗口的重叠 step 数 | 2 | 1-3 |

**注意**：窗口太小可能导致跨窗口的 unit 被错误切断，太大可能超出 LLM 上下文处理能力。

---

## 附录 C：噪声类型补充

正文 Section 3.2 列出了两条核心过滤规则（无效点击 + 快速撤销）。以下是更完整的噪声分类，供逐步实现时参考。**优先实现第 1 类和第 3 条**，前者直接影响分段准确性，后者对 LLM intent 推断的干净程度影响最大。其余可在基础 pipeline 跑通后逐步验证是否值得加。

### 第一类：打断真实操作链路的噪声（高干扰）

1. **通知/弹窗打断**
用户正在写文档，系统弹出 Slack 通知或 Windows Update 提示，用户点了"关闭"或"稍后"后回到原操作。这个交互与用户真正的操作完全无关，但会在事件流里产生 app 切换信号——不仅是噪声本身，还可能导致 3.1 的分段逻辑误判为分段边界。识别方式：事件涉及的窗口存活时间极短（出现到消失 < 3-5 秒），且用户操作仅为关闭/最小化类动作。

2. **被动焦点切换（Window focus without action）**
用户点击了某个窗口使其获得焦点，但没有做任何有意义的操作就切走了。多显示器场景下尤其常见——误触了另一个屏幕的窗口。识别方式：对某个窗口，focus 事件和下一次 focus-out 之间没有任何 CLICK/TYPE/快捷键事件。

### 第二类：掩盖真实意图的噪声（中干扰）

3. **退格修正序列（Backspace correction）**
正文规则 2 覆盖了 Ctrl+Z，但更常见的编辑修正是连续退格。用户输入 "recieve"，按 4 次 Backspace 再重新输入 "ceive"。如果逐个保留这些按键事件，LLM 看到的是一团混乱的字符流。处理方式：将连续的"输入→退格→再输入"序列折叠为最终结果文本，只保留一个等效的 TYPE 事件。

4. **重复点击（Rage click / 等待性重复点击）**
与正文规则 1 不同——规则 1 是"点了但界面没变"，这里是用户因页面加载慢而连续快速点击同一位置（界面可能有微小变化，如 loading 动画）。识别方式：N 个 CLICK 事件在极短时间内（< 1 秒）发生在相近坐标（像素距离 < 10px）。只保留第一个。

5. **选择性点击 vs 操作性点击**
双击选词、三击选段落是文本选择手势。如果事件流把它们记录为两三个独立 CLICK 事件，LLM 会以为用户点了三次。识别方式：同一坐标在 < 500ms 内的连续点击，合并为一个 DOUBLE_CLICK 或 TRIPLE_CLICK 事件。

### 第三类：膨胀事件流但不影响语义的噪声（低干扰）

6. **窗口管理操作**
调整窗口大小、拖动窗口位置、最大化/最小化——这些是用户整理桌面的"元操作"，与操作本身的 intent 无关。如果不过滤，会在事件流中占据位置，浪费 LLM 的 context window。识别方式：操作目标为 title bar、窗口边框拖拽、或系统级快捷键（Win+Up/Down 等）。

7. **探索性悬停/预览（Hover preview）**
用户在文件管理器里把鼠标移过多个文件（出现预览 tooltip），或在浏览器里悬停多个链接查看 URL 预览，但最终只点了一个。如果录制了 hover 事件，会极度膨胀事件流。处理方式：只保留紧接着有 CLICK 的那个 hover。

8. **同一位置的 Ctrl+C 连续触发**
用户选中文本后按了两三次 Ctrl+C（怕没复制上），实际效果完全相同。识别方式：连续 Ctrl+C 事件之间无其他操作且选中区域未变。只保留第一个。

### 不该过滤的"噪声"

用户在搜索框里试了多个关键词（输入→看结果→清空→换词），表面上看很像"噪声"，但这种试探行为本身是 intent 的重要信号——它说明用户在做信息检索且不确定用什么词。如果过滤掉中间的尝试，LLM 推断出来的 unit 就会丢失"用户可能需要多次尝试"这个关键步骤信息。类似地，用户在 UI 上来回对比两个选项（频繁在 Tab A 和 Tab B 间切换）也不应该被当作"无意义切换"过滤掉——它代表的是决策行为。

---

## 附录 D：全流程物理存储总结

本附录汇总整个 pipeline 中每一步的物理产物，明确"什么数据存在哪个文件里"。

### D.1 存储流程总览

```
report.jsonl (原始轨迹)
  → seg_xxx.jsonl (规则分段，物理独立文件)
    → 噪声过滤 (原地过滤，仍在 seg_xxx.jsonl 中)
      → units 字段写入 seg_xxx.jsonl (第一阶段: concrete units)
        → units 字段补充 (第二阶段: parameterized units，仍在 seg_xxx.jsonl 中)
          → 所有 units 汇总聚类 → cu_Base/ 目录 (4 个核心文件)
            → Agent 运行时加载 cu_Base/ 做双通道检索
```

### D.2 逐步详细说明

#### Step 1：原始采集

```
raw_data/
├── user_1/
│   ├── Trace1/
│   │   ├── report.jsonl          # 一条完整轨迹的所有操作事件流（每行一个事件）
│   │   └── screenshots/
│   │       ├── s1_before.png     # 每个操作前后各一张全屏截图
│   │       ├── s1_after.png
│   │       └── ...
│   ├── Trace2/
│   │   └── ...
```

- `report.jsonl`：每行一个事件（点击/输入/滚动等），含时间戳、坐标、窗口元信息（app_name、window_title、URL）
- 截图：每个操作前后各一张，文件名与事件序号对应

#### Step 2：规则信号分段

按时间间隔 + 应用切换 + 窗口标题变化等规则信号，把一条长轨迹**物理切成多个短片段文件**：

```
segments/
├── user_1/
│   ├── seg_001.jsonl    # Trace1 的 events[0 : boundary_0]
│   ├── seg_002.jsonl    # Trace1 的 events[boundary_0 : boundary_1]
│   ├── seg_003.jsonl
│   └── ...              # 对应的截图也复制到相应位置
```

- 每个 `seg_xxx.jsonl` 是原始 `report.jsonl` 的一个连续子序列
- **原始 `report.jsonl` 保留不动**，作为 ground truth

#### Step 3：噪声过滤

对每个 `seg_xxx.jsonl` 内的事件逐条过滤（无效点击丢弃、Ctrl+Z 撤销抵消等）。物理上仍在同一个 segment 文件中操作，被过滤的事件被移除或标记。

#### Step 4：LLM 第一阶段 — Unit 切分 + Concrete Annotation

对每个 segment，LLM 输出的 concrete units **写回 `seg_xxx.jsonl` 的 `units` 字段**：

```jsonc
// seg_001.jsonl 处理后（概念示意）
{
  "segment_id": "seg_001",
  "steps": [...],
  "units": [
    {
      "step_indices": [0, 1, 2],
      "unit_before_state": "Firefox 显示 Amazon 地址表单，Ship to 输入框包含旧地址",
      "unit_precondition": ["已进入地址填写页面", "输入框处于激活状态"],
      "unit_after_state": "输入框包含新地址 '123 Main St'",
      "unit_effect": ["新地址已输入到 Ship to 输入框"],
      "unit_type": "REPLACE_CONTENT",
      "unit_intent": "更新收货地址为新地址",
      "phase1_status": "done"
    },
    ...
  ]
}
```

#### Step 5：LLM 第二阶段 — 抽象参数化

逐个 unit 处理，**在同一个 `seg_xxx.jsonl` 的同一个 unit 对象上补充字段**：

```jsonc
{
  "step_indices": [0, 1, 2],
  // --- 第一阶段字段（保留） ---
  "unit_intent": "更新收货地址为新地址",
  "unit_type": "REPLACE_CONTENT",
  "unit_before_state": "...",
  "unit_after_state": "...",
  // --- 第二阶段新增字段 ---
  "abstract_unit_before_state": "浏览器显示电商地址表单，目标输入框处于激活状态",
  "abstract_unit_after_state": "目标输入框已包含新内容",
  "parameters": [
    {"param_name": "new_address", "param_type": "string", "bound_value": "123 Main St"}
  ],
  "parameterized_action_sequence": [
    {"type": "CLICK", "description": "点击 {{target_field}} 输入框"},
    {"type": "KEY", "description": "Ctrl+A 全选"},
    {"type": "TYPING", "description": "输入 {{new_address}}"}
  ]
}
```

> **Unit 实例始终存储在 `seg_xxx.jsonl` 中**，从第一阶段到第二阶段都是在同一文件的同一对象上逐步丰富字段。

#### Step 6：Phase 3 — Canonical Unit 聚合

从所有 `seg_xxx.jsonl` 中读取所有 parameterized units，聚类后生成独立的 CU Base 目录：

```
cu_Base/
├── cu_Base.json            # ① 所有 Canonical Unit 的完整数据
├── transitions.json        # ② CU 间转移频次邻接表
├── faiss_intent.index      # ③ intent embedding 的 FAISS 向量索引（二进制）
├── mappings.json           # ④ faiss_to_cu + instance_to_cu 映射表
└── config.json             # 构建时的超参数快照
```

| 文件 | 内容 |
|------|------|
| `cu_Base.json` | 每个 CU 含：intent、Unit Tree（多路径树结构）、parameter_defs、统计信息（执行次数/成功率/路径统计）、来源用户和实例 ID 列表 |
| `transitions.json` | `{CU_A: {CU_B: 12, CU_C: 3}, ...}` 邻接频次表 |
| `faiss_intent.index` | FAISS 二进制索引，Agent 运行时用于语义检索 |
| `mappings.json` | `faiss_to_cu`（FAISS 内部序号 → cu_id）+ `instance_to_cu`（unit 实例 ID → cu_id） |

### D.3 两层存储的关系

```
seg_xxx.jsonl  →  存储 unit 实例（具体的、带参数绑定的、属于某条轨迹的某个 segment）
cu_Base.json   →  存储 Canonical Unit（多个实例聚合后的抽象操作单元）
mappings.json  →  instance_to_cu 映射将两者关联
```

- Unit 实例是"原料"，始终留在各自的 segment 文件中
- Canonical Unit 是"聚合产物"，存储在 cu_Base 目录中
- 映射表让系统可以从 CU 追溯到具体来自哪些实例

### D.4 一个具体的例子
第一层：seg_xxx.jsonl — 存 unit 实例（原料）
jsonc// === segments/user_1/seg_007.jsonl ===
{
  "segment_id": "seg_007",
  "steps": [/* 原始事件流 */],
  "units": [
    {
      "unit_id": "u_seg007_01",          // 实例的唯一 ID
      "step_indices": [3, 4, 5],
      // ─── Phase 2 第一阶段 ───
      "unit_intent": "更新收货地址为新地址",
      "unit_type": "REPLACE_CONTENT",
      "unit_before_state": "Firefox 显示 Amazon 地址表单，Ship to 输入框包含旧地址 '456 Oak Ave'",
      "unit_after_state": "Ship to 输入框已更新为 '123 Main St'",
      // ─── Phase 2 第二阶段 ───
      "abstract_unit_before_state": "浏览器显示电商地址表单，目标输入框处于激活状态",
      "abstract_unit_after_state": "目标输入框已包含新内容",
      "parameters": [
        {"param_name": "new_address", "param_type": "string", "bound_value": "123 Main St"},
        {"param_name": "target_field", "param_type": "string", "bound_value": "Ship to 输入框"}
      ],
      "parameterized_action_sequence": [
        {"type": "CLICK",   "description": "点击 {{target_field}}"},
        {"type": "KEY",     "description": "Ctrl+A 全选"},
        {"type": "TYPING",  "description": "输入 {{new_address}}"}
      ]
    }
    // ... seg_007 里的其他 units
  ]
}
jsonc// === segments/user_2/seg_012.jsonl ===
{
  "segment_id": "seg_012",
  "steps": [/* ... */],
  "units": [
    {
      "unit_id": "u_seg012_03",
      "step_indices": [8, 9, 10, 11],
      "unit_intent": "修改配送地址",
      "unit_type": "REPLACE_CONTENT",
      "unit_before_state": "Chrome 显示 eBay 配送地址页，Address 栏显示 '789 Pine Rd'",
      "unit_after_state": "Address 栏已改为 '55 Elm St'",
      "abstract_unit_before_state": "浏览器显示电商地址表单，目标输入框处于激活状态",
      "abstract_unit_after_state": "目标输入框已包含新内容",
      "parameters": [
        {"param_name": "new_address", "param_type": "string", "bound_value": "55 Elm St"},
        {"param_name": "target_field", "param_type": "string", "bound_value": "Address 输入栏"}
      ],
      "parameterized_action_sequence": [
        {"type": "CLICK",   "description": "点击 {{target_field}}"},
        {"type": "KEY",     "description": "Ctrl+A 全选"},
        {"type": "KEY",     "description": "Delete 删除"},
        {"type": "TYPING",  "description": "输入 {{new_address}}"}
      ]
    }
  ]
}
jsonc// === segments/user_3/seg_021.jsonl ===
{
  "segment_id": "seg_021",
  "steps": [/* ... */],
  "units": [
    {
      "unit_id": "u_seg021_02",
      "step_indices": [4, 5, 6],
      "unit_intent": "将收货地址替换为新地址",
      "unit_type": "REPLACE_CONTENT",
      "unit_before_state": "Chrome 显示 Target.com 地址表单，Street address 包含 '100 Maple Dr'",
      "unit_after_state": "Street address 已更新为 '200 Cedar Ln'",
      "abstract_unit_before_state": "浏览器显示电商地址表单，目标输入框处于激活状态",
      "abstract_unit_after_state": "目标输入框已包含新内容",
      "parameters": [
        {"param_name": "new_address", "param_type": "string", "bound_value": "200 Cedar Ln"},
        {"param_name": "target_field", "param_type": "string", "bound_value": "Street address 输入框"}
      ],
      "parameterized_action_sequence": [
        {"type": "CLICK",   "description": "点击 {{target_field}}"},
        {"type": "KEY",     "description": "Ctrl+A 全选"},
        {"type": "TYPING",  "description": "输入 {{new_address}}"}
      ]
    }
  ]
}

Phase 3 聚类：这三个实例的 abstract_unit_before_state 几乎相同，intent embedding 的余弦相似度 ≥ τ_cluster，且都属于同一 unit_type: REPLACE_CONTENT，因此被聚到同一个 cluster → 生成一个 Canonical Unit。

第二层：cu_Base.json — 存 Canonical Unit（聚合产物）
jsonc// === cu_Base/cu_Base.json ===
[
  {
    "cu_id": "cu_00042",
    "intent": "将收货地址替换为新地址",         // 取三者中最详细的
    "intent_embedding_id": 42,
    "unit_type": "REPLACE_CONTENT",
    "abstract_state_before": "浏览器显示电商地址表单，目标输入框处于激活状态",
    "abstract_state_after": "目标输入框已包含新内容",

    "unit_tree": {
      "root": "a1",
      "nodes": {
        "a1": {
          "type": "CLICK",
          "description": "点击 {{target_field}}",
          "params": {"target": "{{target_field}}"},
          "children": ["a2"],
          "source_count": 3
        },
        "a2": {
          "type": "KEY",
          "description": "Ctrl+A 全选",
          "params": {},
          "children": ["a3", "a4"],    // ← 分支！两条路径
          "source_count": 3
        },
        "a3": {
          "type": "TYPING",
          "description": "输入 {{new_address}}",
          "params": {"input": "{{new_address}}"},
          "children": [],
          "source_count": 2,           // user_1 和 user_3 走了这条路径
          "can_terminate": true
        },
        "a4": {
          "type": "KEY",
          "description": "Delete 删除",
          "params": {},
          "children": ["a5"],
          "source_count": 1            // 只有 user_2 走了这条路径
        },
        "a5": {
          "type": "TYPING",
          "description": "输入 {{new_address}}",
          "params": {"input": "{{new_address}}"},
          "children": [],
          "source_count": 1,
          "can_terminate": true
        }
      }
    },

    "parameter_defs": [
      {
        "param_name": "new_address",
        "param_type": "string",
        "description": "要填入的新地址",
        "observed_values": ["123 Main St", "55 Elm St", "200 Cedar Ln"]
      },
      {
        "param_name": "target_field",
        "param_type": "string",
        "description": "目标输入框",
        "observed_values": ["Ship to 输入框", "Address 输入栏", "Street address 输入框"]
      }
    ],

    "execution_count": 3,
    "success_count": 3,
    "success_rate": 1.0,
    "path_stats": {
      "select_all_then_type":   {"count": 2, "success": 2, "success_rate": 1.0},
      "select_all_delete_type": {"count": 1, "success": 1, "success_rate": 1.0}
    },

    "source_users": ["user_1", "user_2", "user_3"],
    "source_instance_ids": ["u_seg007_01", "u_seg012_03", "u_seg021_02"],
    "app_context": {"app_name": "Browser", "url_pattern": "*"},
    "first_seen": "2025-03-10T14:22:00Z",
    "last_seen": "2025-03-28T09:15:00Z"
  }
  // ... 其他 CU
]

桥梁：mappings.json — 把两层连起来
jsonc// === cu_Base/mappings.json ===
{
  "faiss_to_cu": {
    "42": "cu_00042"       // FAISS 内部序号 42 → cu_00042
    // ...
  },
  "instance_to_cu": {
    "u_seg007_01": "cu_00042",   // user_1 的那个实例 → 属于 cu_00042
    "u_seg012_03": "cu_00042",   // user_2 的那个实例 → 属于 cu_00042
    "u_seg021_02": "cu_00042",   // user_3 的那个实例 → 属于 cu_00042
    // ... 其他所有 unit 实例的映射
  }
}

总结一下数据流向：
三个不同用户、不同网站、不同具体地址的操作，各自以 unit 实例的形式留在 seg_007、seg_012、seg_021 中（带着各自的 bound_value）。Phase 3 发现它们的抽象语义一致，聚合成 cu_00042 这一个 Canonical Unit，其中 Unit Tree 还保留了"Ctrl+A 后直接输入"和"Ctrl+A → Delete → 再输入"两条操作路径的分支。而 mappings.json 里的 instance_to_cu 让系统随时可以从 CU 追溯回具体是哪些实例贡献了它。