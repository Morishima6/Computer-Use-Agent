# deal_action

用于处理原始 `report.json` 的脚本集合。

## 流程

1. `generate_report_cutting_time.py`
按时间切分 `steps`，生成 `steps_time1`, `steps_time2`, ...

2. `generate_report_cutting_window.py`
在每个时间段内按窗口变化继续切分，生成 `steps_time1_window1`, `steps_time1_window2`, ...

3. `llm_window_merge.py`
用 LLM 判断相邻窗口段是否应合并，生成 `steps_time1_mix_window1`, ...，并附带 `window_merge_decisions`

4. `llm_task_cut_mix_to_files.py`
再用 LLM 把每个 `mix_window` 段切成更高层任务，并导出为多个任务 JSON 文件

## 命令行用法

### 1. 按时间切分

```powershell
python generate_report_cutting_time.py INPUT_REPORT_JSON [-o OUTPUT_JSON] [-t THRESHOLD_SECONDS]
```

示例：

```powershell
python generate_report_cutting_time.py E:\path\to\report.json -o E:\path\to\report_cutting_time.json -t 30
```

如果不传 `-o`，默认输出为：

```text
<input_stem>_cutting_time.json
```

### 2. 按窗口切分

```powershell
python generate_report_cutting_window.py INPUT_TIME_JSON [-o OUTPUT_JSON]
```

示例：

```powershell
python generate_report_cutting_window.py E:\path\to\report_cutting_time.json -o E:\path\to\report_cutting_window.json
```

如果不传 `-o`，默认会把文件名中的 `_cutting_time` 替换成 `_cutting_window`；如果没有这个后缀，就直接追加 `_cutting_window`。

### 3. LLM 窗口合并

运行前需要设置：

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

如果使用兼容接口，还可以设置：

```powershell
$env:OPENAI_BASE_URL="your_base_url"
```

命令：

```powershell
python llm_window_merge.py INPUT_WINDOW_JSON SCREENSHOT_ROOT [-o OUTPUT_JSON] [-m MODEL]
```

示例：

```powershell
python llm_window_merge.py E:\path\to\report_cutting_window.json E:\path\to\screenshots -o E:\path\to\report_cutting_window_mix.json -m gpt-5.1
```

如果不传 `-o`，默认输出为：

```text
<input_stem>_mix.json
```

### 4. LLM 任务切分并导出多个文件

```powershell
python llm_task_cut_mix_to_files.py INPUT_MIX_JSON [-o OUTPUT_DIR] [-m MODEL] [--prefix PREFIX]
```

示例：

```powershell
python llm_task_cut_mix_to_files.py E:\path\to\report_cutting_window_mix.json -o E:\path\to\output_dir -m gpt-5.1 --prefix report_cutting_llm_task_
```

如果不传 `-o`，默认输出到输入文件所在目录。

## 典型完整流程

```powershell
python generate_report_cutting_time.py E:\path\to\report.json
python generate_report_cutting_window.py E:\path\to\report_cutting_time.json
python llm_window_merge.py E:\path\to\report_cutting_window.json E:\path\to\screenshots
python llm_task_cut_mix_to_files.py E:\path\to\report_cutting_window_mi

