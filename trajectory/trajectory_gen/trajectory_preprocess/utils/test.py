from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[5]
CORE_AGENT_ROOT = PROJECT_ROOT / "Core-Agent"

if str(CORE_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_AGENT_ROOT))

from trajectory.trajectory_gen.trajectory_preprocess.utils.compare_screenshot_similarity import (
    compare_screenshot_similarity,
    compare_screenshot_similarity_report,
)
from trajectory.trajectory_gen.trajectory_preprocess.utils.vlm_similarity_gate import (
    judge_action
)


SIMILARITY_THRESHOLD = 0.98

image_path_a = (
    PROJECT_ROOT
    / "tmp_trace"
    / "documents-export-2026-4-7"
    / "chrome_in-domain"
    / "part-2"
    / "Unknown_uuid_20260406_143211"
    / "result"
    / "screenshots"
    / "s137_before_click.png"
)
image_path_b = (
    PROJECT_ROOT
    / "tmp_trace"
    / "documents-export-2026-4-7"
    / "chrome_in-domain"
    / "part-2"
    / "Unknown_uuid_20260406_143211"
    / "result"
    / "screenshots"
    / "s137_after_click.png"
)
image_path_c = (
    PROJECT_ROOT
    / "tmp_trace"
    / "documents-export-2026-4-7"
    / "chrome_in-domain"
    / "part-2"
    / "Unknown_uuid_20260406_143211"
    / "result"
    / "screenshots"
    / "s138_before_click.png"
)

score = compare_screenshot_similarity(image_path_a, image_path_b)
report = compare_screenshot_similarity_report(image_path_a, image_path_b)

print(f"score: {score:.6f}")
print(report)

if score > SIMILARITY_THRESHOLD:
    print("规则判断相似>0.98")
    # 再判断与下一步操作before的截图是否相似
    print("继续判断下一步操作的before截图与当前操作的after截图的相似度...")
    score_next = compare_screenshot_similarity(image_path_b, image_path_c)
    report_next = compare_screenshot_similarity_report(image_path_b, image_path_c)
    print(f"score: {score_next:.6f}")

    if score_next > SIMILARITY_THRESHOLD:
        print("规则判断相似>0.98，当前操作的after截图与下一步操作的before截图也相似，可能是误判，继续使用VLM判断...")
        action_type = "click"
        # 重复3次验证，增加鲁棒性
        # results = []
        cnt = 0
        for i in range(3):
            result = judge_action(action_type, image_path_a, image_path_b, image_path_c, backend="qwen")
            if result["Judgment"] == "yes":
                cnt += 1
            # results.append(result)
            print(result)
        
        # 3次都同意才删
        # if result["Judgment"] == "yes":
        if cnt == 3:
            print("VLM decision: yes, before/after screenshots are highly similar.")
        else:
            print("VLM decision: no, before/after screenshots have meaningful state changes.")
    else:
        print("规则判断相似>0.98，但当前操作的after截图与下一步操作的before截图相似度较低，说明有明显状态变化，不太可能是误判，跳过VLM判断，不删除。")
else:
    print("Rule similarity is below threshold, skip VLM.")
