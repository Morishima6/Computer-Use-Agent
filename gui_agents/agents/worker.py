from functools import partial
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gui_agents.agents.grounding import ACI
from gui_agents.agents.worker_components.active_cu_prompt import (
    build_active_cu_prompt,
)
from gui_agents.agents.worker_components.cu_runtime import (
    apply_reflection_control,
    build_cu_retrieval_query,
    reset_cu_runtime_to_idle,
    should_run_cu_retrieval,
    strip_cu_retrieval_context,
    update_cu_selection_state,
    wrap_cu_retrieval_context,
)
from gui_agents.agents.worker_components.cu_selection_parser import (
    extract_cu_selection_section,
    parse_reference_params,
    parse_cu_selection_from_plan,
    validate_cu_selection,
)
from gui_agents.agents.worker_components.reflection_controller import (
    build_reflection_fallback_control,
    generate_reflection,
    is_valid_reflection_control,
    parse_reflection_control,
    sanitize_reflection_for_planner,
)
from gui_agents.agents.worker_components.ui_state_extractor import (
    generate_ui_state_text,
    get_state_text_agent,
)
from gui_agents.core.module import BaseModule
from gui_agents.memory.procedural_memory import PROCEDURAL_MEMORY
from gui_agents.utils.common_utils import (
    call_llm_formatted,
    create_pyautogui_code,
    parse_code_from_string,
)
from gui_agents.utils.formatters import (
    CODE_VALID_FORMATTER,
    SINGLE_ACTION_FORMATTER,
    SINGLE_CODE_BLOCK_FORMATTER,
    SINGLE_SECTION_SET_FORMATTER,
)
from trajectory.retrieval.cu_retrieval import (
    CURetriever,
    CUStore,
    build_cu_retrieval_prompt,
)

logger = logging.getLogger("desktopenv.agent")


class Worker(BaseModule):
    def __init__(
        self,
        worker_engine_params: Dict,
        grounding_agent: ACI,
        platform: str = "ubuntu",
        max_trajectory_length: int = 8,
        enable_reflection: bool = True,
        enable_cu_retrieval: bool = False,
        disable_code_agent: bool = False,
    ):
        super().__init__(worker_engine_params, platform)

        self.temperature = worker_engine_params.get("temperature", 0.0)
        self.use_thinking = worker_engine_params.get("model", "") in [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-sonnet-4-5-20250929",
        ]
        self.grounding_agent = grounding_agent
        self.max_trajectory_length = max_trajectory_length
        self.enable_reflection = enable_reflection
        self.enable_cu_retrieval = enable_cu_retrieval
        self.disable_code_agent = disable_code_agent
        self._state_text_agent = None

        self.cu_store = None
        self.cu_retriever = None
        if self.enable_cu_retrieval:
            repo_root = Path(__file__).resolve().parents[2]
            self.cu_store = CUStore(repo_root / "trajectory" / "cu_base")
            self.cu_retriever = CURetriever(self.cu_store)
        self.reset()

    def reset(self):
        skipped_actions = ["set_cell_values"] if self.platform != "linux" else []

        if self.disable_code_agent:
            skipped_actions.append("call_code_agent")
        elif not getattr(self.grounding_agent, "env", None) or not getattr(
            getattr(self.grounding_agent, "env", None), "controller", None
        ):
            skipped_actions.append("call_code_agent")

        self.generator_agent = self._create_agent(
            PROCEDURAL_MEMORY.construct_simple_worker_procedural_memory(
                type(self.grounding_agent), skipped_actions=skipped_actions
            ).replace("CURRENT_OS", self.platform)
        )
        self.reflection_agent = self._create_agent(
            PROCEDURAL_MEMORY.REFLECTION_ON_TRAJECTORY
        )

        self.turn_count = 0
        self.worker_history = []
        self.reflections = []
        self.cost_this_turn = 0
        self.screenshot_inputs = []
        self._reset_cu_runtime_state()

    def _reset_cu_runtime_state(self) -> None:
        self.last_ui_state_text = ""
        self.last_cu_candidates = []
        self.cu_runtime_mode = "idle"
        self.active_cu_id: Optional[str] = None
        self.active_path_id: Optional[str] = None
        self.active_reference_params: Dict[str, Any] = {}
        self.last_completed_cu_id: Optional[str] = None
        self.intent_predict = ""
        self.cu_retry_count = 0
        self.consecutive_no_progress_count = 0
        self.bad_reflection_count = 0
        self.cu_control: Dict[str, Any] = {
            "selection_source": None,
            "valid": False,
            "status": "idle",
            "reason": "",
        }
# ==============================UI 状态文本生成模块=========================
    # def _get_state_text_agent(self):
    #     return get_state_text_agent(self)

    def _generate_ui_state_text(self, model_obs: Dict) -> str:
        return generate_ui_state_text(self, model_obs, logger)

# ===========CU 选择结果解析模块，从 planner 输出里抽取结构化 CU 选择结果====
    # def _parse_reference_params(self, raw_text: str) -> Dict[str, Any]:
    #     return parse_reference_params(raw_text)

    # def _extract_cu_selection_section(self, plan: str) -> str:
    #     return extract_cu_selection_section(plan)

    def _parse_cu_selection_from_plan(self, plan: str) -> Dict[str, Any]:
        return parse_cu_selection_from_plan(plan)

    def _validate_cu_selection(
        self, parsed_selection: Dict[str, Any], retrieval_result: Any
    ) -> Dict[str, Any]:
        return validate_cu_selection(parsed_selection, retrieval_result)
    
# ===========================CU retrieval运行时状态机模块==============================
    def _update_cu_selection_state(self, validated_selection: Dict[str, Any]) -> None:
        update_cu_selection_state(self, validated_selection)

    def _should_run_cu_retrieval(self) -> bool:
        return should_run_cu_retrieval(self)

    def _build_cu_retrieval_query(self, instruction: str) -> str:
        return build_cu_retrieval_query(self, instruction)

    # def _clear_active_cu_state(self) -> None:
    #     clear_active_cu_state(self)

    def _apply_reflection_control(self, control: Dict[str, Any]) -> None:
        apply_reflection_control(self, control)

# ============Active CU prompt 构造模块，把当前 CU/path 状态渲染给 planner=====
    # def _get_active_cu(self) -> Optional[Dict[str, Any]]:
    #     return get_active_cu(self)

    # def _get_active_path_payload(self) -> Optional[Dict[str, Any]]:
    #     return get_active_path_payload(self)

    def _build_active_cu_prompt(self) -> str:
        return build_active_cu_prompt(self)

#  ==============================Reflection 控制模块============================
    def _generate_reflection(
        self, instruction: str, model_obs: Dict
    ) -> Tuple[str, str, Dict[str, Any]]:
        return generate_reflection(self, instruction, model_obs, logger)
    # def _parse_reflection_control(self, reflection_text: str) -> Dict[str, Any]:
    #     return parse_reflection_control(reflection_text)

    # def _is_valid_reflection_control(self, control: Dict[str, Any]) -> bool:
    #     return is_valid_reflection_control(control)

    # def _build_reflection_fallback_control(self) -> Dict[str, Any]:
    #     return build_reflection_fallback_control(self)
    
    def flush_messages(self):
        engine_type = self.engine_params.get("engine_type", "")

        if engine_type in ["anthropic", "openai", "gemini"]:
            max_images = self.max_trajectory_length
            for agent in [self.generator_agent, self.reflection_agent]:
                if agent is None:
                    continue
                img_count = 0
                for i in range(len(agent.messages) - 1, -1, -1):
                    for j in range(len(agent.messages[i]["content"])):
                        if "image" in agent.messages[i]["content"][j].get("type", ""):
                            img_count += 1
                            if img_count > max_images:
                                del agent.messages[i]["content"][j]
        else:
            if len(self.generator_agent.messages) > 2 * self.max_trajectory_length + 1:
                self.generator_agent.messages.pop(1)
                self.generator_agent.messages.pop(1)
            if len(self.reflection_agent.messages) > self.max_trajectory_length + 1:
                self.reflection_agent.messages.pop(1)

    def generate_next_action(self, instruction: str, obs: Dict) -> Tuple[Dict, List]:
        self.grounding_agent.assign_screenshot(obs)
        self.grounding_agent.set_task_instruction(instruction)
        model_obs = self.grounding_agent.get_model_observation()

        generator_message = (
            ""
            if self.turn_count > 0
            else "The initial screen is provided. No action has been taken yet."
        )

        if self.turn_count == 0:
            prompt_with_instructions = self.generator_agent.system_prompt.replace(
                "TASK_DESCRIPTION", instruction
            )
            self.generator_agent.add_system_prompt(prompt_with_instructions)

        reflection, reflection_thoughts, reflection_control = self._generate_reflection(
            instruction, model_obs
        )
        self._apply_reflection_control(reflection_control)
        planner_reflection = sanitize_reflection_for_planner(reflection)
        if planner_reflection:
            generator_message += (
                "REFLECTION: You may use this reflection on the previous action and "
                f"overall trajectory:\n{planner_reflection}\n"
            )

        generator_message += (
            f"\nCurrent Text Buffer = [{','.join(self.grounding_agent.notes)}]\n"
        )
# active_cu状态下构造prompt
        active_cu_prompt = ""
        if self.enable_cu_retrieval and self.cu_runtime_mode == "active_cu":
            active_cu_prompt = self._build_active_cu_prompt()
            if active_cu_prompt:
                generator_message += "\n" + active_cu_prompt + "\n"
# 
        ui_state_text = ""
        retrieval_result = None
        retrieval_executed = False
        retrieval_query = ""
        planner_history_message = ""
        self.last_cu_candidates = []
        if self._should_run_cu_retrieval():
            retrieval_executed = True
            # ui_state_text = self._generate_ui_state_text(model_obs)
            # self.last_ui_state_text = ui_state_text
            retrieval_query = self._build_cu_retrieval_query(instruction)
            retrieval_result = self.cu_retriever.retrieve(
                retrieval_query,
                last_selected_cu_id=self.last_completed_cu_id,
            )
            self.last_cu_candidates = retrieval_result.merged_candidates
            retrieval_prompt = build_cu_retrieval_prompt(retrieval_result)
            if retrieval_prompt:
                generator_message += (
                    "\n" + wrap_cu_retrieval_context(retrieval_prompt) + "\n"
                )
        elif self.enable_cu_retrieval:
            self.last_cu_candidates = []
        else:
            self._reset_cu_runtime_state()

#=====================================================================================
        if (
            hasattr(self.grounding_agent, "last_code_agent_result")
            and self.grounding_agent.last_code_agent_result is not None
        ):
            code_result = self.grounding_agent.last_code_agent_result
            generator_message += "\nCODE AGENT RESULT:\n"
            generator_message += (
                f"Task/Subtask Instruction: {code_result['task_instruction']}\n"
            )
            generator_message += f"Steps Completed: {code_result['steps_executed']}\n"
            generator_message += f"Max Steps: {code_result['budget']}\n"
            generator_message += (
                f"Completion Reason: {code_result['completion_reason']}\n"
            )
            generator_message += f"Summary: {code_result['summary']}\n"
            if code_result["execution_history"]:
                generator_message += "Execution History:\n"
                for i, step in enumerate(code_result["execution_history"]):
                    action = step["action"]
                    if "```python" in action:
                        code_start = action.find("```python") + 9
                        code_end = action.find("```", code_start)
                        if code_end != -1:
                            python_code = action[code_start:code_end].strip()
                            generator_message += (
                                f"Step {i+1}: \n```python\n{python_code}\n```\n"
                            )
                        else:
                            generator_message += f"Step {i+1}: \n{action}\n"
                    elif "```bash" in action:
                        code_start = action.find("```bash") + 7
                        code_end = action.find("```", code_start)
                        if code_end != -1:
                            bash_code = action[code_start:code_end].strip()
                            generator_message += (
                                f"Step {i+1}: \n```bash\n{bash_code}\n```\n"
                            )
                        else:
                            generator_message += f"Step {i+1}: \n{action}\n"
                    else:
                        generator_message += f"Step {i+1}: \n{action}\n"
            generator_message += "\n"

            try:
                from datetime import datetime
                import os

                logs_dir = "logs"
                if not os.path.exists(logs_dir):
                    os.makedirs(logs_dir)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = (
                    f"logs/code_agent_result_step_{self.turn_count + 1}_{timestamp}.txt"
                )

                with open(filename, "w") as f:
                    f.write(f"CODE AGENT RESULT - Step {self.turn_count + 1}\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(
                        f"Task/Subtask Instruction: {code_result['task_instruction']}\n"
                    )
                    f.write(f"Steps Completed: {code_result['steps_executed']}\n")
                    f.write(f"Max Steps: {code_result['budget']}\n")
                    f.write(
                        f"Completion Reason: {code_result['completion_reason']}\n"
                    )
                    f.write(f"Summary: {code_result['summary']}\n")
                    if code_result["execution_history"]:
                        f.write("\nExecution History:\n")
                        for i, step in enumerate(code_result["execution_history"]):
                            f.write(f"\nStep {i+1}:\n")
                            f.write(f"Action: {step['action']}\n")
                            if "thoughts" in step:
                                f.write(f"Thoughts: {step['thoughts']}\n")

                logger.info("Code agent result saved to: %s", filename)
            except Exception as e:
                logger.error("Failed to save code agent result to file: %s", e)

            log_message = "\nCODE AGENT RESULT:\n"
            log_message += (
                f"Task/Subtask Instruction: {code_result['task_instruction']}\n"
            )
            log_message += f"Steps Completed: {code_result['steps_executed']}\n"
            log_message += f"Max Steps: {code_result['budget']}\n"
            log_message += (
                f"Completion Reason: {code_result['completion_reason']}\n"
            )
            log_message += f"Summary: {code_result['summary']}\n"
            if code_result["execution_history"]:
                log_message += "Execution History (truncated):\n"
                total_steps = len(code_result["execution_history"])
                for i, step in enumerate(code_result["execution_history"]):
                    if i < 3 or i >= total_steps - 2:
                        action = step["action"]
                        if "```python" in action:
                            code_start = action.find("```python") + 9
                            code_end = action.find("```", code_start)
                            if code_end != -1:
                                python_code = action[code_start:code_end].strip()
                                log_message += (
                                    f"Step {i+1}: ```python\n{python_code}\n```\n"
                                )
                            else:
                                log_message += f"Step {i+1}: {action}\n"
                        elif "```bash" in action:
                            code_start = action.find("```bash") + 7
                            code_end = action.find("```", code_start)
                            if code_end != -1:
                                bash_code = action[code_start:code_end].strip()
                                log_message += (
                                    f"Step {i+1}: ```bash\n{bash_code}\n```\n"
                                )
                            else:
                                log_message += f"Step {i+1}: {action}\n"
                        else:
                            log_message += f"Step {i+1}: {action}\n"
                    elif i == 3 and total_steps > 5:
                        log_message += (
                            f"... (truncated {total_steps - 5} steps) ...\n"
                        )

            logger.info(
                "WORKER_CODE_AGENT_RESULT_SECTION - Step %s: Code agent result added to "
                "generator message:\n%s",
                self.turn_count + 1,
                log_message,
            )

            self.grounding_agent.last_code_agent_result = None
#=================================================================================================

        self.generator_agent.add_message(
            generator_message, image_content=model_obs["screenshot"], role="user"
        )
        planner_history_message = strip_cu_retrieval_context(generator_message)
        planner_user_message_index = len(self.generator_agent.messages) - 1

        format_checkers = [
            SINGLE_SECTION_SET_FORMATTER,
            SINGLE_CODE_BLOCK_FORMATTER,
            SINGLE_ACTION_FORMATTER,
            partial(CODE_VALID_FORMATTER, self.grounding_agent, model_obs),
        ]
        plan = call_llm_formatted(
            self.generator_agent,
            format_checkers,
            temperature=self.temperature,
            use_thinking=self.use_thinking,
        )
        print("=" * 100)
        print("**Generator message**:\n", generator_message)
        print("-" * 80)
        print("**Raw plan response**:\n", plan)
        print("=" * 100)
        self.worker_history.append(plan)
        if planner_history_message != generator_message:
            self.generator_agent.replace_message_at(
                planner_user_message_index,
                planner_history_message,
                image_content=model_obs["screenshot"],
            )
        self.generator_agent.add_message(plan, role="assistant")
        logger.info("PLAN:\n %s", plan)
        parsed_selection = self._parse_cu_selection_from_plan(plan)
        has_cu_selection_section = bool(extract_cu_selection_section(plan))
        if self.enable_cu_retrieval and retrieval_result is not None:
            validated_selection = self._validate_cu_selection(
                parsed_selection, retrieval_result
            )
            self._update_cu_selection_state(validated_selection)
        elif (
            self.enable_cu_retrieval
            and has_cu_selection_section
            and parsed_selection.get("selected_cu_id") is None
        ):
            reset_cu_runtime_to_idle(self)

        plan_code = parse_code_from_string(plan)
        try:
            assert plan_code, "Plan code should not be empty"
            exec_code = create_pyautogui_code(self.grounding_agent, plan_code, model_obs)
        except Exception as e:
            logger.error(
                "Could not evaluate the following plan code:\n%s\nError: %s",
                plan_code,
                e,
            )
            exec_code = self.grounding_agent.wait(1.333)

        executor_info = {
            "plan": plan,
            "plan_code": plan_code,
            "exec_code": exec_code,
            "ui_state_text": ui_state_text,
            "cu_retrieval_executed": retrieval_executed,
            "cu_retrieval_query": retrieval_query,
            "cu_candidates": (
                [candidate.cu_id for candidate in retrieval_result.merged_candidates]
                if retrieval_result is not None
                else []
            ),
            "selected_cu_id": self.active_cu_id,
            "selected_path_id": self.active_path_id,
            "reference_params": dict(self.active_reference_params),
            "cu_selection_meta": dict(self.cu_control),
            "cu_runtime_mode": self.cu_runtime_mode,
            "active_cu_prompt": active_cu_prompt,
            "last_completed_cu_id": self.last_completed_cu_id,
            "intent_predict": self.intent_predict,
            "reflection": reflection,
            "reflection_thoughts": reflection_thoughts,
            "reflection_control": dict(reflection_control),
            "code_agent_output": (
                self.grounding_agent.last_code_agent_result
                if hasattr(self.grounding_agent, "last_code_agent_result")
                and self.grounding_agent.last_code_agent_result is not None
                else None
            ),
        }
        self.turn_count += 1
        self.screenshot_inputs.append(model_obs["screenshot"])
        self.flush_messages()
        return executor_info, [exec_code]
