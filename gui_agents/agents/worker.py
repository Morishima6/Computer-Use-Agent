from functools import partial
import json
import logging
import textwrap
from typing import Dict, List, Tuple, Callable, Optional

from gui_agents.agents.grounding import ACI
from gui_agents.core.module import BaseModule
from gui_agents.memory.procedural_memory import PROCEDURAL_MEMORY
from gui_agents.utils.common_utils import (
    call_llm_safe,
    call_llm_formatted,
    parse_code_from_string,
    split_thinking_response,
    create_pyautogui_code,
)
from gui_agents.core.mllm import LMMAgent
from gui_agents.utils.formatters import (
    SINGLE_ACTION_FORMATTER,
    CODE_VALID_FORMATTER,
)
from gui_agents.agents.step_retrieval_summary import (
    create_step_retrieval_summarizer,
    generate_step_retrieval_summary,
)
from trajectory.retrieval.step_retrieval import build_prompt_from_step

logger = logging.getLogger("desktopenv.agent")

_STEP_RETRIEVAL_REF_START = "<STEP_RETRIEVAL_REFERENCE_START>"
_STEP_RETRIEVAL_REF_END = "<STEP_RETRIEVAL_REFERENCE_END>"


def _strip_step_retrieval_reference_from_text(text: str) -> str:
    """Remove any step-retrieval reference blocks that accidentally leak into model outputs.

    Design intent:
    - Low-similarity step retrieval (<k) may be provided to the planning agent as *reference-only*.
    - Reflection should rely on the actual plan output, not on the raw retrieved reference text.
    - If the planning model echoes the reference block, strip it before saving into worker_history,
      so it won't be fed into reflection context in the next turn.
    """
    if not text:
        return text

    start = text.find(_STEP_RETRIEVAL_REF_START)
    end = text.find(_STEP_RETRIEVAL_REF_END)
    if start != -1 and end != -1 and end > start:
        end = end + len(_STEP_RETRIEVAL_REF_END)
        text = (text[:start] + text[end:]).strip()

    # Best-effort: if the model echoed the label without markers, drop that section.
    label = "STEP RETRIEVAL REFERENCE"
    idx = text.find(label)
    if idx != -1:
        text = (text[:idx]).strip()

    return text


def _do_step_retrieval(
    obs: Dict,
    threshold: float,
    step_retrieval: Callable,
    screenshot_explainer: Optional[LMMAgent],
    summarizer,
    engine_params: Dict,
) -> Tuple[bool, str, Optional[Dict]]:
    """
    Shared step retrieval logic used by both pre-planning and post-failure paths.

    Returns:
        (False, "", None)
            - No hit, continue normal flow.

        (True, retrieved_content, step_data)
            - Hit found; caller should inject retrieved_content as a hint
              into the generator_message and then re-enter the planning loop.
              step_data may contain full_step_data with after_effects/meta.
    """
    # 1. Generate screenshot_explanation if missing
    if "screenshot_explanation" not in obs and "screenshot" in obs:
        model_obs = obs
        try:
            if screenshot_explainer is None:
                system_prompt = (
                    "You are a UI state describer for step retrieval.\n"
                    "Given a screenshot of a desktop app, describe the current screen state.\n"
                    "Focus on: active app/window, page/view name, visible UI elements "
                    "(buttons/menus/tabs/dialogs), and important visible text.\n"
                    "Be concise but specific. Output plain text only."
                )
                screenshot_explainer = LMMAgent(
                    engine_params=engine_params, system_prompt=system_prompt
                )

            prompt = (
                "Please generate a textual description of the current screen state "
                "based on the screenshot (for semantic retrieval).\n"
                "Requirements: Describe the currently visible interface and key text. "
                "Do not output coordinates or code."
            )
            screenshot_explainer.reset()
            screenshot_explainer.add_message(
                text_content=prompt,
                image_content=model_obs["screenshot"],
                role="user",
                put_text_last=True,
            )
            obs["screenshot_explanation"] = call_llm_safe(
                screenshot_explainer, temperature=0.0
            ).strip()
        except Exception as e:
            logger.error(f"SCREENSHOT EXPLANATION GENERATION FAILED: {e}")

    # 2. Generate step_retrieval_summary if missing
    if "step_retrieval_summary" not in obs and "screenshot" in obs:
        try:
            if summarizer is None:
                summarizer = create_step_retrieval_summarizer(engine_params)

            obs["step_retrieval_summary"] = generate_step_retrieval_summary(
                obs["screenshot"],
                engine_params,
                summarizer=summarizer,
            ).strip()
        except Exception as e:
            logger.error(f"STEP RETRIEVAL SUMMARY GENERATION FAILED: {e}")
            obs["step_retrieval_summary"] = obs.get("screenshot_explanation", "")

    # 3. Call step retrieval
    if "screenshot_explanation" not in obs or "step_retrieval_summary" not in obs:
        return False, "", None

    try:
        step_data = step_retrieval(obs, threshold)
    except Exception as e:
        logger.error(f"STEP RETRIEVAL FAILED: {e}")
        return False, "", None

    if not step_data:
        return False, "", None

    # 4. Build prompt from step data
    try:
        response_prompt = build_prompt_from_step(step_data)
    except Exception as e:
        logger.error(f"BUILD STEP PROMPT FAILED: {e}")
        return False, "", None

    if not isinstance(response_prompt, dict):
        return False, "", None

    similarity_above_k = response_prompt.get("similarity_above_k", False)
    retrieval_content = response_prompt.get("content", "")
    is_append_to_plan = response_prompt.get("isAppend2Plan", False)

    # 5. High-similarity hit → inject hint (no direct execution in failure path)
    hint_block = ""
    if retrieval_content:
        hint_block = (
            f"\n{_STEP_RETRIEVAL_REF_START}\n"
            + retrieval_content
            + f"\n{_STEP_RETRIEVAL_REF_END}\n"
        )
        logger.info(
            "STEP RETRIEVAL: Found reference (similarity_above_k=%s), injecting hint "
            "into generator_message.",
            similarity_above_k,
        )
        return True, hint_block, step_data

    return False, "", None


class Worker(BaseModule):
    def __init__(
        self,
        worker_engine_params: Dict,
        grounding_agent: ACI,
        platform: str = "ubuntu",
        max_trajectory_length: int = 8,
        enable_reflection: bool = True,
        task_retrieval: Optional[Callable[[str], str]] = None,
        step_retrieval: Optional[Callable[[Dict, float], object]] = None,
        step_retrieval_threshold: float = 0.8,
        enable_verify: bool = False,
        verify_engine_params: Optional[Dict] = None,
    ):
        """
        Worker receives the main task and generates actions, without the need of hierarchical planning
        Args:
            worker_engine_params: Dict
                Parameters for the worker agent
            grounding_agent: Agent
                The grounding agent to use
            platform: str
                OS platform the agent runs on (darwin, linux, windows)
            max_trajectory_length: int
                The amount of images turns to keep
            enable_reflection: bool
                Whether to enable reflection
            enable_verify: bool
                Whether to enable pre-execution plan verification
            verify_engine_params: Optional[Dict]
                Separate engine params for the verify agent (uses worker params if None)
        """
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
        self.task_retrieval = task_retrieval
        self.step_retrieval = step_retrieval
        self.step_retrieval_threshold = step_retrieval_threshold
        self.enable_verify = enable_verify
        self.verify_engine_params = verify_engine_params
        self._screenshot_explainer = None
        self._step_retrieval_summarizer = None
        self._verify_agent = None
        self.reset()

    def reset(self):
        if self.platform != "linux":
            skipped_actions = ["set_cell_values"]
        else:
            skipped_actions = []

        # Hide code agent action entirely if no env/controller is available
        if not getattr(self.grounding_agent, "env", None) or not getattr(
            getattr(self.grounding_agent, "env", None), "controller", None
        ):
            skipped_actions.append("call_code_agent")

        sys_prompt = PROCEDURAL_MEMORY.construct_simple_worker_procedural_memory(
            type(self.grounding_agent), skipped_actions=skipped_actions
        ).replace("CURRENT_OS", self.platform)

        self.generator_agent = self._create_agent(sys_prompt)
        self.reflection_agent = self._create_agent(
            PROCEDURAL_MEMORY.REFLECTION_ON_TRAJECTORY
        )
        self._verify_agent = None

        self.turn_count = 0
        self.worker_history = []
        self.reflections = []
        self.cost_this_turn = 0
        self.screenshot_inputs = []
        self._screenshot_explainer = None
        self._step_retrieval_summarizer = None
        self.last_step_retrieval_after_effects = None
        self.last_step_retrieval_meta = None


    def flush_messages(self):
        """Flush messages based on the model's context limits.

        This method ensures that the agent's message history does not exceed the maximum trajectory length.

        Side Effects:
            - Modifies the messages of generator, reflection, and bon_judge agents to fit within the context limits.
        """
        engine_type = self.engine_params.get("engine_type", "")

        # Flush strategy for long-context models: keep all text, only keep latest images
        if engine_type in ["anthropic", "openai", "gemini"]:
            max_images = self.max_trajectory_length
            for agent in [self.generator_agent, self.reflection_agent]:
                if agent is None:
                    continue
                # keep latest k images
                img_count = 0
                for i in range(len(agent.messages) - 1, -1, -1):
                    for j in range(len(agent.messages[i]["content"])):
                        if "image" in agent.messages[i]["content"][j].get("type", ""):
                            img_count += 1
                            if img_count > max_images:
                                del agent.messages[i]["content"][j]

        # Flush strategy for non-long-context models: drop full turns
        else:
            # generator msgs are alternating [user, assistant], so 2 per round
            if len(self.generator_agent.messages) > 2 * self.max_trajectory_length + 1:
                self.generator_agent.messages.pop(1)
                self.generator_agent.messages.pop(1)
            # reflector msgs are all [(user text, user image)], so 1 per round
            if len(self.reflection_agent.messages) > self.max_trajectory_length + 1:
                self.reflection_agent.messages.pop(1)

    def _generate_reflection(self, instruction: str, obs: Dict) -> Tuple[str, str]:
        """
        Generate a reflection based on the current observation and instruction.

        Args:
            instruction (str): The task instruction.
            obs (Dict): The current observation containing the screenshot.

        Returns:
            Optional[str, str]: The generated reflection text and thoughts, if any (turn_count > 0).

        Side Effects:
            - Updates reflection agent's history
            - Generates reflection response with API call
        """
        reflection = None
        reflection_thoughts = None
        model_obs = self.grounding_agent.get_model_observation()
        if self.enable_reflection:
            # Load the initial message
            if self.turn_count == 0:
                # NOTE: Task Retrieval
                task_retrieval_result = ""
                if self.task_retrieval is not None:
                    task_retrieval_result = self.task_retrieval(instruction)
                    logger.info(f"🌵 Task Retrieval: \n{task_retrieval_result}")
                
                text_content = textwrap.dedent(
                    f"""
                    Task Description: {instruction}
                    Current Trajectory below:
                    """
                )
                updated_sys_prompt = (
                    self.reflection_agent.system_prompt + "\n" + task_retrieval_result + "\n" + text_content
                )
                self.reflection_agent.add_system_prompt(updated_sys_prompt)
                self.reflection_agent.add_message(
                    text_content="The initial screen is provided. No action has been taken yet.",
                    image_content=model_obs["screenshot"],
                    role="user",
                )
            # Load the latest action
            else:
                reference_text = ""
                if self.last_step_retrieval_after_effects:
                    reference_text = (
                        "\n\nREFERENCE (retrieved step action_after_effects):\n"
                        + str(self.last_step_retrieval_after_effects)
                        + "\n"
                    )
                self.reflection_agent.add_message(
                    text_content=self.worker_history[-1] + reference_text,
                    image_content=model_obs["screenshot"],
                    role="user",
                )
                full_reflection = call_llm_safe(
                    self.reflection_agent,
                    temperature=self.temperature,
                    use_thinking=self.use_thinking,
                )
                reflection, reflection_thoughts = split_thinking_response(
                    full_reflection
                )
                self.reflections.append(reflection)
                logger.info("REFLECTION THOUGHTS: %s", reflection_thoughts)
                logger.info("REFLECTION: %s", reflection)
                self.last_step_retrieval_after_effects = None
                self.last_step_retrieval_meta = None
        return reflection, reflection_thoughts

    def _verify_plan(
        self, plan_text: str, screenshot, engine_params: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Verify whether the planner's output is executable on the current screen.

        Args:
            plan_text: The raw plan text output from the planner.
            screenshot: The current screenshot image.
            engine_params: Optional engine params override (for separate judge model).

        Returns:
            None if verification is disabled.
            A dict with keys:
                - check_passed: bool
                - details: str
                - failure_summary: str
        """
        if not self.enable_verify:
            return None

        try:
            verify_engine_params = engine_params or self.engine_params
            if self._verify_agent is None:
                self._verify_agent = self._create_agent(
                    PROCEDURAL_MEMORY.VERIFY_AGENT_SYSTEM_PROMPT,
                    engine_params=verify_engine_params,
                )
            else:
                self._verify_agent.reset()

            # Extract Screenshot Analysis and Next Action from plan_text
            analysis_section = ""
            action_section = ""
            in_analysis = False
            in_action = False
            for line in plan_text.splitlines():
                stripped = line.strip().lower()
                if "screenshot analysis" in stripped:
                    in_analysis = True
                    in_action = False
                    continue
                if "next action" in stripped:
                    in_action = True
                    in_analysis = False
                    continue
                if in_analysis:
                    analysis_section += line + "\n"
                if in_action:
                    action_section += line + "\n"

            user_prompt = (
                f"Screenshot Analysis from Planner:\n{analysis_section.strip()}\n\n"
                f"Next Action from Planner:\n{action_section.strip()}"
            )

            self._verify_agent.add_message(
                text_content=user_prompt,
                image_content=screenshot,
                role="user",
            )

            raw_response = self._verify_agent.get_response(temperature=0.0)

            result = json.loads(raw_response.strip())
            return {
                "check_passed": bool(result.get("check_passed", False)),
                "details": str(result.get("details", "")),
                "failure_summary": str(result.get("failure_summary", "")),
            }

        except Exception as e:
            logger.error(f"PLAN VERIFICATION FAILED: {e}")
            return None

    def generate_next_action(self, instruction: str, obs: Dict) -> Tuple[Dict, List]:
        self.grounding_agent.assign_screenshot(obs)
        self.grounding_agent.set_task_instruction(instruction)
        model_obs = self.grounding_agent.get_model_observation()

        # ── Build base generator_message (Reflection + code agent result) ──────
        base_generator_message = (
            ""
            if self.turn_count > 0
            else "The initial screen is provided. No action has been taken yet."
        )
        if self.turn_count == 0:
            prompt_with_instructions = self.generator_agent.system_prompt.replace(
                "TASK_DESCRIPTION", instruction
            )
            self.generator_agent.add_system_prompt(prompt_with_instructions)

        # Reflection
        reflection, reflection_thoughts = self._generate_reflection(instruction, obs)
        if reflection:
            base_generator_message += f"REFLECTION: {reflection}\n"
        base_generator_message += (
            f"\nCurrent Text Buffer = [{','.join(self.grounding_agent.notes)}]\n"
        )

        # Code agent result
        code_agent_output = None
        if (
            hasattr(self.grounding_agent, "last_code_agent_result")
            and self.grounding_agent.last_code_agent_result is not None
        ):
            code_result = self.grounding_agent.last_code_agent_result
            code_agent_output = code_result
            base_generator_message += f"\nCODE AGENT RESULT:\n"
            base_generator_message += f"Task/Subtask Instruction: {code_result['task_instruction']}\n"
            base_generator_message += f"Steps Completed: {code_result['steps_executed']}\n"
            base_generator_message += f"Max Steps: {code_result['budget']}\n"
            base_generator_message += f"Completion Reason: {code_result['completion_reason']}\n"
            base_generator_message += f"Summary: {code_result['summary']}\n"
            if code_result["execution_history"]:
                base_generator_message += "Execution History:\n"
                for i, step in enumerate(code_result["execution_history"]):
                    action = step["action"]
                    if "```python" in action:
                        code_start = action.find("```python") + 9
                        code_end = action.find("```", code_start)
                        python_code = action[code_start:code_end].strip() if code_end != -1 else action
                        base_generator_message += f"Step {i+1}: ```python\n{python_code}\n```\n"
                    elif "```bash" in action:
                        code_start = action.find("```bash") + 7
                        code_end = action.find("```", code_start)
                        bash_code = action[code_start:code_end].strip() if code_end != -1 else action
                        base_generator_message += f"Step {i+1}: ```bash\n{bash_code}\n```\n"
                    else:
                        base_generator_message += f"Step {i+1}: {action}\n"

            try:
                import os
                from datetime import datetime
                logs_dir = "logs"
                if not os.path.exists(logs_dir):
                    os.makedirs(logs_dir)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"logs/code_agent_result_step_{self.turn_count + 1}_{timestamp}.txt"
                with open(filename, "w") as f:
                    f.write(f"CODE AGENT RESULT - Step {self.turn_count + 1}\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Task/Subtask Instruction: {code_result['task_instruction']}\n")
                    f.write(f"Steps Completed: {code_result['steps_executed']}\n")
                    f.write(f"Max Steps: {code_result['budget']}\n")
                    f.write(f"Completion Reason: {code_result['completion_reason']}\n")
                    f.write(f"Summary: {code_result['summary']}\n")
                    if code_result["execution_history"]:
                        f.write("\nExecution History:\n")
                        for i, step in enumerate(code_result["execution_history"]):
                            f.write(f"\nStep {i+1}:\n")
                            f.write(f"Action: {step['action']}\n")
                            if "thoughts" in step:
                                f.write(f"Thoughts: {step['thoughts']}\n")
                logger.info(f"Code agent result saved to: {filename}")
            except Exception as e:
                logger.error(f"Failed to save code agent result to file: {e}")

            self.grounding_agent.last_code_agent_result = None

        # ── Reflection Case 1 → step_retrieval (pre-loop hint injection) ───────────
        # After Reflection, if Case 1 → call step_retrieval BEFORE entering the loop.
        # The retrieved hint (if hit) gets injected into the first planner attempt.
        hint_block_for_planner: Optional[str] = None
        last_step_retrieval_meta = None
        if reflection and "Case 1" in reflection and self.step_retrieval:
            hit, hint, step_data = _do_step_retrieval(
                obs=obs,
                threshold=self.step_retrieval_threshold,
                step_retrieval=self.step_retrieval,
                screenshot_explainer=self._screenshot_explainer,
                summarizer=self._step_retrieval_summarizer,
                engine_params=self.engine_params,
            )
            if hit:
                # Initialize explainer/summarizer for future calls
                if self._screenshot_explainer is None:
                    self._screenshot_explainer = LMMAgent(
                        engine_params=self.engine_params,
                        system_prompt=(
                            "You are a UI state describer for step retrieval.\n"
                            "Given a screenshot of a desktop app, describe the current screen state.\n"
                            "Focus on: active app/window, page/view name, visible UI elements.\n"
                            "Be concise but specific. Output plain text only."
                        ),
                    )
                if self._step_retrieval_summarizer is None:
                    self._step_retrieval_summarizer = create_step_retrieval_summarizer(
                        self.engine_params
                    )
                # Store hint → will be injected into first planner attempt
                hint_block_for_planner = hint
                last_step_retrieval_meta = {
                    "report_path": step_data.get("report_path"),
                    "task_id": step_data.get("task_id"),
                    "step_id": step_data.get("step_id"),
                }
                logger.info(
                    "REFLECTION CASE 1: step_retrieval hit → hint will be injected to planner."
                )
            else:
                logger.info(
                    "REFLECTION CASE 1: step_retrieval miss → planner proceeds without hint."
                )
            # hit=True or hit=False → in both cases we fall through to the loop below
            # (the planner will be invoked with or without the hint)

        # ── Planner + Verify retry loop (max 3 rounds) ─────────────────────────
        max_retries = 3
        attempt = 0
        last_verify_failure: Optional[Dict] = None

        while attempt <= max_retries:
            attempt += 1

            # Build message for this attempt:
            # - base: Reflection + code agent result + (optional) step_retrieval hint
            # - retry: verify failure feedback
            if attempt == 1:
                msg = base_generator_message
                if hint_block_for_planner:
                    msg += (
                        "\nSTEP RETRIEVAL REFERENCE (for planning agent only).\n"
                        "IMPORTANT: Use it silently for planning, but DO NOT quote/copy it in your final plan output.\n"
                        + hint_block_for_planner
                    )
                    hint_block_for_planner = None  # consumed
            else:
                msg = (
                    f"\n[PLAN VERIFICATION FAILED — attempt {attempt}/{max_retries + 1}]\n"
                    f"Failure reason: {last_verify_failure.get('failure_summary', '')}\n"
                    f"Details: {last_verify_failure.get('details', '')}\n"
                    f"IMPORTANT: Re-plan the next action to address the failure above. "
                    f"Do NOT repeat the same action."
                )

            self.generator_agent.add_message(
                msg, image_content=model_obs["screenshot"], role="user"
            )

            # ── Planner ───────────────────────────────────────────────────────
            plan = call_llm_formatted(
                self.generator_agent,
                [SINGLE_ACTION_FORMATTER,
                 partial(CODE_VALID_FORMATTER, self.grounding_agent, obs)],
                temperature=self.temperature,
                use_thinking=self.use_thinking,
            )
            print("-" * 20)
            print(f"****Generator message (attempt={attempt}):\n", msg)
            print("****Raw plan response:\n", plan)
            print("-" * 20)
            plan = _strip_step_retrieval_reference_from_text(plan)
            self.worker_history.append(plan)
            self.generator_agent.add_message(plan, role="assistant")
            logger.info("PLAN:\n %s", plan)

            # Grounding: extract exec_code
            plan_code = parse_code_from_string(plan)
            try:
                assert plan_code, "Plan code should not be empty"
                exec_code = create_pyautogui_code(self.grounding_agent, plan_code, obs)
            except Exception as e:
                logger.error(
                    f"Could not evaluate the following plan code:\n{plan_code}\nError: {e}"
                )
                exec_code = self.grounding_agent.wait(1.333)

            # ── Verify ────────────────────────────────────────────────────────
            verify_result = self._verify_plan(
                plan, model_obs["screenshot"], engine_params=self.verify_engine_params
            )
            if verify_result is not None:
                logger.info(
                    "PLAN VERIFY attempt %d/%d: check_passed=%s  details=%s",
                    attempt, max_retries + 1,
                    verify_result.get("check_passed"),
                    verify_result.get("details"),
                )

            if verify_result is None or verify_result.get("check_passed", True):
                # Verify passed → execute
                break

            # ── Verify failed → step_retrieval ────────────────────────────────
            last_verify_failure = verify_result
            if self.step_retrieval:
                hit, hint, step_data = _do_step_retrieval(
                    obs=obs,
                    threshold=self.step_retrieval_threshold,
                    step_retrieval=self.step_retrieval,
                    screenshot_explainer=self._screenshot_explainer,
                    summarizer=self._step_retrieval_summarizer,
                    engine_params=self.engine_params,
                )
                if hit:
                    # Update instance-level explainer/summarizer
                    if self._screenshot_explainer is None:
                        self._screenshot_explainer = LMMAgent(
                            engine_params=self.engine_params,
                            system_prompt=(
                                "You are a UI state describer for step retrieval.\n"
                                "Given a screenshot of a desktop app, describe the current screen state.\n"
                                "Focus on: active app/window, page/view name, visible UI elements.\n"
                                "Be concise but specific. Output plain text only."
                            ),
                        )
                    if self._step_retrieval_summarizer is None:
                        self._step_retrieval_summarizer = create_step_retrieval_summarizer(
                            self.engine_params
                        )
                    # Store hint for next planner attempt
                    hint_block_for_planner = hint
                    last_step_retrieval_meta = {
                        "report_path": step_data.get("report_path"),
                        "task_id": step_data.get("task_id"),
                        "step_id": step_data.get("step_id"),
                    }
                    logger.info(
                        "VERIFY FAILED: step_retrieval hit → injecting hint, re-entering planner."
                    )
                    continue  # stay in while loop, next iteration will use the hint
                # hit=False: no hint, stay in while loop to re-plan
                logger.info(
                    "VERIFY FAILED: step_retrieval miss → re-entering planner without hint."
                )
            # No step_retrieval or miss: re-enter planner with only verify feedback
            logger.info("VERIFY FAILED: re-entering planner with failure feedback only.")

        # ── Execute ─────────────────────────────────────────────────────────
        executor_info = {
            "plan": plan,
            "plan_code": plan_code,
            "exec_code": exec_code,
            "step_retrieval_used": bool(last_step_retrieval_meta),
            "step_retrieval_meta": last_step_retrieval_meta,
            "reflection": reflection,
            "reflection_thoughts": reflection_thoughts,
            "code_agent_output": code_agent_output,
            "verify_passed": last_verify_failure is None,
            "verify_attempts": attempt,
            "verify_failure": last_verify_failure,
        }
        self.turn_count += 1
        self.screenshot_inputs.append(obs["screenshot"])
        self.flush_messages()
        return executor_info, [exec_code]
