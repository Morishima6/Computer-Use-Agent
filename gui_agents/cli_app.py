import argparse
import base64
import datetime
import io
import logging
import os
import platform
import pyautogui
import signal
import sys
import time
import json
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # Ambler-Agent
root_str = str(ROOT)
if root_str in sys.path:
    sys.path.remove(root_str)
sys.path.insert(0, root_str)
    
from gui_agents.agents.grounding import OSWorldACI
from gui_agents.agents.agent_s import AgentS3
from gui_agents.utils.local_env import LocalEnv
from trajectory.retrieval.task_retrieval import *
from trajectory.retrieval.step_retrieval import *


current_platform = platform.system().lower()

# Global flag to track pause state for debugging
paused = False

_TASK_MATCHER = None
_TASK_MATCHER_INIT_ATTEMPTED = False

# NOTE: task retrieval needs a one-time index load (embeddings), step retrieval doesn't.
def _ensure_task_matcher() -> None:
    global _TASK_MATCHER, _TASK_MATCHER_INIT_ATTEMPTED
    if _TASK_MATCHER_INIT_ATTEMPTED:
        return

    _TASK_MATCHER_INIT_ATTEMPTED = True
    task_matcher = TaskMatcher()
    repo_root = Path(__file__).resolve().parents[1]
    trace_root = repo_root / "trajectory" / "trajectory_base"
    if task_matcher.load_data(str(trace_root)):
        _TASK_MATCHER = task_matcher

# 终端输入控制，实现按一个键无需回车
def get_char():
    """Get a single character from stdin without pressing Enter"""
    try:
        # Import termios and tty on Unix3-like systems
        if platform.system() in ["Darwin", "Linux"]:
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch
        else:
            # Windows fallback
            import msvcrt

            return msvcrt.getch().decode("utf-8", errors="ignore")
    except:
        return input()  # Fallback for non-terminal environments


def signal_handler(signum, frame):
    """Handle Ctrl+C signal for debugging during agent execution"""
    global paused

    if not paused:
        print("\n\n🔸 Ambler Workflow Paused 🔸")
        print("=" * 50)
        print("Options:")
        print("  • Press Ctrl+C again to quit")
        print("  • Press Esc to resume workflow")
        print("=" * 50)

        paused = True

        while paused:
            try:
                print("\n[PAUSED] Waiting for input... ", end="", flush=True)
                char = get_char()

                if ord(char) == 3:  # Ctrl+C
                    print("\n\n🛑 Exiting Ambler...")
                    sys.exit(0)
                elif ord(char) == 27:  # Esc
                    print("\n\n▶️  Resuming Ambler workflow...")
                    paused = False
                    break
                else:
                    print(f"\n   Unknown command: '{char}' (ord: {ord(char)})")

            except KeyboardInterrupt:
                print("\n\n🛑 Exiting Ambler...")
                sys.exit(0)
    else:
        # Already paused, second Ctrl+C means quit
        print("\n\n🛑 Exiting Ambler...")
        sys.exit(0)


# Set up signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

file_handler = logging.FileHandler(
    os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8"
)
debug_handler = logging.FileHandler(
    os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)
sdebug_handler = logging.FileHandler(
    os.path.join("logs", "sdebug-{:}.log".format(datetime_str)), encoding="utf-8"
)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)
sdebug_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
sdebug_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("desktopenv"))
sdebug_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
logger.addHandler(sdebug_handler)

platform_os = platform.system()

# NOTE: 未来可选的权限控制，目前用于本文件的 run_agent 函数中，用于在执行代码前显示
def show_permission_dialog(code: str, action_description: str):
    """Show a platform-specific permission dialog and return True if approved."""
    if platform.system() == "Darwin":
        result = os.system(
            f'osascript -e \'display dialog "Do you want to execute this action?\n\n{code} which will try to {action_description}" with title "Action Permission" buttons {{"Cancel", "OK"}} default button "OK" cancel button "Cancel"\''
        )
        return result == 0
    elif platform.system() == "Linux":
        result = os.system(
            f'zenity --question --title="Action Permission" --text="Do you want to execute this action?\n\n{code}" --width=400 --height=200'
        )
        return result == 0
    return False


def scale_screen_dimensions(width: int, height: int, max_dim_size: int):
    scale_factor = min(max_dim_size / width, max_dim_size / height, 1)
    safe_width = int(width * scale_factor)
    safe_height = int(height * scale_factor)
    return safe_width, safe_height


# 🪴 NOTE
def example_task_retrieval(query: str) -> str:
    """
    输入：query (str) - agent 的查询
    输出：str - 最相似的外部知识，可以直接拼接到prompt当中
    """
    global _TASK_MATCHER
    _ensure_task_matcher()

    if _TASK_MATCHER is not None:
        task, _score = _TASK_MATCHER.find_task(
            query, threshold=0.7, runtime_os=current_platform
        )
        if task:
            report_path = task.get("report_path")
            if report_path and os.path.exists(report_path):
                try:
                    with open(report_path, "r", encoding="utf-8") as f:
                        report = json.load(f)
                    return build_prompt_from_trace(report).strip()
                except Exception:
                    return ""

    return ""


# 🪴 Note
def example_step_retrieval(obs: dict, k: float) -> dict:
    """
    输入：obs (dict) - 包含 screenshot 等观察结果的字典
    输出：dict - 检索到的 step 元数据（包含 similarity/full_step_data 等）
    """

    step_data = find_step_by_similarity(obs["screenshot_explanation"], k=k)
    return step_data



def run_agent(agent, instruction: str, scaled_width: int, scaled_height: int):
    global paused
    obs = {}
    traj = "Task:\n" + instruction
    subtask_traj = ""
    for step in range(15):
        # Check if we're in paused state and wait
        while paused:
            time.sleep(0.1)
        # Get screen shot using pyautogui
        screenshot = pyautogui.screenshot()
        screenshot = screenshot.resize((scaled_width, scaled_height), Image.LANCZOS)

        # Save the screenshot to a BytesIO object
        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG", optimize=True, compress_level=9)

        # Get the byte value of the screenshot
        screenshot_bytes = buffered.getvalue()
        # Convert to base64 string.
        obs["screenshot"] = screenshot_bytes
        
        # Send screenshot to Desktop backend (base64 encoded)
        # 注释掉以避免在日志中输出过长的SCREENSHOT_DATA
        # screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        # print(f"SCREENSHOT_DATA:{screenshot_b64}")
        # sys.stdout.flush()

        # Check again for pause state before prediction
        while paused:
            time.sleep(0.1)

        print(f"\n🔄 Step {step + 1}/15: Getting next action from agent...")

        # Get next action code from the agent
        info, code = agent.predict(instruction=instruction, observation=obs)

        # Log agent's textual reply for this step (the generated plan)
        if isinstance(info, dict) and "plan" in info and info["plan"]:
            print("💬 Agent reply:")
            print(info["plan"])

        # Check if code is None or empty (formatting failed)
        if code is None or len(code) == 0:
            print("❌ Failed to get valid action from agent. Stopping...")
            break

        if "done" in code[0].lower() or "fail" in code[0].lower():
            if platform.system() == "Darwin":
                os.system(
                    f'osascript -e \'display dialog "Task Completed" with title "OpenACI Agent" buttons "OK" default button "OK"\''
                )
            elif platform.system() == "Linux":
                os.system(
                    f'zenity --info --title="OpenACI Agent" --text="Task Completed" --width=200 --height=100'
                )

            break

        if "next" in code[0].lower():
            continue

        if "wait" in code[0].lower():
            print("⏳ Agent requested wait...")
            time.sleep(5)
            continue

        else:
            time.sleep(1.0)
            print("EXECUTING CODE:", code[0])

            # Check for pause state before execution
            while paused:
                time.sleep(0.1)

            # Ask for permission before executing
            exec(code[0])
            time.sleep(1.0)

            # Update task and subtask trajectories
            if "reflection" in info and "plan" in info:
                traj += (
                    "\n\nReflection:\n"
                    + str(info["reflection"])
                    + "\n\n----------------------\n\nPlan:\n"
                    + info["plan"]
                )


def main():
    parser = argparse.ArgumentParser(description="Run AgentS3 with specified model.")
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="Specify the provider to use (e.g., openai, anthropic, etc.)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-2025-08-07",
        help="Specify the model to use (e.g., gpt-5-2025-08-07)",
    )
    parser.add_argument(
        "--model_url",
        type=str,
        default="",
        help="The URL of the main generation model API.",
    )
    parser.add_argument(
        "--model_api_key",
        type=str,
        default="",
        help="The API key of the main generation model.",
    )
    parser.add_argument(
        "--model_temperature",
        type=float,
        default=None,
        help="Temperature to fix the generation model at (e.g. o3 can only be run with 1.0)",
    )

    # Grounding model config: Self-hosted endpoint based (required)
    parser.add_argument(
        "--ground_provider",
        type=str,
        required=True,
        help="The provider for the grounding model",
    )
    parser.add_argument(
        "--ground_url",
        type=str,
        required=True,
        help="The URL of the grounding model",
    )
    parser.add_argument(
        "--ground_api_key",
        type=str,
        default="",
        help="The API key of the grounding model.",
    )
    parser.add_argument(
        "--ground_model",
        type=str,
        required=True,
        help="The model name for the grounding model",
    )
    parser.add_argument(
        "--grounding_width",
        type=int,
        required=True,
        help="Width of screenshot image after processor rescaling",
    )
    parser.add_argument(
        "--grounding_height",
        type=int,
        required=True,
        help="Height of screenshot image after processor rescaling",
    )

    # specific arguments
    parser.add_argument(
        "--max_trajectory_length",
        type=int,
        default=8,
        help="Maximum number of image turns to keep in trajectory",
    )
    parser.add_argument(
        "--enable_reflection",
        action="store_true",
        default=True,
        help="Enable reflection agent to assist the worker agent",
    )
    parser.add_argument(
        "--enable_local_env",
        action="store_true",
        default=False,
        help="Enable local coding environment for code execution (WARNING: Executes arbitrary code locally)",
    )
    
    
    # NOTE: Retrieval Control Arguments
    parser.add_argument(
        "--enable_task_retrieval",
        action="store_true",
        default=False,
        help="Enable task_retrieval function (uses example function if not provided externally)",
    )
    parser.add_argument(
        "--enable_step_retrieval",
        action="store_true",
        default=False,
        help="Enable step_retrieval function (uses example function if not provided externally)",
    )
    parser.add_argument(
        "--step_retrieval_threshold",
        type=float,
        default=0.9,
        help="Threshold K value for step_retrieval score to directly execute action (default: 0.9)",
    )

    args = parser.parse_args()

    # Re-scales screenshot size to ensure it fits in UI-TARS context limit
    screen_width, screen_height = pyautogui.size()
    scaled_width, scaled_height = scale_screen_dimensions(
        screen_width, screen_height, max_dim_size=1280
    )
    print(f"************************************************Original screen size: {screen_width}x{screen_height}")
    print(f"************************************************Scaled screen size: {scaled_width}x{scaled_height}")
    # Load the general engine params
    engine_params = {
        "engine_type": args.provider,
        "model": args.model,
        "base_url": args.model_url,
        "api_key": args.model_api_key,
        "temperature": getattr(args, "model_temperature", None),
    }

    # Load the grounding engine from a custom endpoint
    engine_params_for_grounding = {
        "engine_type": args.ground_provider,
        "model": args.ground_model,
        "base_url": args.ground_url,
        "api_key": args.ground_api_key,
        "grounding_width": args.grounding_width,
        "grounding_height": args.grounding_height,
        # "grounding_width": scaled_width,
        # "grounding_height": scaled_height,
    }

    # Initialize environment based on user preference
    local_env = None
    if args.enable_local_env:
        print(
            "⚠️  WARNING: Local coding environment enabled. This will execute arbitrary code locally!"
        )
        local_env = LocalEnv()

    grounding_agent = OSWorldACI(
        env=local_env,
        platform=current_platform,
        engine_params_for_generation=engine_params,
        engine_params_for_grounding=engine_params_for_grounding,
        width=screen_width,
        height=screen_height,
    )



    # NOTE: Retrieval Functions Initialization
    task_retrieval_func = None
    step_retrieval_func = None
    
    if args.enable_task_retrieval:
        task_retrieval_func = example_task_retrieval
    
    if args.enable_step_retrieval:
        step_retrieval_func = example_step_retrieval

    agent = AgentS3(
        engine_params,
        grounding_agent,
        platform=current_platform,
        max_trajectory_length=args.max_trajectory_length,
        enable_reflection=args.enable_reflection,
        task_retrieval=task_retrieval_func,
        step_retrieval=step_retrieval_func,
        step_retrieval_threshold=args.step_retrieval_threshold,
    )

    while True:
        query = input("Query: ")

        agent.reset()

        # Run the agent on your own device
        run_agent(agent, query, scaled_width, scaled_height)

        response = input("Would you like to provide another query? (y/n): ")
        if response.lower() != "y":
            break


if __name__ == "__main__":
    main()
