import argparse
import io
import logging
import os
import platform
import pyautogui
import signal
import sys
import time
from pathlib import Path
from typing import Dict

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str in sys.path:
    sys.path.remove(root_str)
sys.path.insert(0, root_str)

from gui_agents.agents.agent_s import AgentS3
from gui_agents.agents.grounding import OSWorldACI
from gui_agents.utils.local_env import LocalEnv


current_platform = platform.system().lower()
paused = False


def get_char():
    """Get a single character from stdin without pressing Enter."""
    try:
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
            import msvcrt

            return msvcrt.getch().decode("utf-8", errors="ignore")
    except Exception:
        return input()


def signal_handler(signum, frame):
    """Handle Ctrl+C signal for debugging during agent execution."""
    global paused

    if not paused:
        print("\n\nAmbler Workflow Paused")
        print("=" * 50)
        print("Options:")
        print("  Press Ctrl+C again to quit")
        print("  Press Esc to resume workflow")
        print("=" * 50)

        paused = True

        while paused:
            try:
                print("\n[PAUSED] Waiting for input... ", end="", flush=True)
                char = get_char()

                if ord(char) == 3:
                    print("\n\nExiting Ambler...")
                    sys.exit(0)
                elif ord(char) == 27:
                    print("\n\nResuming Ambler workflow...")
                    paused = False
                    break
                else:
                    print(f"\n   Unknown command: '{char}' (ord: {ord(char)})")
            except KeyboardInterrupt:
                print("\n\nExiting Ambler...")
                sys.exit(0)
    else:
        print("\n\nExiting Ambler...")
        sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

datetime_str = time.strftime("%Y%m%d@%H%M%S")
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

file_handler = logging.FileHandler(
    os.path.join("logs", f"normal-{datetime_str}.log"), encoding="utf-8"
)
debug_handler = logging.FileHandler(
    os.path.join("logs", f"debug-{datetime_str}.log"), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)
sdebug_handler = logging.FileHandler(
    os.path.join("logs", f"sdebug-{datetime_str}.log"), encoding="utf-8"
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


def scale_screen_dimensions(width: int, height: int, max_dim_size: int):
    scale_factor = min(max_dim_size / width, max_dim_size / height, 1)
    safe_width = int(width * scale_factor)
    safe_height = int(height * scale_factor)
    return safe_width, safe_height


def run_agent(agent, instruction: str, scaled_width: int, scaled_height: int):
    global paused
    obs = {}
    traj = "Task:\n" + instruction
    subtask_traj = ""
    for step in range(15):
        while paused:
            time.sleep(0.1)

        screenshot = pyautogui.screenshot()
        screenshot = screenshot.resize((scaled_width, scaled_height), Image.LANCZOS)

        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG", optimize=True, compress_level=9)
        obs["screenshot"] = buffered.getvalue()

        while paused:
            time.sleep(0.1)

        print(f"\nStep {step + 1}/15: Getting next action from agent...")
        info, code = agent.predict(instruction=instruction, observation=obs)

        if isinstance(info, dict) and info.get("plan"):
            print("Agent reply:")
            print(info["plan"])

        if not code:
            print("Failed to get valid action from agent. Stopping...")
            break

        if "done" in code[0].lower() or "fail" in code[0].lower():
            break

        if "next" in code[0].lower():
            continue

        if "wait" in code[0].lower():
            print("Agent requested wait...")
            time.sleep(5)
            continue

        time.sleep(1.0)
        print("EXECUTING CODE:", code[0])

        while paused:
            time.sleep(0.1)

        exec(code[0])
        time.sleep(1.0)

        if "reflection" in info and "plan" in info:
            traj += (
                "\n\nReflection:\n"
                + str(info["reflection"])
                + "\n\n----------------------\n\nPlan:\n"
                + info["plan"]
            )



def main():
    parser = argparse.ArgumentParser(description="Run AgentS3 with specified model.")
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-5-2025-08-07")
    parser.add_argument("--model_url", type=str, default="")
    parser.add_argument("--model_api_key", type=str, default="")
    parser.add_argument("--model_temperature", type=float, default=None)

    parser.add_argument("--ground_provider", type=str, required=True)
    parser.add_argument("--ground_url", type=str, required=True)
    parser.add_argument("--ground_api_key", type=str, default="")
    parser.add_argument("--ground_model", type=str, required=True)
    parser.add_argument("--grounding_width", type=int, required=True)
    parser.add_argument("--grounding_height", type=int, required=True)

    parser.add_argument("--max_trajectory_length", type=int, default=8)
    parser.add_argument("--enable_reflection", action="store_true", default=True)
    parser.add_argument("--enable_local_env", action="store_true", default=False)
    parser.add_argument("--enable_cu_retrieval", action="store_true", default=False)

    args = parser.parse_args()

    screen_width, screen_height = pyautogui.size()
    scaled_width, scaled_height = scale_screen_dimensions(
        screen_width, screen_height, max_dim_size=1280
    )

    engine_params: Dict = {
        "engine_type": args.provider,
        "model": args.model,
        "base_url": args.model_url,
        "api_key": args.model_api_key,
        "temperature": getattr(args, "model_temperature", None),
    }

    engine_params_for_grounding: Dict = {
        "engine_type": args.ground_provider,
        "model": args.ground_model,
        "base_url": args.ground_url,
        "api_key": args.ground_api_key,
        "grounding_width": args.grounding_width,
        "grounding_height": args.grounding_height,
    }

    local_env = None
    if args.enable_local_env:
        print("WARNING: Local coding environment enabled. This will execute arbitrary code locally!")
        local_env = LocalEnv()

    grounding_agent = OSWorldACI(
        env=local_env,
        platform=current_platform,
        engine_params_for_generation=engine_params,
        engine_params_for_grounding=engine_params_for_grounding,
        width=screen_width,
        height=screen_height,
    )

    agent = AgentS3(
        engine_params,
        grounding_agent,
        platform=current_platform,
        max_trajectory_length=args.max_trajectory_length,
        enable_reflection=args.enable_reflection,
        enable_cu_retrieval=args.enable_cu_retrieval,
    )

    while True:
        query = input("Query: ")
        agent.reset()
        run_agent(agent, query, scaled_width, scaled_height)
        response = input("Would you like to provide another query? (y/n): ")
        if response.lower() != "y":
            break


if __name__ == "__main__":
    main()
