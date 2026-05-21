import os
import sys
import subprocess
import time
from datetime import datetime
from typing import List

# 修改windows兼容性问题：Windows 走 msvcrt，非 Windows 才走 select
if os.name == "nt":
    import msvcrt
else:
    import select

# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REMOVE_FIRST_STEP_SCRIPT = os.path.join(
    SCRIPT_DIR,
    "trajectory_preprocess",
    "remove_first_step.py"
)
CALL_CODEX_SCRIPT = os.path.join(
    SCRIPT_DIR,
    "call_codex",
    "call_codex.py"
)


def find_subfolders(root: str) -> List[str]:
    """Return a sorted list of immediate subfolders under root."""
    try:
        entries = os.listdir(root)
    except OSError as e:
        print(f"Error reading directory '{root}': {e}", file=sys.stderr)
        return []

    subfolders = [
        os.path.join(root, name)
        for name in entries
        if os.path.isdir(os.path.join(root, name))
    ]
    return sorted(subfolders)


class UserSkip(Exception):
    """Raised when the user requests to skip the current task."""


def _read_skip_request(timeout_seconds: float) -> bool:
    """Return True when the operator requests skipping the current task."""
    if os.name == "nt":
        deadline = time.time() + timeout_seconds
        chars = []

        while time.time() < deadline:
            while msvcrt.kbhit():
                char = msvcrt.getwche()
                if char in ("\r", "\n"):
                    line = "".join(chars).strip().lower()
                    if line:
                        print()
                    return line == "s"
                if char == "\003":
                    raise KeyboardInterrupt
                if char == "\b":
                    if chars:
                        chars.pop()
                    continue
                chars.append(char)

            time.sleep(0.05)

        return False

    try:
        readable, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
    except (OSError, ValueError):
        time.sleep(timeout_seconds)
        return False

    if not readable:
        return False

    line = sys.stdin.readline()
    return line.strip().lower() == "s"


def run_remove_first_step(conversation_folder: str) -> subprocess.CompletedProcess:
    """Invoke remove_first_step.py on a single conversation folder."""
    if not os.path.exists(REMOVE_FIRST_STEP_SCRIPT):
        raise FileNotFoundError(f"remove_first_step.py not found at: {REMOVE_FIRST_STEP_SCRIPT}")

    cmd = [sys.executable, REMOVE_FIRST_STEP_SCRIPT, conversation_folder]
    timeout_seconds = 5 * 60  # 5 minutes for remove_first_step

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    start = time.time()
    while True:
        returncode = process.poll()
        if returncode is not None:
            stdout, stderr = process.communicate()
            result = subprocess.CompletedProcess(cmd, returncode, stdout, stderr)
            return result

        elapsed = time.time() - start
        if elapsed > timeout_seconds:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_seconds)

        if _read_skip_request(0.5):
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise UserSkip()

        time.sleep(0.1)


def run_call_codex(conversation_folder: str) -> subprocess.CompletedProcess:
    """Invoke call_codex.py on a single conversation folder."""
    if not os.path.exists(CALL_CODEX_SCRIPT):
        raise FileNotFoundError(f"call_codex.py not found at: {CALL_CODEX_SCRIPT}")

    cmd = [sys.executable, CALL_CODEX_SCRIPT, conversation_folder]
    timeout_seconds = 40 * 60  # 40 minutes per folder

    start = time.time()

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        text=True,
    )

    while True:
        returncode = process.poll()
        if returncode is not None:
            return subprocess.CompletedProcess(cmd, returncode)

        elapsed = time.time() - start
        if elapsed > timeout_seconds:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_seconds)

        if _read_skip_request(0.5):
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise UserSkip()


def write_sign_file(conversation_folder: str) -> None:
    """Create or update sign.txt with the last modified time of report.json."""
    report_path = os.path.join(conversation_folder, "report.json")
    try:
        mtime = os.path.getmtime(report_path)
    except OSError:
        return

    timestamp = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    sign_path = os.path.join(conversation_folder, "sign.txt")

    try:
        with open(sign_path, "w", encoding="utf-8") as f:
            f.write(f"Pipeline completed at: {timestamp}\n")
            f.write(f"Steps: remove_first_step -> call_codex\n")
    except OSError as e:
        print(
            f"Warning: could not write sign.txt in '{conversation_folder}': {e}",
            file=sys.stderr,
        )


def main() -> None:
    if len(sys.argv) > 1:
        root_path = sys.argv[1].strip()
    else:
        root_path = input(
            "请输入根文件夹路径（包含对话子文件夹）: "
        ).strip()

    if not root_path:
        print("未提供路径，退出。", file=sys.stderr)
        sys.exit(1)

    root_path = os.path.abspath(os.path.expanduser(root_path))

    if not os.path.isdir(root_path):
        print(f"提供的路径不是目录: {root_path}", file=sys.stderr)
        sys.exit(1)

    subfolders = find_subfolders(root_path)

    # 只保留包含 report.json 的子文件夹
    all_conversation_folders = [
        folder
        for folder in subfolders
        if os.path.isfile(os.path.join(folder, "report.json"))
    ]

    if not all_conversation_folders:
        print(f"在 {root_path} 下未找到包含 report.json 的子文件夹")
        return

    # 跳过已有 sign.txt 的子文件夹（认为已成功处理）
    conversation_folders = [
        folder
        for folder in all_conversation_folders
        if not os.path.isfile(os.path.join(folder, "sign.txt"))
    ]

    if not conversation_folders:
        print(
            f"{root_path} 下的 {len(all_conversation_folders)} 个子文件夹"
            "均已有 sign.txt，无需处理。"
        )
        return

    total = len(conversation_folders)
    successes: List[str] = []
    failures: List[str] = []
    skipped_by_user: List[str] = []

    skipped = len(all_conversation_folders) - total
    print(f"在 {root_path} 下找到 {len(all_conversation_folders)} 个包含 report.json 的子文件夹。")
    if skipped:
        print(f"{skipped} 个子文件夹已有 sign.txt，将跳过。")
    print(
        "运行任务时，输入 's' 并回车可跳过当前子文件夹。"
    )
    print("=" * 60)

    for idx, folder in enumerate(conversation_folders, start=1):
        rel_name = os.path.basename(folder.rstrip(os.sep)) or folder

        print(
            f"[{idx}/{total}] 处理 '{rel_name}' ...",
            end="",
            flush=True,
        )

        step1_ok = False
        step2_ok = False

        # 步骤1: 运行 remove_first_step.py
        try:
            result = run_remove_first_step(folder)
        except subprocess.TimeoutExpired:
            print(" [remove_first_step TIMEOUT]")
            print(
                "  remove_first_step.py 超过 5 分钟，标记为失败。",
                file=sys.stderr,
            )
            failures.append(folder)
            continue
        except UserSkip:
            print(" [用户跳过]")
            print("  用户请求跳过当前子文件夹。", file=sys.stderr)
            skipped_by_user.append(folder)
            continue
        except Exception as e:
            print(f" [remove_first_step 错误]")
            print(f"  调用 remove_first_step.py 出错: {e}", file=sys.stderr)
            failures.append(folder)
            continue

        if result.returncode != 0:
            print(f" [remove_first_step 失败]")
            print(
                f"  remove_first_step.py 退出状态 {result.returncode}。",
                file=sys.stderr,
            )
            failures.append(folder)
            continue

        step1_ok = True

        # 步骤2: 运行 call_codex.py
        try:
            result = run_call_codex(folder)
        except subprocess.TimeoutExpired:
            print(" [call_codex TIMEOUT]")
            print(
                "  call_codex.py 超过 40 分钟，标记为失败。",
                file=sys.stderr,
            )
            failures.append(folder)
            continue
        except UserSkip:
            print(" [用户跳过]")
            print("  用户请求跳过当前子文件夹。", file=sys.stderr)
            skipped_by_user.append(folder)
            continue
        except Exception as e:
            print(f" [call_codex 错误]")
            print(f"  调用 call_codex.py 出错: {e}", file=sys.stderr)
            failures.append(folder)
            continue

        if result.returncode == 0:
            step2_ok = True
        else:
            print(f" [call_codex 失败]")
            print(
                f"  call_codex.py 退出状态 {result.returncode}。",
                file=sys.stderr,
            )
            failures.append(folder)
            continue

        # 两个步骤都成功，写入 sign.txt
        if step1_ok and step2_ok:
            print(" [成功]")
            write_sign_file(folder)
            successes.append(folder)

    # 最终报告
    print("\n" + "=" * 60)
    print("=== 最终报告 ===")
    print(f"根文件夹: {root_path}")
    print(f"总子文件夹数: {total}")
    print(f"成功: {len(successes)}")
    print(f"失败: {len(failures)}")
    if skipped_by_user:
        print(f"用户跳过: {len(skipped_by_user)}")

    if failures:
        print("\n失败的文件夹:")
        for folder in failures:
            print(f"  - {folder}")

    if skipped_by_user:
        print("\n跳过的文件夹:")
        for folder in skipped_by_user:
            print(f"  - {folder}")

    if successes:
        print(f"\n成功处理的文件夹已标记 sign.txt，可通过重新运行脚本跳过。")


if __name__ == "__main__":
    main()
