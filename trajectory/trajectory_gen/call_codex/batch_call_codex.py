import os
import sys
import subprocess
import time
import select
from datetime import datetime
from typing import List


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



def run_call_codex(conversation_folder: str) -> subprocess.CompletedProcess:
    """Invoke call_codex.py on a single conversation folder, allowing skip."""
    script_path = os.path.join(os.path.dirname(__file__), "call_codex.py")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"call_codex.py not found at: {script_path}")

    cmd = [sys.executable, script_path, conversation_folder]
    timeout_seconds = 40 * 60  # 40 minutes per folder
    start = time.time()

    # Let call_codex.py run non-interactively; we keep stdin for our own key handling.
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        text=True,
    )

    while True:
        # Check if the process has finished.
        returncode = process.poll()
        if returncode is not None:
            return subprocess.CompletedProcess(cmd, returncode)

        # Enforce overall timeout.
        elapsed = time.time() - start
        if elapsed > timeout_seconds:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_seconds)

        # Check for user input to skip the current task.
        try:
            readable, _, _ = select.select([sys.stdin], [], [], 0.5)
        except (OSError, ValueError):
            # stdin not selectable (e.g. not a TTY); just sleep briefly.
            time.sleep(0.5)
            continue

        if readable:
            line = sys.stdin.readline()
            if line.strip().lower() == "s":
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
        # If report.json is missing or unreadable, skip creating sign.txt
        return

    timestamp = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    sign_path = os.path.join(conversation_folder, "sign.txt")

    try:
        with open(sign_path, "w", encoding="utf-8") as f:
            f.write(f"report.json last modified at: {timestamp}\n")
    except OSError as e:
        print(
            f"Warning: could not write sign.txt in '{conversation_folder}': {e}",
            file=sys.stderr,
        )


def main() -> None:
    # Get root path from CLI argument or prompt
    if len(sys.argv) > 1:
        root_path = sys.argv[1].strip()
    else:
        root_path = input(
            "Please input the root folder path (containing conversation subfolders, e.g. /home/user/test): "
        ).strip()

    if not root_path:
        print("No path provided, exiting.", file=sys.stderr)
        sys.exit(1)

    root_path = os.path.abspath(os.path.expanduser(root_path))

    if not os.path.isdir(root_path):
        print(f"Provided path is not a directory: {root_path}", file=sys.stderr)
        sys.exit(1)

    subfolders = find_subfolders(root_path)
    # Only keep subfolders that contain a report.json
    all_conversation_folders = [
        folder
        for folder in subfolders
        if os.path.isfile(os.path.join(folder, "report.json"))
    ]

    if not all_conversation_folders:
        print(f"No subfolders with report.json found under: {root_path}")
        return

    # Skip subfolders that already have a sign.txt (assumed successfully processed)
    conversation_folders = [
        folder
        for folder in all_conversation_folders
        if not os.path.isfile(os.path.join(folder, "sign.txt"))
    ]

    if not conversation_folders:
        print(
            f"All {len(all_conversation_folders)} subfolders under {root_path} "
            "already have sign.txt. Nothing to do."
        )
        return

    total = len(conversation_folders)
    successes: List[str] = []
    failures: List[str] = []
    skipped_by_user: List[str] = []

    skipped = len(all_conversation_folders) - total
    print(
        f"Found {len(all_conversation_folders)} subfolders with report.json under {root_path}."
    )
    if skipped:
        print(f"{skipped} subfolders already have sign.txt and will be skipped.")
    print(
        "While a task is running, type 's' and press Enter "
        "to skip the current subfolder."
    )

    for idx, folder in enumerate(conversation_folders, start=1):
        rel_name = os.path.basename(folder.rstrip(os.sep)) or folder

        print(
            f"[{idx}/{total}] Processing '{rel_name}' ... ",
            end="",
            flush=True,
        )

        try:
            result = run_call_codex(folder)
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            print(
                "  call_codex.py exceeded 40 minutes; marked as failure.",
                file=sys.stderr,
            )
            failures.append(folder)
            continue
        except UserSkip:
            print("SKIP")
            print(
                "  Skipped current subfolder by user request.",
                file=sys.stderr,
            )
            skipped_by_user.append(folder)
            continue
        except Exception as e:
            print("FAIL")
            print(f"  Error invoking call_codex.py: {e}", file=sys.stderr)
            failures.append(folder)
            continue

        if result.returncode == 0:
            print("OK")
            write_sign_file(folder)
            successes.append(folder)
        else:
            print("FAIL")
            print(
                f"  call_codex.py exited with status {result.returncode}.",
                file=sys.stderr,
            )
            failures.append(folder)

    # Final report
    print("\n=== Final Report ===")
    print(f"Root folder: {root_path}")
    print(f"Total subfolders: {total}")
    print(f"Succeeded: {len(successes)}")
    print(f"Failed: {len(failures)}")
    if skipped_by_user:
        print(f"Skipped by user: {len(skipped_by_user)}")

    if failures:
        print("\nFailed folders:")
        for folder in failures:
            print(f"- {folder}")

    if skipped_by_user:
        print("\nSkipped folders:")
        for folder in skipped_by_user:
            print(f"- {folder}")


if __name__ == "__main__":
    main()
