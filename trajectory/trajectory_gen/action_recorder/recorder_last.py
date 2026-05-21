import atexit
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime

import pyautogui

# 

# 根据操作系统导入不同的模块
if sys.platform == "win32":
    try:
        from pynput import mouse, keyboard as pynput_keyboard
        from pynput.keyboard import Key, Listener as KeyboardListener
        from pynput.mouse import Listener as MouseListener

        PYNUT_AVAILABLE = True
    except Exception as e:
        print(f"警告: pynput 导入失败: {e}")
        print("将使用备用输入监听方法")
        PYNUT_AVAILABLE = False
elif sys.platform == "darwin":  # macOS
    from pynput import mouse
    from pynput.keyboard import Key, Listener as KeyboardListener
    from pynput.mouse import Listener as MouseListener

    PYNUT_AVAILABLE = True
elif sys.platform.startswith("linux"):
    from pynput import mouse
    from pynput.keyboard import Key, Listener as KeyboardListener
    from pynput.mouse import Listener as MouseListener

    PYNUT_AVAILABLE = True
else:
    print(f"不支持的操作系统: {sys.platform}")
    sys.exit(1)

# 配置常量
VIDEO_FPS = 30
VIDEO_FILENAME = "recording.mp4"
VIDEO_WARMUP_SECONDS = 0.5   # 给 ffmpeg 初始化留出的时间，之后再设 video_start_mono_ns


SYSTEM = platform.system()

# ========== 前台窗口标题获取 ==========

APPLE_SCRIPT = r'''
tell application "System Events"
    set frontApp to first process whose frontmost is true
    set appName to name of frontApp
    tell frontApp
        if (count of windows) is 0 then
            return appName
        else
            try
                set winTitle to name of front window
                return appName & " — " & winTitle
            on error
                return appName
            end try
        end if
    end tell
end tell
'''

if SYSTEM == "Windows":
    try:
        import psutil as _psutil  # type: ignore[import]
        import win32gui as _win32gui  # type: ignore[import]
        import win32process as _win32process  # type: ignore[import]
    except Exception:
        _psutil = None
        _win32gui = None
        _win32process = None
else:
    _psutil = None
    _win32gui = None
    _win32process = None


def _parse_xprop_class_and_title(xprop_output):
    """辅助函数：从 xprop 输出中解析 WM_CLASS 和 _NET_WM_NAME。"""
    wm_class = None
    title = None
    for line in xprop_output.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("WM_CLASS"):
            if "=" in line:
                raw = line.split("=", 1)[1].strip()
                parts = [p.strip().strip('"') for p in raw.split(",")]
                if parts:
                    wm_class = parts[0]
        elif "_NET_WM_NAME" in line or "WM_NAME" in line:
            if "=" in line:
                title = line.split("=", 1)[1].strip().strip('"')
    if not wm_class and not title:
        return None, None
    app = wm_class.split(".", 1)[0] if wm_class else "unknown"
    return app, title


def _get_active_app_title_linux():
    """
    在 Linux/X11 下获取当前前台窗口的 “APP — 标题”。

    优先级：
      1) `xprop -root _NET_ACTIVE_WINDOW` + `xprop -id <win> WM_CLASS _NET_WM_NAME`
      2) 若上一步失败，使用鼠标所在窗口：
         `xdotool getmouselocation --shell` + `xprop -id <WINDOW> ...`
      3) 再失败则退回到 `wmctrl -lx` + 活动窗口 ID 匹配

    任一步出错都返回 None，不影响主流程。
    """

    def by_xprop_active():
        try:
            root_result = subprocess.run(
                ["xprop", "-root", "_NET_ACTIVE_WINDOW"],
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception:
            return None

        window_id = None
        for token in root_result.stdout.split():
            if token.startswith("0x"):
                window_id = token.strip().strip(",")
                break
        if not window_id:
            return None

        try:
            win_result = subprocess.run(
                ["xprop", "-id", window_id, "WM_CLASS", "_NET_WM_NAME"],
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception:
            return None

        app, title = _parse_xprop_class_and_title(win_result.stdout)
        if not app and not title:
            return None
        if title:
            return f"{app} — {title}"
        return app

    def by_pointer_window():
        """通过鼠标所在窗口获取标题（需要 xdotool 和 xprop）。"""
        try:
            loc = subprocess.run(
                ["xdotool", "getmouselocation", "--shell"],
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception:
            return None
        window_id = None
        for line in loc.stdout.splitlines():
            line = line.strip()
            if line.startswith("WINDOW="):
                window_id = line.split("=", 1)[1].strip()
                break
        if not window_id:
            return None
        try:
            win_result = subprocess.run(
                ["xprop", "-id", window_id, "WM_CLASS", "_NET_WM_NAME"],
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception:
            return None
        app, title = _parse_xprop_class_and_title(win_result.stdout)
        if not app and not title:
            return None
        if title:
            return f"{app} — {title}"
        return app

    # 1) 尝试通过 _NET_ACTIVE_WINDOW
    title = by_xprop_active()
    if title:
        return title

    # 2) 若失败，尝试鼠标所在窗口
    title = by_pointer_window()
    if title:
        return title

    # 3) 最后退回 wmctrl（使用活动窗口 ID）
    try:
        root_result = subprocess.run(
            ["xprop", "-root", "_NET_ACTIVE_WINDOW"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None

    window_id = None
    for token in root_result.stdout.split():
        if token.startswith("0x"):
            window_id = token.strip().strip(",")
            break
    if not window_id:
        return None

    try:
        active_id = int(window_id, 16)
    except ValueError:
        return None

    try:
        wmctrl_result = subprocess.run(
            ["wmctrl", "-lx"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None

    for line in wmctrl_result.stdout.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        wid_str, _, _, wm_class, title = parts
        try:
            wid = int(wid_str, 16)
        except ValueError:
            continue
        if wid != active_id:
            continue
        app = wm_class.split(".", 1)[0] if wm_class else "unknown"
        return f"{app} — {title}"

    return None


def get_active_app_title():
    """
    返回当前前台活动窗口的 “APP — 窗口名” 文本。
    若无法获取或依赖缺失，返回 None。
    """
    if SYSTEM == "Darwin":
        try:
            result = subprocess.run(
                ["osascript", "-e", APPLE_SCRIPT],
                capture_output=True,
                text=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        title = result.stdout.strip()
        return title or None
    elif SYSTEM == "Windows":
        if _win32gui is None or _win32process is None or _psutil is None:
            return None
        try:
            hwnd = _win32gui.GetForegroundWindow()
            if not hwnd:
                return None
            title = _win32gui.GetWindowText(hwnd)
            if not title:
                return None
            _, pid = _win32process.GetWindowThreadProcessId(hwnd)
            try:
                app_name = _psutil.Process(pid).name()
            except Exception:
                app_name = "Unknown"
            return f"{app_name} — {title}"
        except Exception:
            return None
    else:
        # Linux / 其它：目前按 X11 + wmctrl 处理
        return _get_active_app_title_linux()


class VideoRecorder:
    """基于 ffmpeg 的全屏录制器。Linux 用 x11grab，macOS 用 avfoundation，Windows 用 gdigrab。

    video_start_mono_ns 是在 ffmpeg 预热 (VIDEO_WARMUP_SECONDS) 结束后记录的，
    之后所有 action 的 monotonic_ns 时间戳减去这个值即可得到相对视频的偏移。
    """

    def __init__(self, output_path, fps=VIDEO_FPS):
        self.output_path = output_path
        self.fps = fps
        self.proc = None
        self.start_mono_ns = None
        self.start_wall = None
        self.end_mono_ns = None
        self.end_wall = None
        self.screen_size = None
        self._stderr_drain_thread = None

    def _build_args(self):
        screen_w, screen_h = pyautogui.size()
        self.screen_size = f"{screen_w}x{screen_h}"
        common_tail = [
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-vsync", "cfr",
            self.output_path,
        ]
        if sys.platform.startswith("linux"):
            display = os.environ.get("DISPLAY", ":0")
            return [
                "ffmpeg", "-y",
                "-f", "x11grab",
                "-framerate", str(self.fps),
                "-video_size", self.screen_size,
                "-i", display,
                *common_tail,
            ]
        if sys.platform == "darwin":
            # avfoundation 屏幕索引随设备而异，默认 1:none（screen 1，无音频）。
            return [
                "ffmpeg", "-y",
                "-f", "avfoundation",
                "-framerate", str(self.fps),
                "-capture_cursor", "1",
                "-i", "1:none",
                *common_tail,
            ]
        if sys.platform == "win32":
            return [
                "ffmpeg", "-y",
                "-f", "gdigrab",
                "-framerate", str(self.fps),
                "-i", "desktop",
                *common_tail,
            ]
        raise RuntimeError(f"不支持的操作系统录屏: {sys.platform}")

    def start(self):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("未找到 ffmpeg，请先安装后再启动录制")
        args = self._build_args()
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)) or ".", exist_ok=True)
        print(f"启动录屏: {' '.join(args)}")
        self.proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        # 给 ffmpeg 一点时间初始化 encoder / 抓到第一帧，然后把 start_mono_ns 锚定到
        # 预热之后那一刻——这样后续 action 的偏移量与视频时间轴基本对齐。
        time.sleep(VIDEO_WARMUP_SECONDS)
        if self.proc.poll() is not None:
            err = ""
            if self.proc.stderr is not None:
                try:
                    err = self.proc.stderr.read().decode("utf-8", errors="replace")
                except Exception:
                    err = ""
            raise RuntimeError(f"ffmpeg 启动失败: {err.strip() or '未知原因'}")
        self.start_mono_ns = time.monotonic_ns()
        self.start_wall = datetime.now().isoformat(timespec="microseconds")
        # 启动 daemon 线程持续排空 stderr，防止管道缓冲区满导致 ffmpeg 阻塞
        self._stderr_drain_thread = threading.Thread(
            target=self._drain_stderr, daemon=True,
        )
        self._stderr_drain_thread.start()
        # 注册 atexit 回调，防止 Python 正常退出时 ffmpeg 成为孤儿进程
        atexit.register(self.stop)

    def _drain_stderr(self):
        """持续读取 ffmpeg stderr 防止管道缓冲区满。"""
        try:
            for _ in self.proc.stderr:
                pass
        except Exception:
            pass

    def stop(self):
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                try:
                    if self.proc.stdin is not None:
                        self.proc.stdin.write(b"q")
                        self.proc.stdin.flush()
                        self.proc.stdin.close()
                except Exception:
                    pass
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
                        self.proc.wait(timeout=5)
        finally:
            self.end_mono_ns = time.monotonic_ns()
            self.end_wall = datetime.now().isoformat(timespec="microseconds")
            self.proc = None

    def to_artifact(self):
        return {
            "path": os.path.basename(self.output_path),
            "video_start_mono_ns": self.start_mono_ns,
            "video_start_wall": self.start_wall,
            "video_end_mono_ns": self.end_mono_ns,
            "video_end_wall": self.end_wall,
            "screen_size": self.screen_size,
            "fps": self.fps,
        }


class PlatformAdapter:
    def __init__(self):
        self.platform = sys.platform

    def get_button_name(self, button):
        try:
            return 'left' if button == mouse.Button.left else 'right'
        except Exception:
            return str(button)


class ModifierTracker:
    MODIFIER_KEYS = {
        'ctrl', 'ctrl_l', 'ctrl_r',
        'alt', 'alt_l', 'alt_r', 'alt_gr',
        'shift', 'shift_l', 'shift_r',
        'cmd', 'super', 'win', 'windows'
    }

    def __init__(self):
        self.pressed_modifiers = set()

    def is_modifier(self, key_name):
        return key_name in self.MODIFIER_KEYS

    def get_normalized_name(self, key_name):
        if key_name in ['ctrl_l', 'ctrl_r']:
            return 'ctrl'
        if key_name in ['alt_l', 'alt_r', 'alt_gr']:
            return 'alt'
        if key_name in ['shift_l', 'shift_r']:
            return 'shift'
        if key_name in ['cmd', 'super', 'win', 'windows']:
            return 'win' if sys.platform == 'win32' else 'cmd'
        return key_name

    def press(self, key_name):
        if self.is_modifier(key_name):
            self.pressed_modifiers.add(self.get_normalized_name(key_name))

    def release(self, key_name):
        if self.is_modifier(key_name):
            n = self.get_normalized_name(key_name)
            self.pressed_modifiers.discard(n)

    def has_modifiers(self):
        return len(self.pressed_modifiers) > 0


class DragTracker:
    def __init__(self):
        self.is_dragging = False
        self.drag_start_pos = None
        self.drag_start_time = None
        self.drag_button = None
        self.drag_distance = 0

    def start_drag(self, x, y, button):
        self.is_dragging = True
        self.drag_start_pos = (x, y)
        self.drag_start_time = time.time()
        self.drag_button = button
        self.drag_distance = 0

    def update_drag(self, x, y):
        if self.is_dragging and self.drag_start_pos:
            self.drag_distance = self.calculate_distance(self.drag_start_pos, (x, y))

    def end_drag(self, x, y):
        if self.is_dragging:
            duration = time.time() - self.drag_start_time
            self.is_dragging = False
            return {
                'start_pos': self.drag_start_pos,
                'end_pos': (x, y),
                'duration': duration,
                'distance': self.drag_distance,
                'button': self.drag_button,
                'start_time': self.drag_start_time  # 返回开始时间
            }
        return None

    @staticmethod
    def calculate_distance(pos1, pos2):
        return ((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2) ** 0.5


class ScrollTracker:
    def __init__(self, recorder):
        self.recorder = recorder
        self.scroll_start_time = None
        self.last_scroll_time = None
        self.accumulated_dx = 0
        self.accumulated_dy = 0
        self.scroll_position = None
        self.step_id = None
        # 滚动起止的 monotonic_ns 时间戳，用于抽帧阶段定位 before / after 帧
        self.start_ts_mono_ns = None
        self.last_ts_mono_ns = None
        self.app_title_before = None
        self.modifiers = []

    def add_scroll(self, x, y, dx, dy):
        current_time = time.time()
        current_mono_ns = time.monotonic_ns()

        # 如果当前没有滚动会话，则开启一个新的
        if self.scroll_start_time is None:
            self.step_id = self.recorder.get_next_step_id()
            self.scroll_start_time = current_time
            self.scroll_position = (x, y)
            self.accumulated_dx = 0
            self.accumulated_dy = 0
            self.start_ts_mono_ns = current_mono_ns
            self.app_title_before = self.recorder._capture_app_title()
            self.modifiers = sorted(self.recorder.modifier_tracker.pressed_modifiers)

        # 累积滚动位移
        self.accumulated_dx += dx
        self.accumulated_dy += dy
        self.last_scroll_time = current_time
        self.last_ts_mono_ns = current_mono_ns

    def record_previous_scroll(self):
        if (self.scroll_start_time is not None and
                (self.accumulated_dx != 0 or self.accumulated_dy != 0)):
            total_dy = self.accumulated_dy
            scroll_type = "down" if total_dy < 0 else "up"
            self.recorder.record_scroll(
                self.scroll_position[0], self.scroll_position[1], scroll_type,
                self.step_id,
                self.start_ts_mono_ns,
                self.last_ts_mono_ns or self.start_ts_mono_ns,
                self.app_title_before,
                self.modifiers,
            )
            self.scroll_start_time = None
            self.accumulated_dx = 0
            self.accumulated_dy = 0
            self.step_id = None
            self.start_ts_mono_ns = None
            self.last_ts_mono_ns = None
            self.app_title_before = None
            self.modifiers = []

    def flush(self):
        self.record_previous_scroll()


class ActionRecorder:
    def __init__(self, instruction, task_info=None):
        # 任务信息
        self.task_info = task_info or {}

        # 创建以"类别_ID_时间戳"命名的主文件夹
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        category = self.task_info.get('task_category', 'Unknown')
        task_id = self.task_info.get('task_id', 'uuid')

        # 清理类别和ID中的特殊字符，避免路径问题
        import re
        category_clean = re.sub(r'[^\w\-_]', '_', category)
        task_id_clean = re.sub(r'[^\w\-_]', '_', task_id)

        self.session_dir = f"{category_clean}_{task_id_clean}_{timestamp}"

        # 创建目录（screenshots 目录由抽帧脚本后续创建）
        if not os.path.exists(self.session_dir):
            os.makedirs(self.session_dir)

        # 录屏产物路径
        self.video_path = os.path.join(self.session_dir, VIDEO_FILENAME)
        self.video_recorder = VideoRecorder(self.video_path)

        # 保存指令
        self.instruction = instruction

        self.actions = []
        self.current_input = ""
        self.input_start_time = None
        self.input_start_mono_ns = None
        self.last_click_pos = None
        self.is_recording = False
        self.current_input_session_id = None  # 当前输入会话ID

        # 步骤计数器 - 确保每个操作都有唯一的步骤ID
        self.step_counter = 0

        # 跟踪当前按下的修饰键
        self.modifier_tracker = ModifierTracker()

        # 拖拽跟踪器
        self.drag_tracker = DragTracker()

        # 滚动跟踪器
        self.scroll_tracker = ScrollTracker(self)

        # 双击/三连击检测已下放到 pending_click.click_count + 定时器，
        # last_click_time / last_click_button 不再使用

        if sys.platform.startswith("linux"):
            self.double_click_threshold = 1.0
        else:
            self.double_click_threshold = 0.5  # 双击时间阈值（秒）

        self.double_click_distance = 10  # 双击位置容差（像素）

        # 新的鼠标事件处理状态
        self.pending_click = None  # 待处理的点击事件
        self.pending_click_timer = None  # 点击确认计时器
        self.is_button_pressed = False  # 跟踪按钮是否按下

        # 特殊键跟踪
        self.special_keys_pressed = set()

        # Caps Lock状态跟踪
        self.caps_lock_on = False
        self.num_lock_on = False

        # ========== Windows ==========
        if sys.platform == 'win32':
            try:
                import win32api
                # Caps Lock
                self.caps_lock_on = (win32api.GetKeyState(0x14) & 1) != 0
                # Num Lock
                self.num_lock_on = (win32api.GetKeyState(0x90) & 1) != 0
            except Exception as e:
                print(f"[警告] 无法读取 Windows Caps/Num 状态: {e}")

        # ========== macOS ==========
        elif sys.platform == 'darwin':
            try:
                # 读取 CapsLock 状态：IOKit HIDCapsLockState
                import subprocess
                out = subprocess.check_output(
                    ['ioreg', '-r', '-k', 'HIDCapsLockState'],
                    text=True
                )
                self.caps_lock_on = "HIDCapsLockState = 1" in out

                # macOS 基本无 NumLock（苹果键盘没有）
                self.num_lock_on = False
            except Exception as e:
                print(f"[警告] 无法读取 macOS Caps 状态: {e}")
        # ========== Linux ==========
        elif sys.platform.startswith('linux'):
            try:
                import subprocess
                # 通过 xset q 读取 LED 状态
                out = subprocess.check_output("xset q", shell=True, text=True)
                self.caps_lock_on = "Caps Lock:   on" in out
                self.num_lock_on = "Num Lock:    on" in out
            except Exception as e:
                print(f"[警告] 无法读取 Linux Caps/Num 状态: {e}")

        print(f"初始 CapsLock={self.caps_lock_on}, NumLock={self.num_lock_on}")

        # 平台特定的设置
        self.platform_adapter = PlatformAdapter()
        self.platform = sys.platform
        print(f"检测到操作系统: {self.platform}")

        # 监听器实例
        self.mouse_listener = None
        self.keyboard_listener = None

        # F12停止标志
        self.f12_pressed = False

        # Backspace 连续统计
        self.backspace_streak = 0
        self.pending_backspace_action = None

        # Win/Cmd 延迟确认：key-down 时暂存 step_id 与 mono_ns，key-up 时若中间没别的事件发生才视为单按
        self.pending_win_cmd = None

        # 操作锁 - 确保一个操作完全结束后才开始下一个操作
        # 使用 RLock，允许已持锁的代码路径（如 start_input_session → _finalize_click_streak）重入
        self.operation_lock = threading.RLock()

    def get_next_step_id(self):
        self.step_counter += 1
        return f"s{self.step_counter}"

    def start_recording(self):
        # 先启动录屏，再启动输入监听器，保证最早的 action 时间戳也落在视频时间轴内
        try:
            self.video_recorder.start()
        except Exception as e:
            print(f"录屏启动失败: {e}")
            raise
        self.is_recording = True
        print("开始记录操作... 按F12停止记录")
        try:
            self.mouse_listener = MouseListener(on_click=self.on_click, on_scroll=self.on_scroll, on_move=self.on_move)
            self.keyboard_listener = KeyboardListener(on_press=self.on_press, on_release=self.on_release)
            self.mouse_listener.start()
            self.keyboard_listener.start()
            print("监听器已启动，等待停止信号...")
            while self.is_recording and not self.f12_pressed:
                time.sleep(0.1)
            self.stop_recording()
        except Exception as e:
            print(f"启动监听器时出错: {e}")
            self.stop_recording()

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        try:
            if self.mouse_listener:
                self.mouse_listener.stop()
        except Exception:
            pass
        try:
            if self.keyboard_listener:
                self.keyboard_listener.stop()
        except Exception:
            pass
        # 在停止录屏前，先 finalize 所有未决的操作，确保时间戳落在视频时间轴内
        with self.operation_lock:
            self.scroll_tracker.flush()
            self.flush_backspace_streak()
            if self.pending_click_timer:
                self.pending_click_timer.cancel()
                self.pending_click_timer = None
            self._finalize_click_streak()
            if self.pending_win_cmd is not None:
                pending = self.pending_win_cmd
                self.pending_win_cmd = None
                self.record_special_key_press(
                    pending['key_name'],
                    pending['step_id'],
                    pending['press_ts_mono_ns'],
                    time.monotonic_ns(),
                    pending.get('app_title_before'),
                )
        # 先停录屏，确保后续 save_report 拿到完整的视频起止时间戳
        print("停止录屏...")
        try:
            self.video_recorder.stop()
        except Exception as e:
            print(f"录屏停止失败: {e}")
        self.save_report()

    def _capture_app_title(self):
        try:
            return get_active_app_title()
        except Exception:
            return None

    def is_real_drag_operation(self, drag_data):
        """
        判断是否为真正的拖拽操作，而不是点击间的误判
        """
        if not drag_data:
            return False

        min_drag_distance = 15
        min_drag_duration = 0.2

        if (drag_data['distance'] < min_drag_distance or
                drag_data['duration'] < min_drag_duration):
            return False

        drag_start_time = drag_data.get('start_time', 0)
        drag_end_time = drag_start_time + drag_data['duration']

        for action in self.actions[-5:]:
            try:
                action_time = datetime.fromisoformat(action['timestamp']).timestamp()
            except Exception:
                continue
            if (drag_start_time <= action_time <= drag_end_time and
                    action['type'] in ['typing', 'typing_start', 'press']):
                return False

        if self.has_interrupted_movement(drag_data):
            return False

        return True

    def has_interrupted_movement(self, drag_data):
        speed = drag_data['distance'] / drag_data['duration'] if drag_data['duration'] > 0 else 0
        if speed > 1000:
            return True
        return False

    def on_move(self, x, y):
        if not self.is_recording:
            return
        with self.operation_lock:
            # 鼠标移动会打断滚动和 Backspace 连击
            self.scroll_tracker.record_previous_scroll()
            self.flush_backspace_streak()
            if self.is_button_pressed and self.drag_tracker.is_dragging:
                self.drag_tracker.update_drag(x, y)

    def on_click(self, x, y, button, pressed):
        if not self.is_recording:
            return
        with self.operation_lock:
            # 点击会打断滚动和 Backspace 连击；也说明 pending 的 win/cmd 不是独立单按
            self.scroll_tracker.record_previous_scroll()
            self.flush_backspace_streak()
            self.pending_win_cmd = None
            if self.input_start_time is not None:
                self.finish_typing()

            button_name = self.platform_adapter.get_button_name(button)
            now = time.time()
            if pressed:
                self.is_button_pressed = True
                press_ts = time.monotonic_ns()
                current_modifiers = sorted(self.modifier_tracker.pressed_modifiers)
                if self.pending_click_timer:
                    self.pending_click_timer.cancel()
                    self.pending_click_timer = None

                extending = False
                if self.pending_click is not None:
                    prev = self.pending_click
                    dist = self.drag_tracker.calculate_distance((prev['x'], prev['y']), (x, y))
                    if (prev['button'] == button_name and
                            now - prev['press_time'] < self.double_click_threshold and
                            dist < self.double_click_distance and
                            prev['click_count'] < 3 and
                            prev.get('modifiers', []) == current_modifiers):
                        extending = True

                if extending:
                    self.pending_click['click_count'] += 1
                    self.pending_click['press_time'] = now
                    self.pending_click['last_press_ts_mono_ns'] = press_ts
                    self.pending_click['x'] = x
                    self.pending_click['y'] = y
                else:
                    if self.pending_click is not None:
                        self._finalize_click_streak()
                    step_id = self.get_next_step_id()
                    self.pending_click = {
                        'x': x, 'y': y, 'button': button_name,
                        'press_time': now,
                        'first_press_ts_mono_ns': press_ts,
                        'last_press_ts_mono_ns': press_ts,
                        'step_id': step_id,
                        'click_count': 1,
                        'app_title_before': self._capture_app_title(),
                        'modifiers': current_modifiers,
                    }

                self.last_click_pos = (x, y)
                self.drag_tracker.start_drag(x, y, button_name)
                self.drag_tracker.drag_start_ts_mono_ns = press_ts
            else:
                self.is_button_pressed = False
                if self.drag_tracker.is_dragging:
                    drag_start_ts = getattr(self.drag_tracker, 'drag_start_ts_mono_ns', None)
                    drag_data = self.drag_tracker.end_drag(x, y)
                    drag_end_ts = time.monotonic_ns()
                    is_real_drag = self.is_real_drag_operation(drag_data)

                    if is_real_drag:
                        if self.pending_click is not None:
                            data = self.pending_click
                            self.pending_click = None
                            self.record_drag(
                                drag_data,
                                data['step_id'],
                                drag_start_ts or data['first_press_ts_mono_ns'],
                                drag_end_ts,
                                data.get('app_title_before'),
                                self._capture_app_title(),
                                data.get('modifiers', []),
                            )
                    else:
                        if self.pending_click is not None:
                            self.pending_click_timer = threading.Timer(
                                self.double_click_threshold,
                                self._finalize_click_streak_threadsafe,
                            )
                            self.pending_click_timer.start()

    def _finalize_click_streak(self):
        """根据 pending_click.click_count 产出 click / double_click / triple_click。
        调用方须已持 operation_lock（RLock 允许重入）。"""
        if self.pending_click is None:
            return
        data = self.pending_click
        self.pending_click = None
        if self.pending_click_timer:
            try:
                self.pending_click_timer.cancel()
            except Exception:
                pass
            self.pending_click_timer = None
        count = data.get('click_count', 1)
        x, y, button = data['x'], data['y'], data['button']
        step_id = data['step_id']
        before_ts = data['first_press_ts_mono_ns']
        after_ts = time.monotonic_ns()
        title_before = data.get('app_title_before')
        title_after = self._capture_app_title()
        modifiers = data.get('modifiers', [])
        if count >= 3:
            self.record_triple_click(x, y, button, step_id, before_ts, after_ts, title_before, title_after, modifiers)
        elif count == 2:
            self.record_double_click(x, y, button, step_id, before_ts, after_ts, title_before, title_after, modifiers)
        else:
            self.record_click(x, y, button, step_id, before_ts, after_ts, title_before, title_after, modifiers)

    def _finalize_click_streak_threadsafe(self):
        """Timer 线程入口：需要先拿到 operation_lock。"""
        with self.operation_lock:
            self._finalize_click_streak()

    def record_click(self, x, y, button_name, step_id, before_ts_mono_ns, after_ts_mono_ns, app_title_before=None, app_title_after=None, modifiers=None):
        modifiers = list(modifiers) if modifiers else []
        param = {'button': button_name, 'num_click': 1}
        if modifiers:
            param['modifiers'] = modifiers
        action = {
            'type': 'click',
            'param': param,
            'target': {'position': (x, y)},
            'before_ts_mono_ns': before_ts_mono_ns,
            'after_ts_mono_ns': after_ts_mono_ns,
            'anchor_position': (x, y),
            'timestamp': datetime.now().isoformat(),
            'step_id': step_id,
            'app_title_before': app_title_before,
            'app_title_after': app_title_after,
        }
        self.actions.append(action)
        self.last_click_pos = (x, y)
        mod_suffix = f" [{'+'.join(modifiers)}]" if modifiers else ""
        print(f"点击: ({x}, {y}) - {button_name}{mod_suffix}")

    def record_double_click(self, x, y, button_name, step_id, before_ts_mono_ns, after_ts_mono_ns, app_title_before=None, app_title_after=None, modifiers=None):
        modifiers = list(modifiers) if modifiers else []
        param = {'button': button_name, 'num_click': 2}
        if modifiers:
            param['modifiers'] = modifiers
        action = {
            'type': 'double_click',
            'param': param,
            'target': {'position': (x, y)},
            'before_ts_mono_ns': before_ts_mono_ns,
            'after_ts_mono_ns': after_ts_mono_ns,
            'anchor_position': (x, y),
            'timestamp': datetime.now().isoformat(),
            'step_id': step_id,
            'app_title_before': app_title_before,
            'app_title_after': app_title_after,
        }
        self.actions.append(action)
        self.last_click_pos = (x, y)
        mod_suffix = f" [{'+'.join(modifiers)}]" if modifiers else ""
        print(f"双击: ({x}, {y}) - {button_name}{mod_suffix}")

    def record_triple_click(self, x, y, button_name, step_id, before_ts_mono_ns, after_ts_mono_ns, app_title_before=None, app_title_after=None, modifiers=None):
        modifiers = list(modifiers) if modifiers else []
        param = {'button': button_name, 'num_click': 3}
        if modifiers:
            param['modifiers'] = modifiers
        action = {
            'type': 'triple_click',
            'param': param,
            'target': {'position': (x, y)},
            'before_ts_mono_ns': before_ts_mono_ns,
            'after_ts_mono_ns': after_ts_mono_ns,
            'anchor_position': (x, y),
            'timestamp': datetime.now().isoformat(),
            'step_id': step_id,
            'app_title_before': app_title_before,
            'app_title_after': app_title_after,
        }
        self.actions.append(action)
        self.last_click_pos = (x, y)
        mod_suffix = f" [{'+'.join(modifiers)}]" if modifiers else ""
        print(f"三连击: ({x}, {y}) - {button_name}{mod_suffix}")

    def calculate_drag_direction(self, start_pos, end_pos):
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360
        return angle

    def record_drag(self, drag_data, step_id, before_ts_mono_ns, after_ts_mono_ns, app_title_before=None, app_title_after=None, modifiers=None):
        modifiers = list(modifiers) if modifiers else []
        sx, sy = drag_data['start_pos']
        ex, ey = drag_data['end_pos']
        angle = self.calculate_drag_direction((sx, sy), (ex, ey))
        param = {'button': drag_data['button']}
        if modifiers:
            param['modifiers'] = modifiers
        action = {
            'type': 'drag_to',
            'param': param,
            'target': {'Begin': {'position': (sx, sy)}, 'End': {'position': (ex, ey)},
                       'describe': {'angle': angle, 'distance': drag_data['distance']}},
            'before_ts_mono_ns': before_ts_mono_ns,
            'after_ts_mono_ns': after_ts_mono_ns,
            'anchor_position': (sx, sy),
            'timestamp': datetime.now().isoformat(),
            'drag_data': drag_data,
            'step_id': step_id,
            'app_title_before': app_title_before,
            'app_title_after': app_title_after,
        }
        self.actions.append(action)
        mod_suffix = f" [{'+'.join(modifiers)}]" if modifiers else ""
        print(f"拖拽: 从 ({sx},{sy}) 到 ({ex},{ey}) 距离: {drag_data['distance']:.2f}{mod_suffix}")

    def on_scroll(self, x, y, dx, dy):
        if not self.is_recording:
            return
        with self.operation_lock:
            # 滚动会打断 Backspace 连击；也说明 pending 的 win/cmd 不是独立单按
            self.flush_backspace_streak()
            self.pending_win_cmd = None
            if self.input_start_time is not None:
                self.finish_typing()

            self.scroll_tracker.add_scroll(x, y, dx, dy)

    def record_scroll(self, x, y, scroll_type, step_id, before_ts_mono_ns, after_ts_mono_ns, app_title_before=None, modifiers=None):
        modifiers = list(modifiers) if modifiers else []
        param = {'type': scroll_type}
        if modifiers:
            param['modifiers'] = modifiers
        action = {
            'type': 'scroll',
            'param': param,
            'target': {'position': (x, y)},
            'before_ts_mono_ns': before_ts_mono_ns,
            'after_ts_mono_ns': after_ts_mono_ns,
            'anchor_position': (x, y),
            'timestamp': datetime.now().isoformat(),
            'step_id': step_id,
            'app_title_before': app_title_before,
            'app_title_after': self._capture_app_title(),
        }
        self.actions.append(action)
        mod_suffix = f" [{'+'.join(modifiers)}]" if modifiers else ""
        print(f"滚动: {scroll_type} at ({x},{y}){mod_suffix}")

    def get_key_name(self, key):
        try:
            if hasattr(key, 'name'):
                return key.name.lower()
            if hasattr(key, 'char') and key.char is not None:
                char = key.char
                if len(char) == 1:
                    code = ord(char)
                    if 1 <= code <= 26:
                        return chr(code + 96)
                return char
            return str(key).replace("'", "").lower()
        except:
            return str(key).lower()

    def get_character_with_caps_lock(self, char, key_name):
        if not char.isalpha():
            return char
        shift = 'shift' in self.modifier_tracker.pressed_modifiers
        if self.caps_lock_on:
            return char.lower() if shift else char.upper()
        else:
            return char.upper() if shift else char.lower()

    def is_numpad_key(self, key_name):
        numpad_keys = {'<97>', '<98>', '<99>', '<100>', '<101>', '<102>', '<103>', '<104>', '<105>', '<96>', 'num_lock',
                       'num_divide', 'num_multiply', 'num_subtract', 'num_add', 'num_enter', 'num_decimal'}
        return key_name in numpad_keys

    def get_numpad_character(self, key_name):
        mapping = {'<97>': '1', '<98>': '2', '<99>': '3', '<100>': '4', '<101>': '5', '<102>': '6', '<103>': '7',
                   '<104>': '8', '<105>': '9', '<96>': '0', 'num_divide': '/', 'num_multiply': '*', 'num_subtract': '-',
                   'num_add': '+', 'num_decimal': '.'}
        return mapping.get(key_name)

    def record_press(self, key_name, step_id, position, press_ts_mono_ns):
        modifiers = list(self.modifier_tracker.pressed_modifiers) if self.modifier_tracker.pressed_modifiers else []
        press_list = modifiers + [key_name] if key_name not in modifiers else modifiers
        title = self._capture_app_title()
        action = {
            'type': 'press',
            'param': {'press_list': press_list},
            'target': {'position': position} if position else {},
            'before_ts_mono_ns': press_ts_mono_ns,
            'after_ts_mono_ns': time.monotonic_ns(),
            'timestamp': datetime.now().isoformat(),
            'step_id': step_id,
            'app_title_before': title,
            'app_title_after': title,
        }
        self.actions.append(action)
        print(f"按键: {'+'.join(press_list)}")

    def flush_backspace_streak(self):
        """将正在累积的 Backspace 连击刷新为一次 press 操作。"""
        if self.pending_backspace_action and self.backspace_streak > 0:
            self.pending_backspace_action.setdefault('param', {}).setdefault('press_list', ['backspace'])
            self.pending_backspace_action['param']['press_count'] = self.backspace_streak
            self.pending_backspace_action['after_ts_mono_ns'] = time.monotonic_ns()
            self.pending_backspace_action['app_title_after'] = self._capture_app_title()
            self.actions.append(self.pending_backspace_action)

        self.pending_backspace_action = None
        self.backspace_streak = 0


    def record_special_key_press(self, key_name, step_id, before_ts_mono_ns, after_ts_mono_ns, app_title_before=None):
        action = {
            'type': 'press',
            'param': {'press_list': [key_name]},
            'target': {},
            'before_ts_mono_ns': before_ts_mono_ns,
            'after_ts_mono_ns': after_ts_mono_ns,
            'timestamp': datetime.now().isoformat(),
            'step_id': step_id,
            'is_special_key': True,
            'app_title_before': app_title_before,
            'app_title_after': self._capture_app_title(),
        }
        self.actions.append(action)
        print(f"特殊键按下: {key_name}")

    def start_input_session(self, trigger_key=None):
        has_non_shift_modifiers = any(mod not in ['shift'] for mod in self.modifier_tracker.pressed_modifiers)
        if has_non_shift_modifiers:
            return

        if self.input_start_time is not None:
            return

        if self.pending_click:
            try:
                if self.pending_click_timer:
                    self.pending_click_timer.cancel()
                    self.pending_click_timer = None
            except:
                pass

            self._finalize_click_streak()

        self.input_start_time = time.time()
        self.input_start_mono_ns = time.monotonic_ns()
        self.current_input_session_id = len(self.actions)
        step_id = self.get_next_step_id()

        action = {
            'type': 'typing_start',
            'position': self.last_click_pos,
            'before_ts_mono_ns': self.input_start_mono_ns,
            'after_ts_mono_ns': None,
            'text': '',
            'trigger_key': trigger_key,
            '_session_id': self.current_input_session_id,
            'timestamp': datetime.now().isoformat(),
            'step_id': step_id,
            'app_title_before': self._capture_app_title(),
            'app_title_after': None,
        }

        self.actions.append(action)

    def on_press(self, key):
        if not self.is_recording:
            return
        with self.operation_lock:
            # 键盘事件会打断滚动；非 Backspace 键会打断 Backspace 连击
            self.scroll_tracker.record_previous_scroll()
            if key != Key.backspace:
                self.flush_backspace_streak()
            try:
                key_name = self.get_key_name(key)
                if key == Key.f12:
                    print("检测到F12键，准备停止记录...")
                    self.f12_pressed = True
                    return
                if key == Key.caps_lock:
                    self.caps_lock_on = not self.caps_lock_on
                    print(f"Caps Lock: {'开' if self.caps_lock_on else '关'}")
                    return
                # Win/Cmd 延迟确认：按下瞬间不写 action，只暂存 snapshot 与 step_id；
                # 若中间有任何其它事件 → 取消 pending（说明是组合键）；
                # 若 key-up 前无其它事件 → 在 on_release 里才正式记录为 special_key。
                normalized_name = self.modifier_tracker.get_normalized_name(key_name)
                if normalized_name in ('win', 'cmd') and not self.modifier_tracker.has_modifiers():
                    cx, cy = pyautogui.position()
                    step_id = self.get_next_step_id()
                    self.pending_win_cmd = {
                        'key_name': normalized_name,
                        'step_id': step_id,
                        'position': (cx, cy),
                        'press_ts_mono_ns': time.monotonic_ns(),
                        'press_time': time.time(),
                        'app_title_before': self._capture_app_title(),
                    }
                    self.modifier_tracker.press(key_name)
                    return
                if self.modifier_tracker.is_modifier(key_name):
                    # 普通修饰键按下说明 win/cmd 进入组合键模式
                    self.pending_win_cmd = None
                    self.modifier_tracker.press(key_name)
                    return

                # 任何非修饰键的 key-down 也说明 win/cmd 并非独立单按
                self.pending_win_cmd = None

                has_non_shift_modifiers = any(mod not in ['shift'] for mod in self.modifier_tracker.pressed_modifiers)

                is_char_key = False
                char_value = None
                if hasattr(key, 'char') and key.char is not None:
                    is_char_key = True
                    char_value = self.get_character_with_caps_lock(key.char, key_name)
                elif self.is_numpad_key(key_name):
                    nc = self.get_numpad_character(key_name)
                    if nc is not None:
                        is_char_key = True
                        char_value = nc
                elif len(key_name) == 1 and key_name.isprintable():
                    is_char_key = True
                    char_value = self.get_character_with_caps_lock(key_name, key_name)

                should_interrupt_typing = (
                                                  has_non_shift_modifiers
                                          ) or (
                                                  not is_char_key and
                                                  key not in [Key.space, Key.backspace, Key.tab, Key.enter]
                                          )

                if should_interrupt_typing:
                    if self.input_start_time is not None:
                        self.finish_typing()

                needs_new_session = (self.input_start_time is None and
                                     self.last_click_pos is not None and
                                     is_char_key and
                                     not has_non_shift_modifiers)

                if not is_char_key and key not in [Key.space, Key.enter, Key.backspace, Key.tab]:
                    cx, cy = pyautogui.position()
                    step_id = self.get_next_step_id()
                    self.record_press(key_name, step_id, (cx, cy), time.monotonic_ns())
                    return

                if needs_new_session:
                    self.start_input_session(key_name)

                if is_char_key:
                    if has_non_shift_modifiers:
                        cx, cy = pyautogui.position()
                        step_id = self.get_next_step_id()
                        self.record_press(key_name, step_id, (cx, cy), time.monotonic_ns())
                    else:
                        if not self.modifier_tracker.pressed_modifiers or 'shift' in self.modifier_tracker.pressed_modifiers:
                            if self.input_start_time is None:
                                self.start_input_session(key_name)
                            self.current_input += char_value
                        else:
                            cx, cy = pyautogui.position()
                            step_id = self.get_next_step_id()
                            self.record_press(key_name, step_id, (cx, cy), time.monotonic_ns())
                elif key == Key.space:
                    if has_non_shift_modifiers:
                        cx, cy = pyautogui.position()
                        step_id = self.get_next_step_id()
                        self.record_press('space', step_id, (cx, cy), time.monotonic_ns())
                    else:
                        if self.input_start_time is not None:
                            self.current_input += ' '
                        else:
                            cx, cy = pyautogui.position()
                            step_id = self.get_next_step_id()
                            self.record_press('space', step_id, (cx, cy), time.monotonic_ns())
                elif key == Key.enter:
                    if has_non_shift_modifiers:
                        cx, cy = pyautogui.position()
                        step_id = self.get_next_step_id()
                        self.record_press('enter', step_id, (cx, cy), time.monotonic_ns())
                    else:
                        if self.input_start_time is not None:
                            self.finish_typing()
                        mx, my = pyautogui.position()
                        step_id = self.get_next_step_id()
                        self.record_press('enter', step_id, (mx, my), time.monotonic_ns())
                    return
                elif key == Key.backspace:
                    if has_non_shift_modifiers:
                        cx, cy = pyautogui.position()
                        step_id = self.get_next_step_id()
                        self.record_press('backspace', step_id, (cx, cy), time.monotonic_ns())
                    else:
                        if self.input_start_time is not None:
                            self.current_input = self.current_input[:-1]
                        else:
                            if self.pending_backspace_action is None:
                                step_id = self.get_next_step_id()
                                self.pending_backspace_action = {
                                    'type': 'press',
                                    'param': {'press_list': ['backspace']},
                                    'target': {},
                                    'before_ts_mono_ns': time.monotonic_ns(),
                                    'after_ts_mono_ns': None,
                                    'timestamp': datetime.now().isoformat(),
                                    'step_id': step_id,
                                    'app_title_before': self._capture_app_title(),
                                    'app_title_after': None,
                                }
                                self.backspace_streak = 1
                            else:
                                self.backspace_streak += 1
                elif key == Key.tab:
                    if has_non_shift_modifiers:
                        cx, cy = pyautogui.position()
                        step_id = self.get_next_step_id()
                        self.record_press('tab', step_id, (cx, cy), time.monotonic_ns())
                    else:
                        if self.input_start_time is not None:
                            self.finish_typing()
                        cx, cy = pyautogui.position()
                        step_id = self.get_next_step_id()
                        self.record_press('tab', step_id, (cx, cy), time.monotonic_ns())
                    return
                else:
                    cx, cy = pyautogui.position()
                    step_id = self.get_next_step_id()
                    self.record_press(key_name, step_id, (cx, cy), time.monotonic_ns())

            except AttributeError:
                if self.input_start_time is not None:
                    self.finish_typing()

                key_name = self.get_key_name(key)
                cx, cy = pyautogui.position()
                step_id = self.get_next_step_id()
                self.record_press(key_name, step_id, (cx, cy), time.monotonic_ns())

    def on_release(self, key):
        if not self.is_recording:
            return
        with self.operation_lock:
            key_name = self.get_key_name(key)
            if self.modifier_tracker.is_modifier(key_name):
                normalized = self.modifier_tracker.get_normalized_name(key_name)
                # Win/Cmd 确认：若 pending 仍在且匹配，则此次视为独立单按
                if (normalized in ('win', 'cmd') and
                        self.pending_win_cmd is not None and
                        self.pending_win_cmd.get('key_name') == normalized):
                    pending = self.pending_win_cmd
                    self.pending_win_cmd = None
                    if self.input_start_time is not None:
                        self.finish_typing()
                    self.record_special_key_press(
                        normalized,
                        pending['step_id'],
                        pending['press_ts_mono_ns'],
                        time.monotonic_ns(),
                        pending.get('app_title_before'),
                    )
                self.modifier_tracker.release(key_name)
                return
            if key in [Key.enter, Key.tab] and self.input_start_time is not None:
                pass

    def finish_typing(self):
        result = None
        if self.input_start_time is not None:
            start_index = None
            for i in range(len(self.actions) - 1, -1, -1):
                if (self.actions[i].get('_session_id') == self.current_input_session_id and
                        self.actions[i]['type'] == 'typing_start'):
                    start_index = i
                    break

            if start_index is None:
                self.current_input = ""
                self.input_start_time = None
                self.input_start_mono_ns = None
                self.current_input_session_id = None
                return None

            if not self.current_input:
                del self.actions[start_index]
            else:
                step_id = self.actions[start_index]['step_id']
                end_ts = time.monotonic_ns()
                self.actions[start_index]['type'] = 'typing'
                self.actions[start_index]['param'] = {'text': self.current_input}
                self.actions[start_index]['target'] = {}
                self.actions[start_index]['after_ts_mono_ns'] = end_ts
                self.actions[start_index]['app_title_after'] = self._capture_app_title()
                self.actions[start_index]['duration'] = time.time() - self.input_start_time
                result = {'step_id': step_id, 'after_ts_mono_ns': end_ts}

        self.current_input = ""
        self.input_start_time = None
        self.input_start_mono_ns = None
        self.current_input_session_id = None
        return result

    def save_report(self):
        self.flush_backspace_streak()
        if self.input_start_time is not None:
            self.finish_typing()
        screen_w, screen_h = pyautogui.size()
        env_info = {'os': self.platform, 'screen': f"{screen_w}x{screen_h}", 'url': self.task_info.get('url', ''),
                    'locale': self.task_info.get('locale', 'en_US')}
        video_artifact = self.video_recorder.to_artifact()
        video_start_mono_ns = video_artifact.get('video_start_mono_ns')
        steps = []
        for i, action in enumerate(self.actions):
            target = action.get('target', {}) or {}
            if isinstance(target, dict) and 'position' in target and 'nl_position' not in target:
                target = dict(target)
                target['nl_position'] = ""
            action_struct = {'type': action['type'], 'target': target}
            if 'param' in action:
                action_struct['param'] = action['param']

            anchor = action.get('anchor_position')
            before_ts = action.get('before_ts_mono_ns')
            after_ts = action.get('after_ts_mono_ns')
            if after_ts is None and before_ts is not None:
                after_ts = before_ts
                print(f"[警告] 步骤 {action.get('step_id', '?')} 缺少 after_ts_mono_ns，已从 before_ts 估算")

            capture_plan = {
                'before_ts_mono_ns': before_ts,
                'after_ts_mono_ns': after_ts,
                'video_start_mono_ns': video_start_mono_ns,
                'anchor_position': list(anchor) if anchor is not None else None,
            }

            step = {
                'step_id': action.get('step_id', f"s{i + 1}"),
                'step_goal': '',
                'now_state': {
                    'screenshot_path_before': None,
                    'screenshot_path_before_part': None,
                    'screenshot_path_after': None,
                    'screenshot_time_before': None,
                    'screenshot_time_after': None,
                    'app_title_before': action.get('app_title_before'),
                    'app_title_after': action.get('app_title_after'),
                },
                'capture_plan': capture_plan,
                'action_preconditions': [],
                'action': action_struct,
                'action_before_state': "",
                'action_after_effects': [],
                'nl_explanation': '',
            }
            steps.append(step)
        report = {
            'task_id': self.task_info.get('task_id', 'uuid'),
            'task_category': self.task_info.get('task_category', ''),
            'task_title': '',
            'instruction': self.instruction,
            'app': '',
            'env': env_info,
            'video_artifact': video_artifact,
            'steps': steps,
        }
        json_report_path = os.path.join(self.session_dir, 'report.json')
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"记录已保存: {len(self.actions)} 个操作")
        print(f"会话目录: {self.session_dir}")
        print(f"录屏文件: {self.video_path}")
        print("使用 extract_frames_v22.py <session_dir> 抽取 before/after 截图")


def get_instruction():
    while True:
        instruction = input("请输入此次操作的指令: ").strip()
        if instruction:
            return instruction
        else:
            print("指令不能为空，请重新输入！")


def get_task_info():
    task_info = {}
    task_id = input("请输入任务ID (留空使用默认): ").strip()
    task_info['task_id'] = task_id if task_id else 'uuid'
    task_category = input("请输入任务类别 (留空使用默认): ").strip()
    task_info['task_category'] = task_category if task_category else 'Unknown'
    task_info.update({'url': '', 'locale': 'en_US'})
    return task_info


def main():
    task_info = get_task_info()
    instruction = get_instruction()
    print("操作记录器 已启动")
    recorder = ActionRecorder(instruction, task_info)
    try:
        recorder.start_recording()
    except KeyboardInterrupt:
        recorder.stop_recording()
    except Exception as e:
        print(f"发生错误: {e}")
        recorder.stop_recording()
    print("进程结束，请按回车退出")
    try:
        input()
    except:
        pass


if __name__ == '__main__':
    main()
