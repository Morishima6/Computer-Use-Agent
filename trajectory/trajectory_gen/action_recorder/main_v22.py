import json
import math
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pyautogui
import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw as PILImageDraw

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
SCREENSHOT_DELAY = 2.0  # 截取图片的延迟（after 截图延迟）
REGION_WIDTH = 200  # partial screenshot width
REGION_HEIGHT = 200  # partial screenshot height
HIGHLIGHT_RADIUS = 20  # circle_radius
HIGHLIGHT_WIDTH = 4  # cross_width


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
        self.before_screenshot = None
        self.step_id = None

    def add_scroll(self, x, y, dx, dy):
        current_time = time.time()

        # 如果当前没有滚动会话，则开启一个新的
        if self.scroll_start_time is None:
            self.step_id = self.recorder.get_next_step_id()
            self.before_screenshot = self.recorder.take_screenshot(self.step_id, "scroll", "before", (x, y))
            self.scroll_start_time = current_time
            self.scroll_position = (x, y)
            self.accumulated_dx = 0
            self.accumulated_dy = 0

        # 累积滚动位移
        self.accumulated_dx += dx
        self.accumulated_dy += dy
        self.last_scroll_time = current_time

    def record_previous_scroll(self):
        if (self.scroll_start_time is not None and
                (self.accumulated_dx != 0 or self.accumulated_dy != 0)):
            total_dy = self.accumulated_dy
            scroll_type = "down" if total_dy < 0 else "up"
            self.recorder.record_scroll(
                self.scroll_position[0], self.scroll_position[1], scroll_type, self.before_screenshot, self.step_id
            )
            self.scroll_start_time = None
            self.accumulated_dx = 0
            self.accumulated_dy = 0
            self.before_screenshot = None
            self.step_id = None

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
        self.screenshot_dir = os.path.join(self.session_dir, "screenshots")

        # 创建目录
        if not os.path.exists(self.session_dir):
            os.makedirs(self.session_dir)
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)

        # 保存指令
        self.instruction = instruction

        self.actions = []
        self.screenshot_count = 0
        self.screenshot_app_titles = {}
        self.current_input = ""
        self.input_start_time = None
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

        # 双击检测相关
        self.last_click_time = None
        self.last_click_button = None

        if sys.platform == "Linux":
            self.double_click_threshold = 1.0
        else:
            self.double_click_threshold = 0.5  # 双击时间阈值（秒）

        self.double_click_distance = 10  # 双击位置容差（像素）

        # 新的鼠标事件处理状态
        self.pending_click = None  # 待处理的点击事件
        self.pending_click_timer = None  # 点击确认计时器
        self.is_button_pressed = False  # 跟踪按钮是否按下

        # 连点（N击）跟踪
        self.click_streak = 0  # 当前连点击数
        self.pending_click_streak = None  # 当前连点的累积数据
        self.pending_click = None  # 兼容字段，当前指向 pending_click_streak
        # 第3次 press 时暂存数据，等待 release 时才拍快照（OS 在 release 后才应用 triple-click 效果）
        self.pending_n_click_pending_release = False
        self.pending_n_click_data = None

        # 特殊键跟踪
        self.special_keys_pressed = set()

        # Caps Lock状态跟踪
        self.caps_lock_on = False
        self.nums_lock_on = False

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

        # 创建线程池用于处理截图任务
        self.executor = ThreadPoolExecutor(max_workers=4)

        # 监听器实例
        self.mouse_listener = None
        self.keyboard_listener = None

        # 用于跟踪待处理的after截图（存放 dict：{'step_id', 'action_type', 'snapshot'}）
        self.pending_after_screenshots = []
        self.pending_lock = threading.Lock()

        # F12停止标志
        self.f12_pressed = False

        # Backspace 连续统计
        self.backspace_streak = 0
        self.pending_backspace_action = None

        # 操作锁 - 确保一个操作完全结束后才开始下一个操作
        self.operation_lock = threading.Lock()

    def get_next_step_id(self):
        self.step_counter += 1
        return f"s{self.step_counter}"

    def start_recording(self):
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
        previous_sigint_handler = None
        can_mask_sigint = threading.current_thread() is threading.main_thread()
        if can_mask_sigint:
            try:
                previous_sigint_handler = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            except Exception:
                previous_sigint_handler = None
        try:
            if self.mouse_listener:
                self.mouse_listener.stop()
        except:
            pass
        try:
            if self.keyboard_listener:
                self.keyboard_listener.stop()
        except:
            pass
        self.scroll_tracker.flush()
        self.flush_backspace_streak()
        if self.pending_click_timer:
            self.pending_click_timer.cancel()
            self.pending_click_timer = None
        if self.pending_n_click_pending_release:
            # triple-click press 已发生但 release 尚未到达（录制中途停止），
            # 立即完成记录，截图取当前画面（不完美但可接受）
            data = self.pending_n_click_data
            self.record_n_click(data['x'], data['y'], data['button'],
                                data['num_click'], data['before_screenshot'], data['step_id'])
            self.pending_n_click_pending_release = False
            self.pending_n_click_data = None
        elif self.pending_click_streak is not None:
            self._commit_click_streak()
        try:
            print("停止记录，等待所有异步截图完成...")
            self.wait_for_pending_screenshots()
            self.executor.shutdown(wait=True)
        except Exception as e:
            print(f"等待异步截图完成时出错: {e}")
        try:
            self.save_report()
        except Exception as e:
            print(f"保存报告时出错: {e}")
        finally:
            if can_mask_sigint and previous_sigint_handler is not None:
                try:
                    signal.signal(signal.SIGINT, previous_sigint_handler)
                except Exception:
                    pass

    def wait_for_pending_screenshots(self):
        max_wait = 5
        start = time.time()
        while True:
            with self.pending_lock:
                remaining = len(self.pending_after_screenshots)
            if remaining == 0:
                break
            if (time.time() - start) > max_wait:
                break
            time.sleep(0.05)
        with self.pending_lock:
            if self.pending_after_screenshots:
                print(f"警告: {len(self.pending_after_screenshots)} 个 after 截图仍未完成，将继续等待线程池收尾")

    def _record_screenshot_app_title(self, filepath, app_title=None):
        """为指定截图文件记录前台窗口标题。

        - app_title 不为 None 时，直接使用该值（通常为截图时刻采集的标题）；
        - 否则在当前时刻调用 get_active_app_title()，作为回退。
        """
        if not filepath:
            return
        title = app_title
        if title is None:
            try:
                title = get_active_app_title()
            except Exception:
                title = None
        if title:
            abs_path = os.path.abspath(filepath)
            self.screenshot_app_titles[abs_path] = title

    def _move_screenshot_app_title(self, old_path, new_path):
        """在重命名截图文件时同步更新标题映射。"""
        if not old_path or not new_path:
            return
        old_abs = os.path.abspath(old_path)
        new_abs = os.path.abspath(new_path)
        if old_abs in self.screenshot_app_titles:
            self.screenshot_app_titles[new_abs] = self.screenshot_app_titles.pop(old_abs)

    def take_screenshot(self, step_id, action_type, timing, position=None, rename_from=None):
        try:
            # 用于复用已有截图（如从 click 复用为 double_click / drag）
            if rename_from and os.path.exists(rename_from):
                filename = f"{step_id}_{timing}_{action_type}.png"
                filepath = os.path.join(self.screenshot_dir, filename)
                os.rename(rename_from, filepath)
                print(f"截图重命名: {os.path.basename(rename_from)} -> {filename}")
                self._move_screenshot_app_title(rename_from, filepath)
                return filepath

            # 在真正截图前采集一次当前前台窗口标题，确保与截图时间尽量一致
            try:
                app_title = get_active_app_title()
            except Exception:
                app_title = None

            screenshot = pyautogui.screenshot()

            # 对于带 position 的 before 截图：
            # 1) 先截取 position 附近的小区域，保存为 xxx(part).png
            # 2) 在整张截图上画红色十字与圆圈，保存为 xxx.png
            if position is not None and timing == 'before':
                x, y = position
                width, height = screenshot.size

                # ----------- 1) 保存局部(part)截图 -----------
                half_w = REGION_WIDTH // 2
                half_h = REGION_HEIGHT // 2
                left = max(x - half_w, 0)
                top = max(y - half_h, 0)
                right = min(left + REGION_WIDTH, width)
                bottom = min(top + REGION_HEIGHT, height)

                region = screenshot.crop((left, top, right, bottom))
                part_filename = f"{step_id}_{timing}_{action_type}(part).png"
                part_filepath = os.path.join(self.screenshot_dir, part_filename)
                region.save(part_filepath)
                self.screenshot_count += 1
                print(f"局部截图保存: {part_filename}")

                # ----------- 2) 在整张图上添加红色标记 -----------
                draw = PILImageDraw.Draw(screenshot)
                r = HIGHLIGHT_RADIUS
                draw.ellipse([x - r, y - r, x + r, y + r], outline='red', width=HIGHLIGHT_WIDTH)
                line_length = r * 2
                draw.line([x - line_length, y, x + line_length, y], fill='red', width=HIGHLIGHT_WIDTH)
                draw.line([x, y - line_length, x, y + line_length], fill='red', width=HIGHLIGHT_WIDTH)

            # ------------------ 保存整张截图（含标记或原图） ------------------
            filename = f"{step_id}_{timing}_{action_type}.png"
            filepath = os.path.join(self.screenshot_dir, filename)

            # 保存时统一转为 RGB，避免透明通道问题
            if screenshot.mode != "RGB":
                screenshot_to_save = screenshot.convert("RGB")
            else:
                screenshot_to_save = screenshot

            screenshot_to_save.save(filepath)
            self.screenshot_count += 1
            print(f"截图保存: {filename}")
            self._record_screenshot_app_title(filepath, app_title)
            return filepath
        except Exception as e:
            print(f"截图失败: {e}")
            return None

    def _capture_snapshot(self):
        """立即抓取并返回包含截图和当前 app 标题的 snapshot（冻结画面）。"""
        try:
            snap = pyautogui.screenshot()
        except Exception as e:
            print(f"snapshot 捕获失败: {e}")
            return None
        # 在截图时刻采集一次 app_title，避免 after 延迟保存时被前台窗口切换影响
        try:
            title = get_active_app_title()
        except Exception:
            title = None
        return {"image": snap, "app_title": title}

    def _schedule_after_from_snapshot(self, step_id, action_type, snapshot, action_ref):
        """
        将 snapshot 保存任务提交到线程池：
         - 线程先 sleep(SCREENSHOT_DELAY)
         - 之后将 snapshot 保存为 after 文件（不再重新截图）
        action_ref 是对对应 action dict 的引用，用于写入 after_screenshot 字段
        """

        def worker():
            try:
                time.sleep(SCREENSHOT_DELAY)
                if snapshot is None:
                    after_path = self.take_screenshot(step_id, action_type, 'after', None)
                else:
                    # snapshot 结构：{"image": PIL.Image, "app_title": str | None}
                    snap_img = snapshot.get("image") if isinstance(snapshot, dict) else snapshot
                    snap_title = snapshot.get("app_title") if isinstance(snapshot, dict) else None
                    if snap_img is None:
                        after_path = self.take_screenshot(step_id, action_type, 'after', None)
                    else:
                        filename = f"{step_id}_after_{action_type}.png"
                        filepath = os.path.join(self.screenshot_dir, filename)
                        try:
                            snap = snap_img
                            if snap.mode != "RGB":
                                snap = snap.convert("RGB")
                            snap.save(filepath)
                            self.screenshot_count += 1
                            after_path = filepath
                            print(f"（延期保存）截图保存: {filename}")
                            self._record_screenshot_app_title(after_path, snap_title)
                        except Exception as e:
                            print(f"保存 snapshot 失败: {e}")
                            after_path = self.take_screenshot(step_id, action_type, 'after', None)
                if action_ref is not None:
                    action_ref['after_screenshot'] = after_path
            finally:
                with self.pending_lock:
                    for i, item in enumerate(self.pending_after_screenshots):
                        if item.get('step_id') == step_id and item.get('action_type') == action_type:
                            self.pending_after_screenshots.pop(i)
                            break

        with self.pending_lock:
            self.pending_after_screenshots.append(
                {'step_id': step_id, 'action_type': action_type, 'snapshot': snapshot})
        self.executor.submit(worker)

    def _save_pending_snapshot_immediately(self, step_id, action_type):
        """立即将 pending 列表里对应的 snapshot 保存为 after，不等待延迟（用于 flush）"""
        with self.pending_lock:
            idx = None
            snapshot = None
            for i, item in enumerate(self.pending_after_screenshots):
                if item.get('step_id') == step_id and item.get('action_type') == action_type:
                    idx = i
                    snapshot = item.get('snapshot')
                    break
            else:
                return None
            self.pending_after_screenshots.pop(idx)

        if snapshot is None:
            return self.take_screenshot(step_id, action_type, 'after', None)
        try:
            snap_img = snapshot.get("image") if isinstance(snapshot, dict) else snapshot
            snap_title = snapshot.get("app_title") if isinstance(snapshot, dict) else None
            if snap_img is None:
                return self.take_screenshot(step_id, action_type, 'after', None)
            filename = f"{step_id}_after_{action_type}.png"
            filepath = os.path.join(self.screenshot_dir, filename)
            snap = snap_img
            if snap.mode != "RGB":
                snap = snap.convert("RGB")
            snap.save(filepath)
            self.screenshot_count += 1
            print(f"（立即保存）截图保存: {filename}")
            self._record_screenshot_app_title(filepath, snap_title)
            return filepath
        except Exception as e:
            print(f"立即保存 snapshot 失败: {e}")
            return self.take_screenshot(step_id, action_type, 'after', None)

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
        # 鼠标移动会打断滚动和 Backspace 连击
        self.scroll_tracker.record_previous_scroll()
        self.flush_backspace_streak()
        if self.is_button_pressed and self.drag_tracker.is_dragging:
            self.drag_tracker.update_drag(x, y)

    def on_click(self, x, y, button, pressed):
        if not self.is_recording:
            return
        with self.operation_lock:
            # 点击会打断滚动和 Backspace 连击
            self.scroll_tracker.record_previous_scroll()
            self.flush_backspace_streak()
            if self.input_start_time is not None:
                self.finish_typing()

            button_name = self.platform_adapter.get_button_name(button)
            now = time.time()

            if pressed:
                self.is_button_pressed = True

                # 如果有等待中的 Timer，先取消
                if self.pending_click_timer:
                    self.pending_click_timer.cancel()
                    self.pending_click_timer = None

                # 判断当前点击是否与上一次点击构成连续（N击序列）
                is_continuous = (
                    self.last_click_time is not None and
                    self.last_click_button == button_name and
                    now - self.last_click_time < self.double_click_threshold and
                    self.drag_tracker.calculate_distance(self.last_click_pos or (0, 0), (x, y))
                    < self.double_click_distance
                )

                if is_continuous:
                    # 仍在连击窗口内，累积次数（不再重新分配 step_id，不再覆盖截图）
                    self.click_streak += 1
                    if self.click_streak >= 3:
                        # 第3次 press 时先不截图：OS 在 release 时才应用 triple-click 效果，
                        # 必须在 release 时拍快照才能捕获正确画面。暂存状态，在 release 时处理。
                        self.pending_n_click_pending_release = True
                        self.pending_n_click_data = {
                            'x': x, 'y': y, 'button': button_name,
                            'num_click': self.click_streak,
                            'before_screenshot': self.pending_click_streak['before_screenshot'],
                            'step_id': self.pending_click_streak['step_id'],
                        }
                        return
                else:
                    # 连击被打断（间隔太久 / 位置变化 / 按钮不同）
                    # 先将之前的 pending streak 确认为单/双击
                    if self.pending_click_streak is not None:
                        self._commit_click_streak()
                    # 开始新的连点计数（新的 pending streak 需要分配新的 step_id）
                    self.click_streak = 1

                    # 更新历史记录
                    self.last_click_time = now
                    self.last_click_pos = (x, y)
                    self.last_click_button = button_name

                    # 开始新的 pending streak：分配新的 step_id
                    step_id = self.get_next_step_id()
                    before = self.take_screenshot(step_id, 'click', 'before', (x, y))
                    self.pending_click_streak = {
                        'x': x, 'y': y,
                        'button': button_name,
                        'press_time': now,
                        'before_screenshot': before,
                        'step_id': step_id,
                    }

                    if not self.drag_tracker.is_dragging:
                        self.drag_tracker.start_drag(x, y, button_name)

                    # 启动 Timer，等待后续点击或超时确认
                    self.pending_click_timer = threading.Timer(
                        self.double_click_threshold, self._on_click_timer_expire)
                    self.pending_click_timer.start()

                # 连击中（is_continuous=True）：仅更新 last_click_*，不分配新 step_id，不覆盖截图

            else:
                # 鼠标释放
                self.is_button_pressed = False

                # 如果第3次点击的 press 已触发 triple-click 确认，release 时拍快照并重置
                if self.pending_n_click_pending_release:
                    data = self.pending_n_click_data
                    self.record_n_click(data['x'], data['y'], data['button'],
                                        data['num_click'], data['before_screenshot'], data['step_id'])
                    self.pending_n_click_pending_release = False
                    self.pending_n_click_data = None
                    self.click_streak = 0
                    self.pending_click_streak = None
                    if self.pending_click_timer:
                        self.pending_click_timer.cancel()
                        self.pending_click_timer = None
                    return

                if self.drag_tracker.is_dragging:
                    drag_data = self.drag_tracker.end_drag(x, y)
                    is_real_drag = self.is_real_drag_operation(drag_data)

                    if is_real_drag:
                        self.record_drag(drag_data)
                        # 拖拽处理完后，重置连点状态
                        self.click_streak = 0
                        self.pending_click_streak = None
                        if self.pending_click_timer:
                            self.pending_click_timer.cancel()
                            self.pending_click_timer = None
                    else:
                        # 误判为拖拽，退回为普通点击：启动等待确认的 Timer
                        if self.pending_click_timer:
                            self.pending_click_timer.cancel()
                        self.pending_click_timer = threading.Timer(
                            self.double_click_threshold, self._on_click_timer_expire)
                        self.pending_click_timer.start()

    def _on_click_timer_expire(self):
        """Timer 超时回调：将 pending 连点确认为单/双击 action。"""
        with self.operation_lock:
            if self.pending_click_streak is not None:
                self._commit_click_streak()
            self.pending_click_timer = None

    def _commit_click_streak(self):
        """根据 click_streak 值将当前连点提交为对应类型的 action。"""
        if self.pending_click_streak is None:
            return
        num = self.click_streak if self.click_streak > 0 else 1
        if num == 1:
            self.record_click(
                self.pending_click_streak['x'], self.pending_click_streak['y'],
                self.pending_click_streak['button'],
                self.pending_click_streak['before_screenshot'],
                self.pending_click_streak['step_id'])
        elif num == 2:
            self.record_double_click(
                self.pending_click_streak['x'], self.pending_click_streak['y'],
                self.pending_click_streak['button'],
                self.pending_click_streak['step_id'],
                self.pending_click_streak['before_screenshot'])
        else:
            self.record_n_click(
                self.pending_click_streak['x'], self.pending_click_streak['y'],
                self.pending_click_streak['button'],
                num,
                self.pending_click_streak['before_screenshot'],
                self.pending_click_streak['step_id'])
        self.click_streak = 0
        self.pending_click_streak = None

    def flush_pending_after_clicks(self):
        to_process = []
        click_action_types = {
            'click', 'double_click',
            'triple_click', 'quad_click', 'penta_click'
        }
        with self.pending_lock:
            for item in list(self.pending_after_screenshots):
                if item.get('action_type') in click_action_types:
                    to_process.append((item.get('step_id'), item.get('action_type')))
        for sid, action_type in to_process:
            saved = self._save_pending_snapshot_immediately(sid, action_type)
            if saved:
                for act in self.actions:
                    if act.get('step_id') == sid:
                        act['after_screenshot'] = saved
                        break

    def record_click(self, x, y, button_name, before_screenshot, step_id):
        action = {
            'type': 'click',
            'param': {'button': button_name, 'num_click': 1},
            'target': {'position': (x, y)},
            'before_screenshot': before_screenshot,
            'after_screenshot': None,
            'timestamp': datetime.now().isoformat(),
            'step_id': step_id
        }
        self.actions.append(action)
        self.last_click_pos = (x, y)
        print(f"点击: ({x}, {y}) - {button_name}")

        snapshot = self._capture_snapshot()
        self._schedule_after_from_snapshot(step_id, 'click', snapshot, action)

    def record_double_click(self, x, y, button_name, step_id=None, before_screenshot=None):
        """记录双击。step_id 和 before_screenshot 优先使用传入值（来自连击流程），否则从 pending_click_streak 获取。"""
        ps = self.pending_click_streak
        if step_id is None:
            step_id = ps['step_id']
        if before_screenshot is None:
            before_screenshot = ps['before_screenshot']
        before = self.take_screenshot(step_id, 'double_click', 'before', (x, y), rename_from=before_screenshot)
        action = {
            'type': 'double_click',
            'param': {'button': button_name, 'num_click': 2},
            'target': {'position': (x, y)},
            'before_screenshot': before,
            'after_screenshot': None,
            'timestamp': datetime.now().isoformat(),
            'step_id': step_id
        }
        self.actions.append(action)
        print(f"双击: ({x}, {y}) - {button_name}")

        snapshot = self._capture_snapshot()
        self._schedule_after_from_snapshot(step_id, 'double_click', snapshot, action)

    def record_n_click(self, x, y, button_name, num_click, before_screenshot, step_id=None):
        """记录 N 击（num_click >= 3）。action_type 为 triple_click / quad_click / ...，param.num_click 为实际次数。
        step_id 若不提供则自动分配（用于独立调用；连击流程中由调用方保证传入）。"""
        # 动态生成 action_type：triple_click / quad_click / ...
        click_suffix = {3: 'triple', 4: 'quad', 5: 'penta'}
        if num_click in click_suffix:
            action_type = f"{click_suffix[num_click]}_click"
        else:
            action_type = f"{num_click}_click"
        if step_id is None:
            step_id = self.get_next_step_id()
        if before_screenshot is None:
            before = self.take_screenshot(step_id, action_type, 'before', (x, y))
        else:
            # 传入的 before_screenshot 来自第一次 click，action_type 为 'click'；
            # 需要将截图文件（含局部图）重命名为正确的 N-click action_type。
            before_dir = os.path.dirname(before_screenshot)
            before_base = os.path.basename(before_screenshot)  # e.g. "s2_before_click.png"
            # 替换 action_type 部分：s2_before_click.png -> s2_before_triple_click.png
            before_fixed = before_base.replace('_before_click', f'_before_{action_type}')
            before_fixed_path = os.path.join(before_dir, before_fixed)
            if before_fixed != before_base and os.path.exists(before_screenshot):
                os.rename(before_screenshot, before_fixed_path)
                print(f"截图重命名(连击): {before_base} -> {before_fixed}")
                # 同时重命名局部图
                part_base = before_base.replace('.png', '(part).png')
                part_fixed = before_fixed.replace('.png', '(part).png')
                part_old = os.path.join(before_dir, part_base)
                part_new = os.path.join(before_dir, part_fixed)
                if os.path.exists(part_old):
                    os.rename(part_old, part_new)
                    print(f"截图重命名(连击局部): {part_base} -> {part_fixed}")
            else:
                before_fixed_path = before_screenshot
            before = before_fixed_path
        action = {
            'type': action_type,
            'param': {'button': button_name, 'num_click': num_click},
            'target': {'position': (x, y)},
            'before_screenshot': before,
            'after_screenshot': None,
            'timestamp': datetime.now().isoformat(),
            'step_id': step_id
        }
        self.actions.append(action)
        print(f"{num_click}击: ({x}, {y}) - {button_name}")

        snapshot = self._capture_snapshot()

        # N≥3 的连击：立即同步保存 after 截图，不再走 2 秒延迟线程池。
        # 原因：连击结束后下一个操作（通常是 Ctrl+C/X/V 等）几乎立即触发，
        # 会生成同路径的截图覆盖 pending 的延迟任务。
        if num_click >= 3:
            self._immediate_save_after_snapshot(step_id, action_type, snapshot, action)
        else:
            self._schedule_after_from_snapshot(step_id, action_type, snapshot, action)

    def _immediate_save_after_snapshot(self, step_id, action_type, snapshot, action_ref):
        """立即将 snapshot 保存为 after 截图，不等待延迟（专用于 N≥3 连击）。"""
        try:
            snap_img = snapshot.get("image") if isinstance(snapshot, dict) else snapshot
            snap_title = snapshot.get("app_title") if isinstance(snapshot, dict) else None
            if snap_img is None:
                after_path = self.take_screenshot(step_id, action_type, 'after', None)
            else:
                filename = f"{step_id}_after_{action_type}.png"
                filepath = os.path.join(self.screenshot_dir, filename)
                snap = snap_img
                if snap.mode != "RGB":
                    snap = snap.convert("RGB")
                snap.save(filepath)
                self.screenshot_count += 1
                print(f"（立即保存）截图保存: {filename}")
                self._record_screenshot_app_title(filepath, snap_title)
                after_path = filepath
            if action_ref is not None:
                action_ref['after_screenshot'] = after_path
        except Exception as e:
            print(f"立即保存 after 截图失败: {e}")
            after_path = self.take_screenshot(step_id, action_type, 'after', None)
            if action_ref is not None:
                action_ref['after_screenshot'] = after_path

    def calculate_drag_direction(self, start_pos, end_pos):
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360
        return angle

    def record_drag(self, drag_data):
        if not self.pending_click_streak:
            return
        sx, sy = drag_data['start_pos']
        ex, ey = drag_data['end_pos']
        step_id = self.pending_click_streak['step_id']
        old_before = self.pending_click_streak['before_screenshot']
        angle = self.calculate_drag_direction((sx, sy), (ex, ey))
        before = self.take_screenshot(step_id, 'drag', 'before', (sx, sy), rename_from=old_before)
        action = {
            'type': 'drag_to',
            'param': {'button': drag_data['button']},
            'target': {'Begin': {'position': (sx, sy)}, 'End': {'position': (ex, ey)},
                       'describe': {'angle': angle, 'distance': drag_data['distance']}},
            'before_screenshot': before,
            'after_screenshot': None,
            'timestamp': datetime.now().isoformat(),
            'drag_data': drag_data,
            'step_id': step_id
        }
        self.actions.append(action)
        print(f"拖拽: 从 ({sx},{sy}) 到 ({ex},{ey}) 距离: {drag_data['distance']:.2f}")

        snapshot = self._capture_snapshot()
        self._schedule_after_from_snapshot(step_id, 'drag', snapshot, action)

    def on_scroll(self, x, y, dx, dy):
        if not self.is_recording:
            return
        with self.operation_lock:
            # 滚动会打断 Backspace 连击
            self.flush_backspace_streak()
            if self.input_start_time is not None:
                self.finish_typing()

            self.scroll_tracker.add_scroll(x, y, dx, dy)

    def record_scroll(self, x, y, scroll_type, before_screenshot=None, step_id=None):
        if step_id is None:
            step_id = self.get_next_step_id()
        if before_screenshot is None:
            before_screenshot = self.take_screenshot(step_id, 'scroll', 'before', (x, y))
        action = {
            'type': 'scroll',
            'param': {'type': scroll_type},
            'target': {'position': (x, y)},
            'before_screenshot': before_screenshot,
            'after_screenshot': None,
            'timestamp': datetime.now().isoformat(),
            'step_id': step_id
        }
        self.actions.append(action)
        print(f"滚动: {scroll_type} at ({x},{y})")

        snapshot = self._capture_snapshot()
        self._schedule_after_from_snapshot(step_id, 'scroll', snapshot, action)

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

    def record_press_with_before(self, key_name, before_screenshot, step_id, position):
        modifiers = list(self.modifier_tracker.pressed_modifiers) if self.modifier_tracker.pressed_modifiers else []
        press_list = modifiers + [key_name] if key_name not in modifiers else modifiers
        action = {
            'type': 'press',
            'param': {'press_list': press_list},
            'target': {},
            'before_screenshot': before_screenshot,
            'after_screenshot': None,
            'timestamp': datetime.now().isoformat(),
            'step_id': step_id
        }
        self.actions.append(action)
        print(f"按键: {'+'.join(press_list)}")

        snapshot = self._capture_snapshot()
        self._schedule_after_from_snapshot(step_id, 'press', snapshot, action)

    def flush_backspace_streak(self):
        """将正在累积的 Backspace 连击刷新为一次 press 操作。"""
        if self.pending_backspace_action and self.backspace_streak > 0:
            # 确保参数结构存在
            self.pending_backspace_action.setdefault('param', {}).setdefault('press_list', ['backspace'])
            self.pending_backspace_action['param']['press_count'] = self.backspace_streak

            action = self.pending_backspace_action
            self.actions.append(action)

            # 安排 after 截图
            snapshot = self._capture_snapshot()
            self._schedule_after_from_snapshot(action['step_id'], 'press', snapshot, action)

        self.pending_backspace_action = None
        self.backspace_streak = 0


    def record_special_key_press_with_before(self, key_name, before_screenshot, step_id):
        action = {
            'type': 'press',
            'param': {'press_list': [key_name]},
            'target': {},
            'before_screenshot': before_screenshot,
            'after_screenshot': None,
            'timestamp': datetime.now().isoformat(),
            'step_id': step_id,
            'is_special_key': True
        }
        self.actions.append(action)
        print(f"特殊键按下: {key_name}")

        snapshot = self._capture_snapshot()
        self._schedule_after_from_snapshot(step_id, 'special_key', snapshot, action)

    def start_input_session(self, trigger_key=None):
        has_non_shift_modifiers = any(mod not in ['shift'] for mod in self.modifier_tracker.pressed_modifiers)
        if has_non_shift_modifiers:
            return

        if self.input_start_time is not None:
            self.flush_pending_after_clicks()
            return

        if self.pending_click_streak is not None:
            try:
                if self.pending_click_timer:
                    self.pending_click_timer.cancel()
                    self.pending_click_timer = None
            except:
                pass

            self._commit_click_streak()

        self.input_start_time = time.time()
        self.current_input_session_id = len(self.actions)
        step_id = self.get_next_step_id()

        before = None
        if self.actions:
            last_action = self.actions[-1]
            if last_action['type'] == 'click' and last_action.get('after_screenshot'):
                before = last_action['after_screenshot']
        if before is None:
            before_pos = self.last_click_pos if self.last_click_pos else None
            before = self.take_screenshot(step_id, 'typing', 'before', before_pos)

        action = {
            'type': 'typing_start',
            'position': self.last_click_pos,
            'before_screenshot': before,
            'text': '',
            'after_screenshot': None,
            'trigger_key': trigger_key,
            '_session_id': self.current_input_session_id,
            'timestamp': datetime.now().isoformat(),
            'step_id': step_id
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
                if key_name in ['win', 'cmd'] and not self.modifier_tracker.has_modifiers():
                    if self.input_start_time is not None:
                        self.finish_typing()

                    cx, cy = pyautogui.position()
                    step_id = self.get_next_step_id()
                    before = self.take_screenshot(step_id, 'special_key', 'before', (cx, cy))
                    self.record_special_key_press_with_before(key_name, before, step_id)
                    self.modifier_tracker.press(key_name)
                    return
                if self.modifier_tracker.is_modifier(key_name):
                    self.modifier_tracker.press(key_name)
                    return

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
                                                  not is_char_key and
                                                  key not in [Key.space, Key.backspace, Key.tab, Key.enter]
                                          ) or (
                                                  is_char_key and has_non_shift_modifiers
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
                    before = self.take_screenshot(step_id, 'press', 'before', (cx, cy))
                    self.record_press_with_before(key_name, before, step_id, (cx, cy))
                    return

                if needs_new_session:
                    self.start_input_session(key_name)

                if is_char_key:
                    if has_non_shift_modifiers:
                        cx, cy = pyautogui.position()
                        step_id = self.get_next_step_id()
                        before = self.take_screenshot(step_id, 'press', 'before', (cx, cy))
                        self.record_press_with_before(key_name, before, step_id, (cx, cy))
                    else:
                        if not self.modifier_tracker.pressed_modifiers or 'shift' in self.modifier_tracker.pressed_modifiers:
                            if self.input_start_time is None:
                                self.start_input_session(key_name)
                            self.current_input += char_value
                        else:
                            cx, cy = pyautogui.position()
                            step_id = self.get_next_step_id()
                            before = self.take_screenshot(step_id, 'press', 'before', (cx, cy))
                            self.record_press_with_before(key_name, before, step_id, (cx, cy))
                elif key == Key.space:
                    if has_non_shift_modifiers:
                        cx, cy = pyautogui.position()
                        step_id = self.get_next_step_id()
                        before = self.take_screenshot(step_id, 'press', 'before', (cx, cy))
                        self.record_press_with_before('space', before, step_id, (cx, cy))
                    else:
                        if self.input_start_time is not None:
                            self.current_input += ' '
                        else:
                            cx, cy = pyautogui.position()
                            step_id = self.get_next_step_id()
                            before = self.take_screenshot(step_id, 'press', 'before', (cx, cy))
                            self.record_press_with_before('space', before, step_id, (cx, cy))
                elif key == Key.enter:
                    if has_non_shift_modifiers:
                        cx, cy = pyautogui.position()
                        step_id = self.get_next_step_id()
                        before = self.take_screenshot(step_id, 'press', 'before', (cx, cy))
                        self.record_press_with_before('enter', before, step_id, (cx, cy))
                    else:
                        if self.input_start_time is not None:
                            finish_info = self.finish_typing()
                            mx, my = pyautogui.position()
                            if finish_info and finish_info.get('after_screenshot'):
                                before = finish_info['after_screenshot']
                            else:
                                step_id_tmp = self.get_next_step_id()
                                before = self.take_screenshot(step_id_tmp, 'press', 'before', (mx, my))
                            step_id = self.get_next_step_id()
                            self.record_press_with_before('enter', before, step_id, (mx, my))
                        else:
                            mx, my = pyautogui.position()
                            step_id = self.get_next_step_id()
                            before = self.take_screenshot(step_id, 'press', 'before', (mx, my))
                            self.record_press_with_before('enter', before, step_id, (mx, my))
                    return
                elif key == Key.backspace:
                    # Backspace 特殊处理：在没有 typing 时，连续 Backspace 视为一次原子 press 操作
                    if has_non_shift_modifiers:
                        # 带修饰键的 Backspace 仍然按单次 press 记录
                        cx, cy = pyautogui.position()
                        step_id = self.get_next_step_id()
                        before = self.take_screenshot(step_id, 'press', 'before', (cx, cy))
                        self.record_press_with_before('backspace', before, step_id, (cx, cy))
                    else:
                        if self.input_start_time is not None:
                            # 正在输入文本时，Backspace 仅作为编辑行为
                            self.current_input = self.current_input[:-1]
                        else:
                            # 不在输入状态：开始或继续 Backspace 连击，不立即写入 actions
                            if self.pending_backspace_action is None:
                                cx, cy = pyautogui.position()
                                step_id = self.get_next_step_id()
                                before = self.take_screenshot(step_id, 'press', 'before', (cx, cy))
                                self.pending_backspace_action = {
                                    'type': 'press',
                                    'param': {'press_list': ['backspace']},
                                    'target': {},
                                    'before_screenshot': before,
                                    'after_screenshot': None,
                                    'timestamp': datetime.now().isoformat(),
                                    'step_id': step_id
                                }
                                self.backspace_streak = 1
                            else:
                                self.backspace_streak += 1
                elif key == Key.tab:
                    if has_non_shift_modifiers:
                        cx, cy = pyautogui.position()
                        step_id = self.get_next_step_id()
                        before = self.take_screenshot(step_id, 'press', 'before', (cx, cy))
                        self.record_press_with_before('tab', before, step_id, (cx, cy))
                    else:
                        if self.input_start_time is not None:
                            self.current_input += '	'
                        else:
                            cx, cy = pyautogui.position()
                            step_id = self.get_next_step_id()
                            before = self.take_screenshot(step_id, 'press', 'before', (cx, cy))
                            self.record_press_with_before('tab', before, step_id, (cx, cy))
                else:
                    cx, cy = pyautogui.position()
                    step_id = self.get_next_step_id()
                    before = self.take_screenshot(step_id, 'press', 'before', (cx, cy))
                    self.record_press_with_before(key_name, before, step_id, (cx, cy))

            except AttributeError:
                if self.input_start_time is not None:
                    self.finish_typing()

                key_name = self.get_key_name(key)
                cx, cy = pyautogui.position()
                step_id = self.get_next_step_id()
                before = self.take_screenshot(step_id, 'press', 'before', (cx, cy))
                self.record_press_with_before(key_name, before, step_id, (cx, cy))

    def on_release(self, key):
        if not self.is_recording:
            return
        key_name = self.get_key_name(key)
        if self.modifier_tracker.is_modifier(key_name):
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
                self.current_input_session_id = None
                return None

            if not self.current_input:
                del self.actions[start_index]
            else:
                step_id = self.actions[start_index]['step_id']
                after = self.take_screenshot(step_id, 'typing', 'after', None)

                self.actions[start_index]['type'] = 'typing'
                self.actions[start_index]['param'] = {'text': self.current_input}
                self.actions[start_index]['target'] = {}
                self.actions[start_index]['after_screenshot'] = after
                self.actions[start_index]['duration'] = time.time() - self.input_start_time

                result = {'step_id': step_id, 'after_screenshot': after}

        self.current_input = ""
        self.input_start_time = None
        self.current_input_session_id = None
        return result

    def _get_screenshot_time_str(self, filepath):
        """
        根据截图文件的修改时间生成时间戳字符串，格式为 "%Y-%m-%d %H:%M:%S"
        """
        if not filepath:
            return None
        try:
            ts = os.path.getmtime(filepath)
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        except KeyboardInterrupt:
            print(f"警告: 读取截图时间被中断，已跳过: {filepath}")
            return None
        except Exception:
            return None

    def _safe_relpath(self, filepath):
        if not filepath:
            return None
        try:
            return os.path.relpath(os.path.abspath(filepath), os.path.abspath(self.session_dir))
        except Exception:
            return os.path.basename(filepath)

    def _get_screenshot_report_data(self, filepath):
        if not filepath:
            return {'relative_path': None, 'time': None, 'app_title': None}

        abs_path = os.path.abspath(filepath)
        return {
            'relative_path': self._safe_relpath(abs_path),
            'time': self._get_screenshot_time_str(abs_path),
            'app_title': self.screenshot_app_titles.get(abs_path)
        }

    def save_report(self):
        # 确保将未刷新的 Backspace 连击写入记录
        self.flush_backspace_streak()
        if self.input_start_time is not None:
            self.finish_typing()
        os.makedirs(self.session_dir, exist_ok=True)
        try:
            screen_w, screen_h = pyautogui.size()
        except Exception:
            screen_w, screen_h = ("unknown", "unknown")
        env_info = {'os': self.platform, 'screen': f"{screen_w}x{screen_h}", 'url': self.task_info.get('url', ''),
                    'locale': self.task_info.get('locale', 'en_US')}
        steps = []
        for i, action in enumerate(self.actions):
            before_info = self._get_screenshot_report_data(action.get('before_screenshot'))
            after_info = self._get_screenshot_report_data(action.get('after_screenshot'))

            before = before_info['relative_path']
            before_part = None
            if before:
                before_dir, before_file = os.path.split(before)
                base_name, ext = os.path.splitext(before_file)
                part_file = f"{base_name}(part){ext}"
                part_abs = os.path.join(self.session_dir, before_dir, part_file)
                if os.path.exists(part_abs):
                    before_part = self._safe_relpath(part_abs)

            # 构造 action 结构，补充 nl_position 等字段（如果有 position）
            target = action.get('target', {}) or {}
            if isinstance(target, dict) and 'position' in target and 'nl_position' not in target:
                target = dict(target)
                target['nl_position'] = ""
            action_struct = {'type': action.get('type', 'unknown'), 'target': target}
            if 'param' in action:
                action_struct['param'] = action['param']

            step = {'step_id': action.get('step_id', f"s{i + 1}"), 'step_goal': '',
                    'now_state': {'screenshot_path_before': before,
                                  'screenshot_path_before_part': before_part,
                                  'screenshot_path_after': after_info['relative_path'],
                                  'screenshot_time_before': before_info['time'],
                                  'screeenshot_time_after': after_info['time'],
                                  'app_title_before': before_info['app_title'],
                                  'app_title_after': after_info['app_title']},
                    'action_preconditions': [],
                    'action': action_struct,
                    'action_before_state': "",
                    'action_after_effects': [],
                    'nl_explanation': ''}
            steps.append(step)
        report = {'task_id': self.task_info.get('task_id', 'uuid'),
                  'task_category': self.task_info.get('task_category', ''), 'task_title': '',
                  'instruction': self.instruction, 'app': '', 'env': env_info, 'steps': steps}
        json_report_path = os.path.join(self.session_dir, 'report.json')
        json_saved = False
        txt_saved = False

        try:
            with open(json_report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            json_saved = True
        except Exception as e:
            print(f"保存 report.json 失败: {e}")

        try:
            self.save_readable_report(report)
            txt_saved = True
        except Exception as e:
            print(f"保存 report.txt 失败: {e}")

        if json_saved and txt_saved:
            print(f"记录已保存: {len(self.actions)} 个操作, {self.screenshot_count} 张截图")
        elif json_saved or txt_saved:
            print(f"记录已部分保存: {len(self.actions)} 个操作, {self.screenshot_count} 张截图")
        else:
            print(f"记录保存失败: {len(self.actions)} 个操作, {self.screenshot_count} 张截图")
        print(f"会话目录: {self.session_dir}")
        print(f"报告状态: report.json={'成功' if json_saved else '失败'}, report.txt={'成功' if txt_saved else '失败'}")

    def save_readable_report(self, report):
        txt_report_path = os.path.join(self.session_dir, 'report.txt')
        os.makedirs(self.session_dir, exist_ok=True)
        with open(txt_report_path, 'w', encoding='utf-8') as f:
            f.write("操作记录报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"任务ID: {report.get('task_id', '')}\n")
            f.write(f"任务类别: {report.get('task_category', '')}\n")
            f.write(f"任务标题: {report.get('task_title', '')}\n")
            f.write(f"指令: {report.get('instruction', '')}\n")
            f.write(f"应用: {report.get('app', '')}\n")
            f.write(f"环境: {json.dumps(report.get('env', {}), ensure_ascii=False, default=str)}\n")
            f.write(f"会话目录: {self.session_dir}\n")
            f.write(f"总操作数: {len(self.actions)}\n")
            f.write(f"总截图数: {self.screenshot_count}\n")
            f.write("=" * 50 + "\n\n")

            f.write("步骤详情:\n")
            f.write("-" * 30 + "\n")
            for i, step in enumerate(report.get('steps', []), 1):
                f.write(f"步骤 #{i}:\n")
                f.write(f"  步骤ID: {step.get('step_id', '')}\n")
                f.write(f"  步骤目标: {step.get('step_goal', '')}\n")
                f.write(f"  当前状态:\n")
                now_state = step.get('now_state', {}) or {}
                before = now_state.get('screenshot_path_before')
                before_part = now_state.get('screenshot_path_before_part')
                after = now_state.get('screenshot_path_after')
                if before_part:
                    f.write(f"    操作前截图(局部): {before_part}\n")
                    f.write(f"    操作前截图(全图): {before}\n")
                else:
                    f.write(f"    操作前截图: {before}\n")
                f.write(f"    操作后截图: {after}\n")
                f.write(f"  动作前提条件: {json.dumps(step.get('action_preconditions', []), ensure_ascii=False, default=str)}\n")
                f.write(f"  动作: {json.dumps(step.get('action', {}), ensure_ascii=False, default=str)}\n")
                # 可选动作前状态
                if 'action_before_state' in step:
                    f.write(
                        f"  动作前状态: {json.dumps(step.get('action_before_state', ''), ensure_ascii=False, default=str)}\n"
                    )
                # 兼容 action_after_effects / action_effects 两种字段名
                effects = step.get('action_after_effects', step.get('action_effects', []))
                f.write(f"  动作后效果: {json.dumps(effects, ensure_ascii=False, default=str)}\n")
                f.write(f"  自然语言解释: {step.get('nl_explanation', '')}\n")
                f.write("\n")


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
