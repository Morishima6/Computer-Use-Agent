import json
import math
import os
import platform
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
        print("停止记录，等待所有异步截图完成...")
        self.wait_for_pending_screenshots()
        self.save_report()
        self.executor.shutdown(wait=True)

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
                print(f"警告: {len(self.pending_after_screenshots)} 个 after 截图仍未完成")

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
                if self.pending_click_timer:
                    self.pending_click_timer.cancel()
                is_double = False
                if (self.last_click_time is not None and
                        self.last_click_button == button_name and
                        now - self.last_click_time < self.double_click_threshold and
                        self.drag_tracker.calculate_distance(self.last_click_pos or (0, 0),
                                                             (x, y)) < self.double_click_distance):
                    is_double = True
                self.last_click_time = now
                self.last_click_pos = (x, y)
                self.last_click_button = button_name
                if not self.drag_tracker.is_dragging:
                    self.drag_tracker.start_drag(x, y, button_name)
                if is_double and self.pending_click:
                    self.record_double_click(x, y, button_name)
                    self.pending_click = None
                else:
                    self.pending_click = {'x': x, 'y': y, 'button': button_name, 'press_time': now,
                                          'before_screenshot': None}
                    step_id = self.get_next_step_id()
                    before = self.take_screenshot(step_id, 'click', 'before', (x, y))
                    self.pending_click['before_screenshot'] = before
                    self.pending_click['step_id'] = step_id
            else:
                self.is_button_pressed = False
                if self.drag_tracker.is_dragging:
                    drag_data = self.drag_tracker.end_drag(x, y)
                    is_real_drag = self.is_real_drag_operation(drag_data)

                    if is_real_drag:
                        self.record_drag(drag_data)
                        self.pending_click = None
                    else:
                        if self.pending_click:
                            self.pending_click_timer = threading.Timer(self.double_click_threshold,
                                                                       self.confirm_single_click)
                            self.pending_click_timer.start()

    def confirm_single_click(self):
        if self.pending_click:
            click_data = self.pending_click
            self.record_click(click_data['x'], click_data['y'], click_data['button'], click_data['before_screenshot'],
                              click_data['step_id'])
            self.pending_click = None

    def flush_pending_after_clicks(self):
        to_process = []
        with self.pending_lock:
            for item in list(self.pending_after_screenshots):
                if item.get('action_type') == 'click':
                    to_process.append(item.get('step_id'))
        for sid in to_process:
            saved = self._save_pending_snapshot_immediately(sid, 'click')
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

    def record_double_click(self, x, y, button_name):
        if not self.pending_click:
            return
        step_id = self.pending_click['step_id']
        old_before = self.pending_click['before_screenshot']
        before = self.take_screenshot(step_id, 'double_click', 'before', (x, y), rename_from=old_before)
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

    def calculate_drag_direction(self, start_pos, end_pos):
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360
        return angle

    def record_drag(self, drag_data):
        if not self.pending_click:
            return
        sx, sy = drag_data['start_pos']
        ex, ey = drag_data['end_pos']
        step_id = self.pending_click['step_id']
        old_before = self.pending_click['before_screenshot']
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

        if self.pending_click:
            try:
                if self.pending_click_timer:
                    self.pending_click_timer.cancel()
                    self.pending_click_timer = None
            except:
                pass

            self.confirm_single_click()

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
        except Exception:
            return None

    def save_report(self):
        # 确保将未刷新的 Backspace 连击写入记录
        self.flush_backspace_streak()
        if self.input_start_time is not None:
            self.finish_typing()
        screen_w, screen_h = pyautogui.size()
        env_info = {'os': self.platform, 'screen': f"{screen_w}x{screen_h}", 'url': self.task_info.get('url', ''),
                    'locale': self.task_info.get('locale', 'en_US')}
        steps = []
        for i, action in enumerate(self.actions):
            before = None
            before_part = None
            after = None
            screenshot_time_before = None
            screenshot_time_after = None
            app_title_before = None
            app_title_after = None
            if action.get('before_screenshot'):
                before = os.path.relpath(action['before_screenshot'], self.session_dir)
                # 如果存在对应的局部(part)截图，则一并记录
                before_dir, before_file = os.path.split(before)
                base_name, ext = os.path.splitext(before_file)
                part_file = f"{base_name}(part){ext}"
                part_abs = os.path.join(self.session_dir, before_dir, part_file)
                if os.path.exists(part_abs):
                    before_part = os.path.relpath(part_abs, self.session_dir)
                before_abs = os.path.join(self.session_dir, before)
                screenshot_time_before = self._get_screenshot_time_str(before_abs)
                app_title_before = self.screenshot_app_titles.get(os.path.abspath(before_abs))
            if action.get('after_screenshot'):
                after = os.path.relpath(action['after_screenshot'], self.session_dir)
                after_abs = os.path.join(self.session_dir, after)
                screenshot_time_after = self._get_screenshot_time_str(after_abs)
                app_title_after = self.screenshot_app_titles.get(os.path.abspath(after_abs))

            # 构造 action 结构，补充 nl_position 等字段（如果有 position）
            target = action.get('target', {}) or {}
            if isinstance(target, dict) and 'position' in target and 'nl_position' not in target:
                target = dict(target)
                target['nl_position'] = ""
            action_struct = {'type': action['type'], 'target': target}
            if 'param' in action:
                action_struct['param'] = action['param']

            step = {'step_id': action.get('step_id', f"s{i + 1}"), 'step_goal': '',
                    'now_state': {'screenshot_path_before': before,
                                  'screenshot_path_before_part': before_part,
                                  'screenshot_path_after': after,
                                  'screenshot_time_before': screenshot_time_before,
                                  'screeenshot_time_after': screenshot_time_after,
                                  'app_title_before': app_title_before,
                                  'app_title_after': app_title_after},
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
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self.save_readable_report(report)
        print(f"记录已保存: {len(self.actions)} 个操作, {self.screenshot_count} 张截图")
        print(f"会话目录: {self.session_dir}")

    def save_readable_report(self, report):
        txt_report_path = os.path.join(self.session_dir, 'report.txt')
        with open(txt_report_path, 'w', encoding='utf-8') as f:
            f.write("操作记录报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"任务ID: {report['task_id']}\n")
            f.write(f"任务类别: {report['task_category']}\n")
            f.write(f"任务标题: {report['task_title']}\n")
            f.write(f"指令: {report['instruction']}\n")
            f.write(f"应用: {report['app']}\n")
            f.write(f"环境: {json.dumps(report['env'], ensure_ascii=False)}\n")
            f.write(f"会话目录: {self.session_dir}\n")
            f.write(f"总操作数: {len(self.actions)}\n")
            f.write(f"总截图数: {self.screenshot_count}\n")
            f.write("=" * 50 + "\n\n")

            f.write("步骤详情:\n")
            f.write("-" * 30 + "\n")
            for i, step in enumerate(report['steps'], 1):
                f.write(f"步骤 #{i}:\n")
                f.write(f"  步骤ID: {step['step_id']}\n")
                f.write(f"  步骤目标: {step['step_goal']}\n")
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
                f.write(f"  动作前提条件: {json.dumps(step['action_preconditions'], ensure_ascii=False)}\n")
                f.write(f"  动作: {json.dumps(step['action'], ensure_ascii=False)}\n")
                # 可选动作前状态
                if 'action_before_state' in step:
                    f.write(
                        f"  动作前状态: {json.dumps(step.get('action_before_state', ''), ensure_ascii=False)}\n"
                    )
                # 兼容 action_after_effects / action_effects 两种字段名
                effects = step.get('action_after_effects', step.get('action_effects', []))
                f.write(f"  动作后效果: {json.dumps(effects, ensure_ascii=False)}\n")
                f.write(f"  自然语言解释: {step['nl_explanation']}\n")
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
