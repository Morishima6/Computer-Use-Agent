#!/usr/bin/env python3
"""
OSWorld任务配置脚本
读取JSON文件的config字段，在虚拟机上提前完成配置

用法:
    python osworld_config.py <json_file>         # 处理单个文件
    python osworld_config.py <directory>         # 处理目录下的所有JSON
    python osworld_config.py -d <json_file>      # 测试运行(不实际执行)
"""

import json
import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

# 尝试导入requests，如果不存在则使用urllib
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    try:
        from urllib.request import urlretrieve
        from urllib.error import URLError
    except ImportError:
        pass


class OSWorldConfigurator:
    """OSWorld任务配置器，负责执行config字段中的配置操作"""
    
    def __init__(self, json_path: str, dry_run: bool = False, verbose: bool = False, continue_on_error: bool = False):
        """
        初始化配置器
        
        Args:
            json_path: JSON文件路径
            dry_run: 是否只打印操作而不执行
            verbose: 是否输出详细信息
            continue_on_error: 遇到错误是否继续执行
        """
        self.json_path = json_path
        self.dry_run = dry_run
        self.verbose = verbose
        self.continue_on_error = continue_on_error
        self.config: List[Dict[str, Any]] = []
        self.task_id: str = ""
        self.snapshot: str = ""
        
    def load_config(self) -> bool:
        """加载并解析JSON文件"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.task_id = data.get('id', '')
            self.snapshot = data.get('snapshot', '')
            self.config = data.get('config', [])
            
            print(f"Task ID: {self.task_id}")
            print(f"Snapshot: {self.snapshot}")
            print(f"Config operations: {len(self.config)}")
                
            return True
        except FileNotFoundError:
            print(f"[ERROR] File not found: {self.json_path}", file=sys.stderr)
            return False
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON: {e}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            return False
    
    def execute_config(self) -> bool:
        """执行所有配置操作"""
        if not self.config:
            print("[WARN] No config operations to execute")
            return True
            
        print(f"\nExecuting {len(self.config)} config operations...")
        print("-" * 50)
            
        success = True
        has_error = False
        
        for i, operation in enumerate(self.config):
            op_type = operation.get('type', 'unknown')
            params = operation.get('parameters', {})
            
            print(f"\n[{i+1}/{len(self.config)}] {op_type}")
            
            if self.verbose:
                print(f"    Parameters: {params}")
            
            if self.dry_run:
                print(f"    [DRY RUN] Would execute this operation")
                continue
            
            # 执行对应的操作
            handler = getattr(self, f'handle_{op_type}', None)
            if handler:
                try:
                    result = handler(params)
                    if result:
                        print(f"    [OK]")
                    else:
                        print(f"    [FAIL]")
                        success = False
                        has_error = True
                        if not self.continue_on_error:
                            print("    [INFO] Stopping due to error. Use --continue-on-error to continue.")
                            break
                except Exception as e:
                    print(f"    [ERROR] {e}", file=sys.stderr)
                    success = False
                    has_error = True
                    if not self.continue_on_error:
                        print("    [INFO] Stopping due to error. Use --continue-on-error to continue.")
                        break
            else:
                print(f"    [WARN] Unknown operation type: {op_type}")
                
        print("-" * 50)
        
        if has_error and not success:
            print("[WARN] Some operations failed. Check logs above.")
            
        return success
    
    # ==================== 下载操作 ====================
    
    def _download_with_requests(self, url: str, path: str) -> bool:
        """使用requests下载文件"""
        try:
            response = requests.get(url, timeout=120, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0 and self.verbose:
                            progress = (downloaded / total_size) * 100
                            print(f"    Downloading: {progress:.1f}%", end='\r')
            
            if self.verbose:
                print(f"    Download complete: {path}")
            return True
            
        except Exception as e:
            print(f"    [ERROR] Download failed: {e}", file=sys.stderr)
            return False
    
    def _download_with_urllib(self, url: str, path: str) -> bool:
        """使用urllib下载文件（备选方案）"""
        try:
            urlretrieve(url, path)
            return True
        except Exception as e:
            print(f"    [ERROR] Download failed: {e}", file=sys.stderr)
            return False
    
    def handle_download(self, params: Dict[str, Any]) -> bool:
        """处理download操作 - 下载文件"""
        files = params.get('files', [])
        
        if not files:
            print("    [WARN] No files to download")
            return True
        
        all_success = True
            
        for file_info in files:
            url = file_info.get('url')
            path = file_info.get('path')
            
            if not url or not path:
                print("    [WARN] Missing url or path, skipping")
                continue
            
            print(f"    URL: {url}")
            print(f"    Path: {path}")
            
            # 确保目标目录存在
            try:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"    [ERROR] Cannot create directory: {e}", file=sys.stderr)
                all_success = False
                continue
            
            # 下载文件
            if HAS_REQUESTS:
                if not self._download_with_requests(url, path):
                    all_success = False
            else:
                if not self._download_with_urllib(url, path):
                    all_success = False
                    
        return all_success
    
    # ==================== 文件操作 ====================
    
    def handle_open(self, params: Dict[str, Any]) -> bool:
        """处理open操作 - 用默认应用打开文件"""
        path = params.get('path')
        
        if not path:
            print("    [ERROR] Missing path parameter")
            return False
        
        print(f"    Path: {path}")
        
        # 检查文件是否存在
        if not Path(path).exists():
            print(f"    [ERROR] File does not exist: {path}")
            return False
        
        # 检查是否为桌面环境
        if os.environ.get('DISPLAY') is None:
            print(f"    [WARN] No DISPLAY set, skipping open operation")
            return True
        
        try:
            # 使用xdg-open (Linux)
            subprocess.run(['xdg-open', path], 
                         check=True, 
                         timeout=10,
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
            print(f"    [OK] Opened with default application")
            return True
        except subprocess.CalledProcessError as e:
            print(f"    [ERROR] Failed to open file: {e}")
            return False
        except FileNotFoundError:
            print(f"    [WARN] xdg-open not found, trying alternative methods")
            try:
                subprocess.run(['gnome-open', path], 
                             check=True, 
                             timeout=10,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
                return True
            except:
                print(f"    [WARN] No suitable command to open file")
                return False
        except Exception as e:
            print(f"    [ERROR] {e}")
            return False
    
    # ==================== 应用启动操作 ====================
    
    def handle_launch(self, params: Dict[str, Any]) -> bool:
        """处理launch操作 - 启动应用程序"""
        command = params.get('command', [])
        
        if not command:
            print("    [ERROR] Missing command parameter")
            return False
        
        print(f"    Command: {' '.join(command)}")
        
        # 检查是否为桌面环境
        if os.environ.get('DISPLAY') is None:
            print(f"    [WARN] No DISPLAY set, application may not launch correctly")
        
        try:
            # 使用subprocess启动应用，不等待完成
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"    [OK] Launched (PID: {process.pid})")
            return True
        except Exception as e:
            print(f"    [ERROR] Failed to launch: {e}")
            return False
    
    def handle_activate_window(self, params: Dict[str, Any]) -> bool:
        """处理activate_window操作 - 激活窗口"""
        window_name = params.get('window_name')
        
        if not window_name:
            print("    [ERROR] Missing window_name parameter")
            return False
        
        print(f"    Window: {window_name}")
        
        # 检查xdotool是否可用
        try:
            subprocess.run(['which', 'xdotool'], 
                          check=True, 
                          capture_output=True)
        except subprocess.CalledProcessError:
            print(f"    [WARN] xdotool not installed, skipping activate_window")
            return True
        
        try:
            # 使用xdotool激活窗口
            result = subprocess.run(
                ['xdotool', 'search', '--name', window_name, 'activate'],
                check=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            print(f"    [OK] Window activated")
            return True
        except subprocess.CalledProcessError as e:
            print(f"    [WARN] Failed to activate window: {e.stderr}")
            return True  # 不算失败，继续执行
        except subprocess.TimeoutExpired:
            print(f"    [WARN] Window activation timed out")
            return True
        except Exception as e:
            print(f"    [ERROR] {e}")
            return False
    
    # ==================== 命令执行操作 ====================
    
    def handle_execute(self, params: Dict[str, Any]) -> bool:
        """处理execute操作 - 执行命令"""
        command = params.get('command', [])
        
        if not command:
            print("    [ERROR] Missing command parameter")
            return False
        
        # 获取屏幕尺寸用于替换占位符
        screen_width = 1920  # 默认值
        screen_height = 1080  # 默认值
        try:
            import subprocess
            result = subprocess.run(
                ['xdotool', 'getdisplaygeometry'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split()
                if len(parts) >= 2:
                    screen_width = int(parts[0])
                    screen_height = int(parts[1])
        except Exception:
            pass  # 使用默认值
        
        screen_width_half = screen_width // 2
        screen_height_half = screen_height // 2
        
        # 替换命令中的占位符
        processed_command = []
        for cmd_part in command:
            cmd_str = str(cmd_part)
            cmd_str = cmd_str.replace('{SCREEN_WIDTH}', str(screen_width))
            cmd_str = cmd_str.replace('{SCREEN_HEIGHT}', str(screen_height))
            cmd_str = cmd_str.replace('{SCREEN_WIDTH_HALF}', str(screen_width_half))
            cmd_str = cmd_str.replace('{SCREEN_HEIGHT_HALF}', str(screen_height_half))
            processed_command.append(cmd_str)
        
        print(f"    Command: {' '.join(str(c) for c in processed_command)}")
        
        try:
            # command是列表 [python, -c, "code"] 或 [cmd, arg1, arg2]
            result = subprocess.run(
                processed_command,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                print(f"    [WARN] Command returned non-zero: {result.returncode}")
                if result.stderr:
                    print(f"    stderr: {result.stderr[:500]}")
            
            if self.verbose:
                if result.stdout:
                    print(f"    stdout: {result.stdout[:500]}")
                    
            return True
            
        except subprocess.TimeoutExpired:
            print(f"    [ERROR] Command timed out")
            return False
        except FileNotFoundError as e:
            print(f"    [ERROR] Command not found: {e}")
            return False
        except Exception as e:
            print(f"    [ERROR] {e}")
            return False
    
    def handle_command(self, params: Dict[str, Any]) -> bool:
        """处理command操作 - 执行shell命令（与execute类似）"""
        command = params.get('command')
        shell = params.get('shell', False)
        
        if not command:
            print("    [ERROR] Missing command parameter")
            return False
        
        # 获取屏幕尺寸用于替换占位符
        screen_width = 1920  # 默认值
        screen_height = 1080  # 默认值
        try:
            result = subprocess.run(
                ['xdotool', 'getdisplaygeometry'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split()
                if len(parts) >= 2:
                    screen_width = int(parts[0])
                    screen_height = int(parts[1])
        except Exception:
            pass  # 使用默认值
        
        screen_width_half = screen_width // 2
        screen_height_half = screen_height // 2
        
        # 替换命令中的占位符
        if isinstance(command, list):
            processed_command = []
            for cmd_part in command:
                cmd_str = str(cmd_part)
                cmd_str = cmd_str.replace('{SCREEN_WIDTH}', str(screen_width))
                cmd_str = cmd_str.replace('{SCREEN_HEIGHT}', str(screen_height))
                cmd_str = cmd_str.replace('{SCREEN_WIDTH_HALF}', str(screen_width_half))
                cmd_str = cmd_str.replace('{SCREEN_HEIGHT_HALF}', str(screen_height_half))
                processed_command.append(cmd_str)
            cmd_str = ' '.join(processed_command)
        else:
            cmd_str = str(command)
            cmd_str = cmd_str.replace('{SCREEN_WIDTH}', str(screen_width))
            cmd_str = cmd_str.replace('{SCREEN_HEIGHT}', str(screen_height))
            cmd_str = cmd_str.replace('{SCREEN_WIDTH_HALF}', str(screen_width_half))
            cmd_str = cmd_str.replace('{SCREEN_HEIGHT_HALF}', str(screen_height_half))
            processed_command = cmd_str
            
        print(f"    Command: {cmd_str[:100]}{'...' if len(cmd_str) > 100 else ''}")
        
        try:
            if isinstance(command, list) and not shell:
                # 列表形式，不使用shell
                result = subprocess.run(
                    processed_command,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
            else:
                # 字符串形式，使用shell
                result = subprocess.run(
                    processed_command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
            
            if result.returncode != 0:
                print(f"    [WARN] Command returned non-zero: {result.returncode}")
                if result.stderr:
                    print(f"    stderr: {result.stderr[:500]}")
            
            if self.verbose:
                if result.stdout:
                    print(f"    stdout: {result.stdout[:500]}")
                    
            return True
            
        except subprocess.TimeoutExpired:
            print(f"    [ERROR] Command timed out")
            return False
        except FileNotFoundError as e:
            print(f"    [ERROR] Command not found: {e}")
            return False
        except Exception as e:
            print(f"    [ERROR] {e}")
            return False
    
    # ==================== 等待操作 ====================
    
    def handle_sleep(self, params: Dict[str, Any]) -> bool:
        """处理sleep操作 - 等待"""
        seconds = params.get('seconds', 1)
        
        try:
            seconds = float(seconds)
        except (ValueError, TypeError):
            seconds = 1
        
        if self.verbose:
            print(f"    Waiting {seconds} seconds...")
        
        try:
            time.sleep(seconds)
            return True
        except Exception as e:
            print(f"    [ERROR] {e}")
            return False
    
    # ==================== Chrome操作 ====================
    
    def handle_chrome_open_tabs(self, params: Dict[str, Any]) -> bool:
        """处理chrome_open_tabs操作 - 在Chrome中打开标签页"""
        urls = params.get('urls_to_open', [])
        
        if not urls:
            print("    [WARN] No URLs to open")
            return True
        
        print(f"    URLs: {urls}")
        
        # 检查Chrome是否可用
        chrome_paths = [
            '/usr/bin/google-chrome',
            '/usr/bin/chromium',
            '/usr/bin/chromium-browser',
            'google-chrome',
            'chromium',
            'chromium-browser'
        ]
        
        chrome_cmd = None
        for path in chrome_paths:
            try:
                subprocess.run(['which', path], check=True, capture_output=True)
                chrome_cmd = path
                break
            except subprocess.CalledProcessError:
                continue
        
        if not chrome_cmd:
            print("    [WARN] Chrome not found, skipping open_tabs")
            return True
        
        try:
            for url in urls:
                subprocess.Popen(
                    [chrome_cmd, '--new-tab', url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"    [OK] Opened: {url}")
                time.sleep(0.5)  # 避免同时打开太多标签页
            return True
        except Exception as e:
            print(f"    [ERROR] Failed to open tabs: {e}")
            return False
    
    def handle_chrome_close_tabs(self, params: Dict[str, Any]) -> bool:
        """处理chrome_close_tabs操作 - 关闭Chrome标签页"""
        print("    [INFO] Chrome close tabs operation")
        # 这需要更复杂的浏览器自动化，暂不实现
        print("    [WARN] Not implemented - requires browser automation")
        return True
    
    def handle_update_browse_history(self, params: Dict[str, Any]) -> bool:
        """处理update_browse_history操作 - 更新浏览器历史"""
        print("    [INFO] Update browse history operation")
        # 这需要浏览器配置或扩展，暂不实现
        print("    [WARN] Not implemented - requires browser configuration")
        return True
    
    # ==================== Google Drive操作 ====================
    
    def handle_googledrive(self, params: Dict[str, Any]) -> bool:
        """处理googledrive操作 - Google Drive文件操作"""
        settings_file = params.get('settings_file')
        operations = params.get('operation', [])
        args = params.get('args', [])
        
        print(f"    Settings: {settings_file}")
        print(f"    Operations: {operations}")
        
        # 这需要Google Drive API或相关工具
        print("    [WARN] Google Drive operation not implemented - requires API setup")
        return True
    
    def handle_login(self, params: Dict[str, Any]) -> bool:
        """处理login操作 - 登录操作"""
        settings_file = params.get('settings_file')
        platform = params.get('platform', '')
        
        print(f"    Settings: {settings_file}")
        print(f"    Platform: {platform}")
        
        # 这需要浏览器自动化来完成登录
        print("    [WARN] Login operation not implemented - requires browser automation")
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="OSWorld任务配置脚本 - 读取JSON文件的config字段完成环境配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python osworld_config.py task.json                    # 处理单个文件
  python osworld_config.py /path/to/directory          # 处理目录下所有JSON
  python osworld_config.py task.json -d                # 测试运行(不执行)
  python osworld_config.py task.json -v                # 详细输出
  python osworld_config.py /path -r                    # 递归处理子目录
  python osworld_config.py task.json --skip-network    # 跳过网络检查
        """
    )
    parser.add_argument(
        'json_path',
        nargs='?',
        help="JSON文件路径或包含JSON文件的目录"
    )
    parser.add_argument(
        '-d', '--dry-run',
        action='store_true',
        help="只打印操作而不实际执行"
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="输出详细信息"
    )
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help="递归处理子目录"
    )
    parser.add_argument(
        '--skip-network',
        action='store_true',
        help="跳过网络连通性检查"
    )
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help="遇到错误时继续执行后续操作"
    )
    
    args = parser.parse_args()
    
    if not args.json_path:
        parser.print_help()
        sys.exit(1)
    
    json_path = Path(args.json_path)
    
    if not json_path.exists():
        print(f"[ERROR] Path does not exist: {json_path}", file=sys.stderr)
        sys.exit(1)
    
    # 检查网络连接（仅在非dry-run且未跳过时）
    if not args.dry_run and not args.skip_network:
        print("Checking network connectivity...")
        try:
            if HAS_REQUESTS:
                requests.get('https://huggingface.co', timeout=5)
            else:
                import socket
                socket.create_connection(("huggingface.co", 80), timeout=5)
            print("[OK] Network is available")
        except Exception as e:
            print(f"[WARN] Network may not be available: {e}")
            print(f"[INFO] Use --skip-network to skip this check if files are pre-downloaded")
    
    if json_path.is_file():
        # 处理单个文件
        configurator = OSWorldConfigurator(
            str(json_path),
            dry_run=args.dry_run,
            verbose=args.verbose,
            continue_on_error=args.continue_on_error
        )
        
        if configurator.load_config():
            if configurator.execute_config():
                print("\n[OK] Configuration completed successfully")
                sys.exit(0)
            else:
                print("\n[FAIL] Configuration failed")
                sys.exit(1)
        else:
            sys.exit(1)
            
    elif json_path.is_dir():
        # 处理目录
        if args.recursive:
            # 递归查找所有JSON文件
            json_files = list(json_path.rglob("*.json"))
        else:
            json_files = list(json_path.glob("*.json"))
        
        if not json_files:
            print(f"[ERROR] No JSON files found in {json_path}", file=sys.stderr)
            sys.exit(1)
        
        print(f"Processing {len(json_files)} JSON files...")
        
        success_count = 0
        fail_count = 0
        
        for json_file in json_files:
            print(f"\n{'='*60}")
            print(f"Processing: {json_file}")
            print('='*60)
            
            configurator = OSWorldConfigurator(
                str(json_file),
                dry_run=args.dry_run,
                verbose=args.verbose,
                continue_on_error=args.continue_on_error
            )
            
            if configurator.load_config():
                if configurator.execute_config():
                    success_count += 1
                    print(f"\n[OK] Completed: {json_file.name}")
                else:
                    fail_count += 1
                    print(f"\n[FAIL] Failed: {json_file.name}")
            else:
                fail_count += 1
                print(f"\n[FAIL] Failed to load: {json_file.name}")
        
        print(f"\n{'='*60}")
        print(f"Summary: {success_count} succeeded, {fail_count} failed")
        sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
