# build.py

import os
import sys
import platform
import subprocess
import shutil
from datetime import datetime

doc = 'main_v22.py'

class BuildManager:
    def __init__(self):
        self.current_platform = platform.system().lower()
        self.version = datetime.now().strftime("%Y.%m.%d.%H%M")
        self.built_platforms = []  # 记录成功构建的平台

    def build_windows(self):
        """构建 Windows 版本"""
        print("🔨 构建 Windows 版本...")

        # 清理之前的构建
        for folder in ['build', 'dist']:
            if os.path.exists(folder):
                shutil.rmtree(folder)

        # Windows 特定的隐藏导入
        hidden_imports = [
            'pynput.keyboard._win32',
            'pynput.mouse._win32',
            'pyautogui',
            'keyboard',
            'PIL',
            'PIL._imaging',
            'PIL._imagingft',
            'psutil'
        ]

        params = [
            doc,
            '--onefile',
            '--console',
            '--name=ActionRecorder',
            '--clean',
            '--noconfirm',
            '--uac-admin',  # 请求管理员权限
        ]

        # 添加隐藏导入
        for imp in hidden_imports:
            params.extend(['--hidden-import', imp])

        try:
            subprocess.run([sys.executable, '-m', 'PyInstaller'] + params, check=True)
            print("✅ Windows 构建完成")
            self.built_platforms.append('windows')
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Windows 构建失败: {e}")
            return False

    def build_linux(self):
        """构建 Linux 版本"""
        print("🔨 构建 Linux 版本...")

        # 清理之前的构建
        for folder in ['build', 'dist']:
            if os.path.exists(folder):
                shutil.rmtree(folder)

        # Linux 特定的隐藏导入
        hidden_imports = [
            'pynput.keyboard._xorg',
            'pynput.mouse._xorg',
            'pyautogui',
            'PIL',
            'PIL._imaging',
            'PIL._imagingft',
            'psutil'
        ]

        params = [
            doc,
            '--onefile',
            '--console',
            '--name=ActionRecorder',
            '--clean',
            '--noconfirm',
        ]

        # 添加隐藏导入
        for imp in hidden_imports:
            params.extend(['--hidden-import', imp])

        try:
            subprocess.run([sys.executable, '-m', 'PyInstaller'] + params, check=True)
            print("✅ Linux 构建完成")
            self.built_platforms.append('linux')
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Linux 构建失败: {e}")
            return False

    def build_macos(self):
        """构建 macOS 版本"""
        print("🔨 构建 macOS 版本...")

        # 清理之前的构建
        for folder in ['build', 'dist']:
            if os.path.exists(folder):
                shutil.rmtree(folder)

        # macOS 特定的隐藏导入
        hidden_imports = [
            'pynput.keyboard._darwin',
            'pynput.mouse._darwin',
            'pyautogui',
            'PIL',
            'PIL._imaging',
            'PIL._imagingft',
            'psutil'
        ]

        params = [
            doc,
            '--onefile',
            '--console',
            '--name=ActionRecorder',
            '--clean',
            '--noconfirm',
        ]

        # 添加隐藏导入
        for imp in hidden_imports:
            params.extend(['--hidden-import', imp])

        try:
            subprocess.run([sys.executable, '-m', 'PyInstaller'] + params, check=True)
            print("✅ macOS 构建完成")
            self.built_platforms.append('macos')
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ macOS 构建失败: {e}")
            return False

    def create_release_packages(self, specific_platforms=None):
        """创建指定平台的发布包"""
        print("📦 创建发布包...")

        # 如果没有指定平台，使用已构建的平台
        if specific_platforms is None:
            platforms_to_package = self.built_platforms
        else:
            platforms_to_package = specific_platforms

        if not platforms_to_package:
            print("❌ 没有找到已构建的平台，请先构建平台")
            return 0

        success_count = 0

        for platform_name in platforms_to_package:
            # 创建发布目录
            release_dir = f"ActionRecorder_{platform_name}_v{self.version}"
            if os.path.exists(release_dir):
                shutil.rmtree(release_dir)
            os.makedirs(release_dir)

            # 复制可执行文件
            if platform_name == 'windows':
                exe_src = 'dist/ActionRecorder.exe'
                exe_dst = os.path.join(release_dir, 'ActionRecorder.exe')
            else:
                exe_src = 'dist/ActionRecorder'
                exe_dst = os.path.join(release_dir, 'ActionRecorder')

            if os.path.exists(exe_src):
                shutil.copy2(exe_src, exe_dst)

                # 在 Unix 系统上添加执行权限
                if platform_name != 'windows':
                    os.chmod(exe_dst, 0o755)

                # 创建使用说明
                self.create_platform_readme(release_dir, platform_name)
                print(f"✅ {platform_name} 发布包创建完成: {release_dir}")
                success_count += 1
            else:
                print(f"❌ {platform_name} 可执行文件未找到: {exe_src}")

        print(f"\n🎉 成功创建 {success_count} 个平台版本")
        return success_count

    def create_platform_readme(self, release_dir, platform_name):
        """创建平台特定的使用说明"""

        instructions = {
            'windows': """
🖥️ Windows 使用说明:

1. 运行 ActionRecorder.exe
2. 同意管理员权限请求
3. 正常使用电脑进行操作
4. 按 F12 键停止记录
5. 回车停止运行
6. 查看生成的会话文件夹中的报告和截图

📋 系统要求:
- Windows 7/8/10/11
- 需要管理员权限
- .NET Framework 4.5+

⚠️ 注意事项:
- 杀毒软件可能会误报，请添加到白名单
- 首次运行可能需要授权 UAC
""",
            'linux': """
🐧 Linux 使用说明:

1. 给程序执行权限:
   chmod +x ActionRecorder

2. 运行程序:
   ./ActionRecorder

3. 正常使用电脑进行操作
4. 按 F12 键停止记录
5. 回车停止运行
6. 查看生成的会话文件夹中的报告和截图

📋 系统要求:
- Ubuntu/Debian/CentOS 等主流发行版
- 可能需要安装依赖:
  sudo apt-get install python3-tk python3-dev

⚠️ 注意事项:
- 可能需要 X11 显示服务器
- 确保有足够的输入设备权限
""",
            'macos': """
🍎 macOS 使用说明:

1. 首次运行需要授权:
   - 系统偏好设置 → 安全性与隐私 → 辅助功能
   - 解锁并添加 ActionRecorder
   - 同样在"输入监控"中授权

2. 运行 ActionRecorder
3. 正常使用电脑进行操作
4. 按 F12 键停止记录
5. 回车停止运行
6. 查看生成的会话文件夹中的报告和截图

📋 系统要求:
- macOS 10.12 或更高版本
- 需要辅助功能权限

⚠️ 注意事项:
- 首次运行系统会提示授权
- 如果无法运行，请右键点击 → 打开
"""
        }

        readme_content = f"""操作记录器 v{self.version} - {platform_name.upper()} 版本

{instructions.get(platform_name, '请参考通用使用说明')}

📞 技术支持: 17828820676 (南京大学软件学院 小代同学)
- 开发者平台: {platform_name}
- 版本: {self.version}
- 构建时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 系统架构: {platform.machine()}
"""

        readme_path = os.path.join(release_dir, 'README.txt')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)


def main():
    print("🚀 跨平台操作记录器构建工具")
    print("=" * 50)

    # 检查 PyInstaller 是否安装
    try:
        import PyInstaller
    except ImportError:
        print("❌ 请先安装 PyInstaller: pip install pyinstaller")
        return

    manager = BuildManager()

    print(f"当前系统: {platform.system()} {platform.release()}")
    print(f"构建版本: {manager.version}")
    print()

    print("选择构建选项:")
    print("1. 构建 Windows 版本")
    print("2. 构建 Linux 版本")
    print("3. 构建 macOS 版本")
    print("4. 构建所有平台版本")
    print("5. 仅创建发布包（需要先构建，不推荐）")

    try:
        choice = input("\n请输入选择 (1-5): ").strip()

        if choice == '1':
            if manager.build_windows():
                manager.create_release_packages(['windows'])
        elif choice == '2':
            if manager.build_linux():
                manager.create_release_packages(['linux'])
        elif choice == '3':
            if manager.build_macos():
                manager.create_release_packages(['macos'])
        elif choice == '4':
            # 构建所有平台
            results = []
            results.append(manager.build_windows())
            results.append(manager.build_linux())
            results.append(manager.build_macos())

            if any(results):
                # 只为成功构建的平台创建发布包
                manager.create_release_packages()
        elif choice == '5':
            # 只为已构建的平台创建发布包
            manager.create_release_packages()
        else:
            print("❌ 无效选择")

    except KeyboardInterrupt:
        print("\n👋 用户取消")
    except Exception as e:
        print(f"❌ 发生错误: {e}")


if __name__ == "__main__":
    main()