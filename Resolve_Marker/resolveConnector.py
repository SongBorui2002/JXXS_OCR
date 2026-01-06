#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DaVinci Resolve 连接器
统一管理 Resolve 连接和环境设置
"""

import sys
import os


# 首先设置 fusionscript.so 环境变量（必须在导入前）
# 注意：强制覆盖可能存在的错误环境变量
possible_lib_paths = [
    "/Applications/DaVinci Resolve20.2/DaVinci Resolve20.2.app/Contents/Libraries/Fusion/fusionscript.so",
    "/Applications/DaVinci Resolve19.0/DaVinci Resolve19.0.app/Contents/Libraries/Fusion/fusionscript.so",
    "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so",
]

found = False
for path in possible_lib_paths:
    if os.path.exists(path):
        os.environ["RESOLVE_SCRIPT_LIB"] = path
        print(f"✓ 找到并设置 fusionscript.so: {path}")
        found = True
        break

if not found:
    print("⚠️  警告：未找到 fusionscript.so 文件")

# 添加 DaVinciResolveScript 模块路径
module_paths = [
    "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules/",
]

for path in module_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.append(path)

# 导入 DaVinciResolveScript
try:
    import DaVinciResolveScript as dvr_script
except ImportError as e:
    raise ImportError(f"无法导入 DaVinciResolveScript: {e}\n请确保 Resolve 正在运行并启用了外部脚本访问")


class ResolveConnector:
    """DaVinci Resolve 连接器基类"""
    
    def __init__(self):
        """初始化并连接到 Resolve"""
        self.resolve = dvr_script.scriptapp("Resolve")
        if not self.resolve:
            raise ConnectionError("无法连接到 Resolve，请确保 Resolve 正在运行")
        
        self.project_manager = self.resolve.GetProjectManager()
        if not self.project_manager:
            raise ConnectionError("无法获取项目管理器")
        
        self.project = self.project_manager.GetCurrentProject()
        if not self.project:
            raise ConnectionError("没有打开的项目")
    
    def get_project_name(self):
        """获取项目名称"""
        return self.project.GetName()
    
    def get_timeline(self):
        """获取当前 timeline"""
        timeline = self.project.GetCurrentTimeline()
        if not timeline:
            raise ConnectionError("没有打开的 timeline")
        return timeline
    
    def get_media_pool(self):
        """获取 Media Pool"""
        media_pool = self.project.GetMediaPool()
        if not media_pool:
            raise ConnectionError("无法获取 MediaPool")
        return media_pool

