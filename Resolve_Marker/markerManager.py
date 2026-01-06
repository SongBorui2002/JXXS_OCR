#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DaVinci Resolve Marker 管理器
用于获取和管理时间线中的标记点
"""

from typing import Dict, List, Optional, Any
import json
from resolveConnector import ResolveConnector


class MarkerInfo:
    """标记点信息类"""

    def __init__(self, frame_id: float, color: str, name: str, note: str,
                 duration: float, custom_data: str):
        self.frame_id = frame_id
        self.color = color
        self.name = name
        self.note = note
        self.duration = duration
        self.custom_data = custom_data

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'frame_id': self.frame_id,
            'color': self.color,
            'name': self.name,
            'note': self.note,
            'duration': self.duration,
            'custom_data': self.custom_data
        }

    def __str__(self) -> str:
        return f"Marker(frame={self.frame_id}, color={self.color}, name='{self.name}')"

    def __repr__(self) -> str:
        return self.__str__()


class MarkerManager:
    """DaVinci Resolve Marker 管理器"""

    # 标记颜色映射
    COLOR_MAPPING = {
        'VFX': 'Green',
        'DI': 'Yellow',  # DI使用Yellow颜色
        'green': 'Green',
        'orange': 'Orange',
        'red': 'Red',
        'blue': 'Blue',
        'cyan': 'Cyan',
        'magenta': 'Magenta',
        'yellow': 'Yellow'
    }

    def __init__(self):
        """初始化 Marker 管理器"""
        self.connector = ResolveConnector()

    def get_all_markers(self) -> Dict[float, MarkerInfo]:
        """
        获取当前时间线中所有的标记点

        Returns:
            Dict[float, MarkerInfo]: 以frame_id为键，MarkerInfo对象为值的字典

        Raises:
            ConnectionError: 当无法获取时间线时抛出
        """
        try:
            timeline = self.connector.get_timeline()
            raw_markers = timeline.GetMarkers()

            if not raw_markers:
                return {}

            markers = {}
            for frame_id, marker_data in raw_markers.items():
                marker_info = MarkerInfo(
                    frame_id=float(frame_id),
                    color=marker_data.get('color', ''),
                    name=marker_data.get('name', ''),
                    note=marker_data.get('note', ''),
                    duration=float(marker_data.get('duration', 1.0)),
                    custom_data=marker_data.get('customData', '')
                )
                markers[float(frame_id)] = marker_info

            return markers

        except Exception as e:
            raise ConnectionError(f"获取标记点失败: {str(e)}")

    def add_marker(self, frame_id: float, color: str, name: str = "",
                   note: str = "", duration: float = 1.0,
                   custom_data: str = "") -> bool:
        """
        在指定帧添加标记点

        Args:
            frame_id (float): 帧号
            color (str): 标记点颜色，支持预定义的颜色名称或标准颜色名
            name (str): 标记点名称
            note (str): 标记点注释
            duration (float): 标记点持续时间（帧）
            custom_data (str): 自定义数据

        Returns:
            bool: 添加是否成功

        Raises:
            ConnectionError: 当无法获取时间线时抛出
        """
        try:
            timeline = self.connector.get_timeline()

            # 标准化颜色名称
            normalized_color = self.COLOR_MAPPING.get(color.lower(), color)

            # 调用DaVinci Resolve API添加标记点
            success = timeline.AddMarker(frame_id, normalized_color, name, note, duration, custom_data)

            if success:
                print(f"✓ 已添加标记点: 帧{frame_id}, 颜色{normalized_color}, 名称'{name}'")
            else:
                print(f"✗ 添加标记点失败: 帧{frame_id}")

            return success

        except Exception as e:
            print(f"添加标记点时出错: {str(e)}")
            return False

    def add_markers_from_csv(self, csv_file_path: str, frame_col: int = 0,
                           note_col: int = 2, type_col: int = 5,
                           has_header: bool = True) -> Dict[str, int]:
        """
        从CSV文件批量添加标记点

        Args:
            csv_file_path (str): CSV文件路径
            frame_col (int): 帧号列索引（从0开始）
            note_col (int): 注释列索引（从0开始）
            type_col (int): 类型列索引（从0开始）
            has_header (bool): 是否有表头行

        Returns:
            Dict[str, int]: 添加结果统计 {'success': 成功数量, 'failed': 失败数量, 'total': 总数}
        """
        import csv

        result = {'success': 0, 'failed': 0, 'total': 0}

        try:
            with open(csv_file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                rows = list(reader)

            # 跳过表头
            start_row = 1 if has_header else 0

            for row in rows[start_row:]:
                if len(row) <= max(frame_col, note_col, type_col):
                    print(f"⚠️ 跳过无效行: {row}")
                    continue

                try:
                    # 解析数据
                    frame_id = float(row[frame_col].strip())
                    note = row[note_col].strip()
                    marker_type = row[type_col].strip()

                    # 根据类型确定颜色和名称
                    if marker_type == 'VFX':
                        color = 'Green'
                        name = 'VFX'
                    elif marker_type == 'DI':
                        color = 'Yellow'  # DI使用Yellow颜色
                        name = 'DI'
                    else:
                        color = 'Blue'  # 默认颜色
                        name = marker_type

                    result['total'] += 1

                    # 添加标记点
                    if self.add_marker(frame_id, color, name, note):
                        result['success'] += 1
                    else:
                        result['failed'] += 1

                except (ValueError, IndexError) as e:
                    print(f"⚠️ 解析行失败: {row}, 错误: {str(e)}")
                    result['failed'] += 1
                    result['total'] += 1

        except FileNotFoundError:
            print(f"✗ 找不到CSV文件: {csv_file_path}")
        except Exception as e:
            print(f"读取CSV文件时出错: {str(e)}")

        return result

    def delete_marker_at_frame(self, frame_num: float) -> bool:
        """
        删除指定帧号的标记点

        Args:
            frame_num (float): 要删除标记点的帧号

        Returns:
            bool: 删除是否成功

        Raises:
            ConnectionError: 当无法获取时间线时抛出
        """
        try:
            timeline = self.connector.get_timeline()

            success = timeline.DeleteMarkerAtFrame(frame_num)

            if success:
                print(f"✓ 已删除帧{frame_num}的标记点")
            else:
                print(f"✗ 删除帧{frame_num}的标记点失败（可能不存在）")

            return success

        except Exception as e:
            print(f"删除标记点时出错: {str(e)}")
            return False

    def delete_marker_by_custom_data(self, custom_data: str) -> bool:
        """
        删除具有指定自定义数据的第一个匹配标记点

        Args:
            custom_data (str): 自定义数据

        Returns:
            bool: 删除是否成功

        Raises:
            ConnectionError: 当无法获取时间线时抛出
        """
        try:
            timeline = self.connector.get_timeline()

            success = timeline.DeleteMarkerByCustomData(custom_data)

            if success:
                print(f"✓ 已删除自定义数据为'{custom_data}'的标记点")
            else:
                print(f"✗ 删除自定义数据为'{custom_data}'的标记点失败（可能不存在）")

            return success

        except Exception as e:
            print(f"删除标记点时出错: {str(e)}")
            return False

    def delete_markers_by_color(self, color: str) -> bool:
        """
        删除指定颜色的所有标记点

        Args:
            color (str): 标记点颜色（如 'Green', 'Red', 'Blue' 等），支持 "All" 删除所有标记点

        Returns:
            bool: 删除是否成功

        Raises:
            ConnectionError: 当无法获取时间线时抛出
        """
        try:
            timeline = self.connector.get_timeline()

            # 标准化颜色名称
            normalized_color = self.COLOR_MAPPING.get(color.lower(), color)

            success = timeline.DeleteMarkersByColor(normalized_color)

            if success:
                if normalized_color.lower() == 'all':
                    print("✓ 已删除所有标记点")
                else:
                    print(f"✓ 已删除所有{normalized_color}颜色的标记点")
            else:
                if normalized_color.lower() == 'all':
                    print("✗ 删除所有标记点失败")
                else:
                    print(f"✗ 删除{normalized_color}颜色的标记点失败")

            return success

        except Exception as e:
            print(f"删除标记点时出错: {str(e)}")
            return False

    def delete_all_markers(self) -> bool:
        """
        删除所有标记点

        Returns:
            bool: 删除是否成功
        """
        return self.delete_markers_by_color("All")

    def clear_markers_in_frame_range(self, start_frame: float, end_frame: float) -> Dict[str, int]:
        """
        删除指定帧范围内的所有标记点

        Args:
            start_frame (float): 起始帧号（包含）
            end_frame (float): 结束帧号（包含）

        Returns:
            Dict[str, int]: 删除结果统计 {'deleted': 删除数量, 'total_found': 范围内标记点总数}
        """
        result = {'deleted': 0, 'total_found': 0}

        try:
            all_markers = self.get_all_markers()

            # 找到范围内所有标记点
            markers_to_delete = []
            for frame_id, marker in all_markers.items():
                if start_frame <= frame_id <= end_frame:
                    markers_to_delete.append(frame_id)
                    result['total_found'] += 1

            # 删除标记点
            for frame_id in markers_to_delete:
                if self.delete_marker_at_frame(frame_id):
                    result['deleted'] += 1

        except Exception as e:
            print(f"清除帧范围标记点时出错: {str(e)}")

        return result

    def get_markers_list(self) -> List[MarkerInfo]:
        """
        获取当前时间线中所有的标记点（列表格式）

        Returns:
            List[MarkerInfo]: MarkerInfo对象的列表，按frame_id排序
        """
        markers_dict = self.get_all_markers()
        # 按frame_id排序
        return [markers_dict[frame_id] for frame_id in sorted(markers_dict.keys())]

    def get_markers_by_color(self, color: str) -> List[MarkerInfo]:
        """
        根据颜色获取标记点

        Args:
            color (str): 标记点颜色（如 'Green', 'Red', 'Blue' 等）

        Returns:
            List[MarkerInfo]: 指定颜色的标记点列表
        """
        all_markers = self.get_all_markers()
        return [marker for marker in all_markers.values() if marker.color == color]

    def get_marker_at_frame(self, frame_id: float) -> Optional[MarkerInfo]:
        """
        获取指定帧的标记点

        Args:
            frame_id (float): 帧号

        Returns:
            Optional[MarkerInfo]: 如果存在标记点则返回，否则返回None
        """
        all_markers = self.get_all_markers()
        return all_markers.get(frame_id)

    def get_markers_with_custom_data(self, custom_data: str) -> List[MarkerInfo]:
        """
        获取具有指定自定义数据的标记点

        Args:
            custom_data (str): 自定义数据

        Returns:
            List[MarkerInfo]: 具有指定自定义数据的标记点列表
        """
        all_markers = self.get_all_markers()
        return [marker for marker in all_markers.values() if marker.custom_data == custom_data]

    def get_markers_summary(self) -> Dict[str, Any]:
        """
        获取标记点的汇总信息

        Returns:
            Dict[str, Any]: 包含标记点统计信息的字典
        """
        all_markers = self.get_all_markers()

        summary = {
            'total_count': len(all_markers),
            'colors': {},
            'frame_range': None,
            'markers': []
        }

        if all_markers:
            # 统计颜色分布
            for marker in all_markers.values():
                color = marker.color
                summary['colors'][color] = summary['colors'].get(color, 0) + 1

            # 获取帧范围
            frame_ids = list(all_markers.keys())
            summary['frame_range'] = {
                'min': min(frame_ids),
                'max': max(frame_ids)
            }

            # 转换为字典列表用于JSON序列化
            summary['markers'] = [marker.to_dict() for marker in self.get_markers_list()]

        return summary

    def print_markers(self, markers: Optional[List[MarkerInfo]] = None) -> None:
        """
        打印标记点信息

        Args:
            markers (Optional[List[MarkerInfo]]): 要打印的标记点列表，如果为None则打印所有标记点
        """
        if markers is None:
            markers = self.get_markers_list()

        if not markers:
            print("当前时间线没有标记点")
            return

        print(f"找到 {len(markers)} 个标记点:")
        print("-" * 80)
        print("<10")
        print("-" * 80)

        for marker in markers:
            print("<10")

    def export_markers_to_json(self, file_path: str) -> bool:
        """
        将标记点导出为JSON文件

        Args:
            file_path (str): 导出文件路径

        Returns:
            bool: 导出是否成功
        """
        try:
            summary = self.get_markers_summary()

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            print(f"标记点已导出到: {file_path}")
            return True

        except Exception as e:
            print(f"导出标记点失败: {str(e)}")
            return False


# 使用示例
if __name__ == "__main__":
    try:
        # 创建标记管理器
        marker_manager = MarkerManager()

        # 获取所有标记点
        all_markers = marker_manager.get_all_markers()
        print(f"总共找到 {len(all_markers)} 个标记点")

        # 打印所有标记点
        marker_manager.print_markers()

        # 获取标记点汇总
        summary = marker_manager.get_markers_summary()
        print(f"\n标记点汇总: {json.dumps(summary, indent=2, ensure_ascii=False)}")

        # 导出为JSON
        marker_manager.export_markers_to_json("timeline_markers.json")

        # 示例：添加单个标记点
        print("\n--- 添加标记点示例 ---")
        # marker_manager.add_marker(1000.0, 'Green', 'VFX', '测试标记点')

        # 示例：从CSV批量添加标记点
        print("\n--- 从CSV批量添加标记点示例 ---")
        # csv_result = marker_manager.add_markers_from_csv("EP24_detected_frames_paddle_refactored.csv")
        # print(f"CSV导入结果: {csv_result}")

        # 示例：删除标记点
        print("\n--- 删除标记点示例 ---")
        # 删除指定帧的标记点
        # marker_manager.delete_marker_at_frame(1000.0)

        # 删除指定颜色的所有标记点
        # marker_manager.delete_markers_by_color('Green')

        # 删除所有标记点
        # marker_manager.delete_all_markers()

        # 删除指定帧范围内的标记点
        # range_result = marker_manager.clear_markers_in_frame_range(1000.0, 5000.0)
        # print(f"帧范围删除结果: {range_result}")

    except Exception as e:
        print(f"错误: {str(e)}")
