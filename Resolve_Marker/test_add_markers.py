#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 MarkerManager 添加标记点功能
"""

from markerManager import MarkerManager
import os


def test_add_markers():
    """测试添加标记点功能"""
    try:
        print("正在初始化 MarkerManager...")
        marker_manager = MarkerManager()

        print("✓ 成功连接到 DaVinci Resolve")

        # 测试添加单个标记点
        print("\n测试: 添加单个标记点")
        success = marker_manager.add_marker(
            frame_id=1000.0,
            color='Green',
            name='VFX',
            note='测试VFX标记点',
            duration=1.0
        )
        if success:
            print("✓ 单个标记点添加成功")
        else:
            print("✗ 单个标记点添加失败")

        # 测试从CSV文件批量添加标记点
        print("\n测试: 从CSV批量添加标记点")

        # 检查CSV文件是否存在
        csv_path = "EP25_detected_frames_paddle_refactored.csv"
        if not os.path.exists(csv_path):
            # 如果当前目录没有，尝试上级目录
            csv_path = "../EP25_detected_frames_paddle_refactored.csv"

        if os.path.exists(csv_path):
            print(f"找到CSV文件: {csv_path}")

            # 读取CSV文件的前几行来确认格式
            import csv
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                first_row = next(reader)

            print(f"CSV表头: {header}")
            print(f"第一行数据: {first_row}")
            print(f"列数: {len(first_row)}")

            # 添加标记点
            result = marker_manager.add_markers_from_csv(csv_path)
            print(f"\n批量添加结果: {result}")
            print(f"成功: {result['success']}, 失败: {result['failed']}, 总数: {result['total']}")

        else:
            print(f"✗ 找不到CSV文件: {csv_path}")

        # 验证添加结果
        print("\n验证: 检查添加后的标记点")
        all_markers = marker_manager.get_all_markers()
        print(f"当前总标记点数: {len(all_markers)}")

        # 显示最近添加的标记点
        markers_list = marker_manager.get_markers_list()
        if markers_list:
            print("最近的标记点:")
            for marker in markers_list[-5:]:  # 显示最后5个
                print(f"  {marker}")

        print("\n🎉 测试完成！")

    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


def test_csv_parsing():
    """测试CSV解析功能（不连接Resolve）"""
    print("测试CSV解析功能...")

    try:
        csv_path = "EP24_detected_frames_paddle_refactored.csv"
        if not os.path.exists(csv_path):
            csv_path = "../EP24_detected_frames_paddle_refactored.csv"

        if not os.path.exists(csv_path):
            print(f"✗ 找不到CSV文件: {csv_path}")
            return

        import csv

        print(f"解析CSV文件: {csv_path}")
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = list(reader)

        print(f"总行数: {len(rows)}")
        print(f"表头: {rows[0]}")

        # 模拟解析逻辑
        for i, row in enumerate(rows[1:6], 1):  # 只显示前5行
            if len(row) >= 6:
                frame_id = row[0].strip()
                note = row[2].strip()
                marker_type = row[5].strip()

                if marker_type == 'VFX':
                    color = 'Green'
                    name = 'VFX'
                elif marker_type == 'DI':
                    color = 'Yellow'  # DI使用Yellow颜色
                    name = 'DI'
                else:
                    color = 'Blue'
                    name = marker_type

                print(f"行{i}: frame={frame_id}, color={color}, name={name}, note='{note[:50]}...'")
            else:
                print(f"行{i}: 数据不完整 - {row}")

    except Exception as e:
        print(f"CSV解析测试失败: {str(e)}")


if __name__ == "__main__":
    # 首先测试CSV解析（不需要Resolve连接）
    test_csv_parsing()
    print("\n" + "="*50 + "\n")

    # 然后测试完整的添加功能
    test_add_markers()
