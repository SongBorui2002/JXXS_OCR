#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 MarkerManager 类的功能
"""

from markerManager import MarkerManager


def test_marker_manager():
    """测试标记管理器"""
    try:
        print("正在初始化 MarkerManager...")
        marker_manager = MarkerManager()

        print("✓ 成功连接到 DaVinci Resolve")

        # 测试获取所有标记点
        print("\n测试: 获取所有标记点")
        all_markers = marker_manager.get_all_markers()
        print(f"✓ 找到 {len(all_markers)} 个标记点")

        # 测试获取标记点列表
        print("\n测试: 获取标记点列表")
        markers_list = marker_manager.get_markers_list()
        print(f"✓ 获取到 {len(markers_list)} 个标记点")

        # 打印标记点详情
        if markers_list:
            print("\n测试: 打印标记点详情")
            marker_manager.print_markers()
        else:
            print("当前时间线没有标记点")

        # 测试获取不同颜色的标记点
        print("\n测试: 按颜色分组标记点")
        colors = ['Green', 'Red', 'Blue', 'Cyan', 'Magenta', 'Yellow']
        for color in colors:
            color_markers = marker_manager.get_markers_by_color(color)
            if color_markers:
                print(f"✓ {color}: {len(color_markers)} 个标记点")

        # 测试获取标记点汇总
        print("\n测试: 获取标记点汇总信息")
        summary = marker_manager.get_markers_summary()
        print("✓ 汇总信息获取成功")
        print(f"  总数量: {summary['total_count']}")
        if summary['colors']:
            print(f"  颜色分布: {summary['colors']}")
        if summary['frame_range']:
            print(f"  帧范围: {summary['frame_range']}")

        # 测试导出功能
        print("\n测试: 导出标记点到JSON")
        success = marker_manager.export_markers_to_json("test_markers.json")
        if success:
            print("✓ 导出成功")
        else:
            print("✗ 导出失败")

        print("\n🎉 所有测试完成！")

        # 删除所有标记点
        marker_manager.delete_all_markers()

    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_marker_manager()
