"""
信息处理服务
负责结果过滤、合并、规范化
"""

from ast import If
import csv
import os
from typing import List, Dict, Any, Tuple
from paddle_ocr_service import OCRResult
from config import OUTPUT_CSV_HEADERS

class ResultProcessor:
    """信息处理服务"""

    def __init__(self, video_path: str):
        """初始化结果处理器"""
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]

        print("结果处理器初始化完成")

    def process_text(self, text: str, text_type: str) -> str:
        """规范化处理识别结果"""
        if not text:
            return ""

        if text_type == "VFX":
            # 替换各种变体
            text = text.replace('VEX:', 'VFX:')
            text = text.replace('VFX;', 'VFX:')
            text = text.replace('VEX;', 'VFX:')
            text = text.replace('VFX.', 'VFX:')  # 处理点号
            text = text.replace('VEX.', 'VFX:')
            text = text.replace('FX:', 'VFX:')
            text = text.replace('VVFX:', 'VFX:')

            # 如果不以"VFX:"开头，但以"VFX"开头，添加冒号
            if not text.startswith('VFX:') and text.startswith('VFX'):
                text = 'VFX:' + text[3:]
            elif not text.startswith('VFX:'):
                if text[:4].lower().replace(' ', '') in ['vfx', 'vex', 'vpx', 'vix']:
                    colon_pos = text.find(':')
                    semicolon_pos = text.find(';')
                    if colon_pos != -1:
                        text = 'VFX:' + text[colon_pos + 1:]
                    elif semicolon_pos != -1:
                        text = 'VFX:' + text[semicolon_pos + 1:]
                    else:
                        text = 'VFX:' + text[4:]
                else:
                    text = 'VFX:' + text
        else:  # DI
            text = text.replace('D1:', 'DI:')
            text = text.replace('D1;', 'DI:')
            text = text.replace('Di', 'DI:')
            text = text.replace('Di;', 'DI:')
            text = text.replace('Di:', 'DI:')
            text = text.replace('Dl:', 'DI:')
            text = text.replace('D|:', 'DI:')
            text = text.replace('DL:', 'DI:')
            text = text.replace('01:', 'DI:')
            text = text.replace('DI;', 'DI:')
            text = text.replace('D1;', 'DI:')
            text = text.replace('Dl;', 'DI:')
            text = text.replace('D|;', 'DI:')
            text = text.replace('DL;', 'DI:')
            text = text.replace('01;', 'DI:')
            if not text.startswith('DI:') and text.startswith('DI'):
                text = 'DI:' + text[3:]
            elif not text.startswith('DI:'):
                if text[:3].lower().replace(' ', '') in ['di', 'd1', 'dl', 'ol', 'oi', '01']:
                    colon_pos = text.find(':')
                    semicolon_pos = text.find(';')
                    if colon_pos != -1:
                        text = 'DI:' + text[colon_pos + 1:]
                    elif semicolon_pos != -1:
                        text = 'DI:' + text[semicolon_pos + 1:]
                    else:
                        text = 'DI:' + text[3:]
                else:
                    text = 'DI:' + text

        return text

    def filter_results(self, ocr_results: List[OCRResult], min_confidence: float = 0.0) -> List[OCRResult]:
        """过滤OCR结果"""
        filtered_results = []

        for result in ocr_results:
            # 置信度过滤
            if result.confidence < min_confidence:
                continue

            # 文本规范化
            processed_text = self.process_text(result.text, result.text_type)

            # 只保留有意义的文本
            if processed_text.strip():
                result.text = processed_text
                filtered_results.append(result)

        print(f"结果过滤完成: 原始 {len(ocr_results)} 个结果，过滤后 {len(filtered_results)} 个结果")
        return filtered_results

    def deduplicate_results(self, ocr_results: List[OCRResult], time_threshold: float = 1.0) -> List[OCRResult]:
        """去重处理：合并时间相近的相似结果"""
        if not ocr_results:
            return ocr_results

        # 按帧号排序
        sorted_results = sorted(ocr_results, key=lambda x: x.frame_number)

        deduplicated = []
        current_group = [sorted_results[0]]

        for i in range(1, len(sorted_results)):
            current = sorted_results[i]
            prev = current_group[-1]

            # 检查是否属于同一组（时间相近且类型相同）
            time_diff = abs(current.frame_number - prev.frame_number)
            same_type = current.text_type == prev.text_type

            if time_diff <= time_threshold * 25 and same_type:  # 1秒 = 25帧（假设25fps）
                current_group.append(current)
            else:
                # 处理当前组
                best_result = self._select_best_from_group(current_group)
                deduplicated.append(best_result)
                current_group = [current]

        # 处理最后一组
        if current_group:
            best_result = self._select_best_from_group(current_group)
            deduplicated.append(best_result)

        print(f"去重完成: 原始 {len(ocr_results)} 个结果，去重后 {len(deduplicated)} 个结果")
        return deduplicated

    def deduplicate_by_continuous_frames_iou(self, ocr_results: List[OCRResult], max_frame_gap: int = 3, iou_threshold: float = 0.8) -> List[OCRResult]:
        """基于连续帧和IoU的去重处理"""
        if not ocr_results:
            return ocr_results

        # 按帧号排序
        sorted_results = sorted(ocr_results, key=lambda x: x.frame_number)

        deduplicated = []
        i = 0

        while i < len(sorted_results):
            current_result = sorted_results[i]
            continuous_group = [current_result]
            j = i + 1

            # 查找连续的帧组（帧号差距不超过max_frame_gap）
            while j < len(sorted_results):
                next_result = sorted_results[j]

                # 检查帧号连续性和类型相同
                frame_gap = next_result.frame_number - continuous_group[-1].frame_number
                type_match = next_result.text_type == current_result.text_type

                if (frame_gap <= max_frame_gap and type_match):

                    # 计算IoU
                    current_bbox = self._bbox_from_paddle_points(current_result.bbox) if isinstance(current_result.bbox, list) else current_result.bbox
                    next_bbox = self._bbox_from_paddle_points(next_result.bbox) if isinstance(next_result.bbox, list) else next_result.bbox

                    iou = self._calculate_iou(current_bbox, next_bbox)

                    if iou >= iou_threshold:
                        continuous_group.append(next_result)
                        j += 1
                    else:
                        break  # IoU不满足，结束当前组
                else:
                    break  # 帧号不连续或类型不同，结束当前组

            # 处理当前连续组
            if len(continuous_group) >= 10:  # 只有长度>=10的组才认为是真正的字幕
                # 从连续组中选择最佳结果，但保持第一帧的时间
                best_result = self._select_best_from_continuous_group(continuous_group)
                # 保持第一帧的帧号和时间码
                best_result.frame_number = continuous_group[0].frame_number
                best_result.timecode = continuous_group[0].timecode
                deduplicated.append(best_result)
                print(f"连续帧组去重: {len(continuous_group)} 帧 -> 1 帧 (帧 {continuous_group[0].frame_number})")
            elif len(continuous_group) > 1:
                print(f"跳过短连续组: {len(continuous_group)} 帧 (帧 {continuous_group[0].frame_number}) - 长度不足10帧")
            else:
                # 单个结果直接删除
                print(f"删除单帧结果: 帧 {continuous_group[0].frame_number}")

            i = j  # 移动到下一组的开始

        print(f"连续帧IoU去重完成: 原始 {len(ocr_results)} 个结果，去重后 {len(deduplicated)} 个结果")
        return deduplicated

    def _select_best_from_continuous_group(self, group: List[OCRResult]) -> OCRResult:
        """从连续帧组中选择最佳结果（基于置信度和文本清晰度）"""
        if len(group) == 1:
            return group[0]

        # 首先选择置信度最高的结果
        best_result = max(group, key=lambda x: x.confidence)

        # 如果有多个相同置信度的，选择文本长度更长的（假设更完整）
        same_confidence = [r for r in group if abs(r.confidence - best_result.confidence) < 0.01]
        if len(same_confidence) > 1:
            best_result = max(same_confidence, key=lambda x: len(x.text.strip()))

        return best_result

    def _select_best_from_group(self, group: List[OCRResult]) -> OCRResult:
        """从一组结果中选择最好的"""
        if len(group) == 1:
            return group[0]

        # 选择置信度最高的结果
        best_result = max(group, key=lambda x: x.confidence)

        # 如果有多个相同置信度的，选择像素数量最大的
        same_confidence = [r for r in group if r.confidence == best_result.confidence]
        if len(same_confidence) > 1:
            best_result = max(same_confidence, key=lambda x: x.pixel_count)

        return best_result

    def merge_similar_texts(self, ocr_results: List[OCRResult]) -> List[OCRResult]:
        """合并相似的文本结果"""
        if not ocr_results:
            return ocr_results

        merged = []
        used_indices = set()

        for i, result in enumerate(ocr_results):
            if i in used_indices:
                continue

            similar_group = [result]

            # 查找相似的文本
            for j in range(i + 1, len(ocr_results)):
                if j in used_indices:
                    continue

                other = ocr_results[j]

                # 相似性判断：相同类型、文本相似度高、时间相近
                if (result.text_type == other.text_type and
                    self._text_similarity(result.text, other.text) > 0.8 and
                    abs(result.frame_number - other.frame_number) <= 25):  # 1秒内

                    similar_group.append(other)
                    used_indices.add(j)

            # 从相似组中选择最好的
            best_result = self._select_best_from_group(similar_group)
            merged.append(best_result)

        print(f"文本合并完成: 原始 {len(ocr_results)} 个结果，合并后 {len(merged)} 个结果")
        return merged

    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """计算两个边界框的IoU（交并比）"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # 计算交集区域
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        # 如果没有交集
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        # 交集面积
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # 两个框的面积
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        # 并集面积
        union_area = area1 + area2 - inter_area

        # 避免除零错误
        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def _bbox_from_paddle_points(self, points: List[List[int]]) -> Tuple[int, int, int, int]:
        """从PaddleOCR的4个点坐标转换为标准矩形格式 (x1, y1, x2, y2)"""
        if not points or len(points) < 4:
            return (0, 0, 0, 0)

        # 取所有点的x,y坐标
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]

        x1 = min(x_coords)
        y1 = min(y_coords)
        x2 = max(x_coords)
        y2 = max(y_coords)

        return (x1, y1, x2, y2)

    def _text_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        if not text1 or not text2:
            return 0.0

        # 简单相似度计算：公共字符比例
        set1 = set(text1.lower())
        set2 = set(text2.lower())

        if not set1 or not set2:
            return 0.0

        intersection = set1.intersection(set2)
        union = set1.union(set2)

        return len(intersection) / len(union)

    def process_results(self, ocr_results: List[OCRResult]) -> List[OCRResult]:
        """完整的后处理流程"""
        print(f"开始后处理 {len(ocr_results)} 个OCR结果")

        # 1. 过滤低质量结果
        filtered = self.filter_results(ocr_results, min_confidence=0.1)

        # 2. 基于连续帧和IoU的去重
        deduplicated = self.deduplicate_by_continuous_frames_iou(filtered, max_frame_gap=3, iou_threshold=0.8)

        print(f"后处理完成: 最终 {len(deduplicated)} 个结果")
        return deduplicated

    def save_to_csv(self, results: List[OCRResult], output_file: str = None) -> str:
        """保存结果为CSV文件"""
        if not output_file:
            output_file = f"{self.video_name}_detected_frames_paddle_refactored.csv"

        with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(OUTPUT_CSV_HEADERS)

            for result in results:
                writer.writerow([
                    result.frame_number,
                    result.timecode,
                    result.text,
                    result.pixel_count,
                    f"{result.confidence:.3f}",
                    result.text_type
                ])

        print(f"结果已保存到: {output_file}")
        return output_file

    def get_statistics(self, results: List[OCRResult]) -> Dict[str, Any]:
        """获取处理统计信息"""
        if not results:
            return {
                "total_results": 0,
                "vfx_count": 0,
                "di_count": 0,
                "avg_confidence": 0.0,
                "frame_range": "N/A"
            }

        stats = {
            "total_results": len(results),
            "vfx_count": sum(1 for r in results if r.text_type == "VFX"),
            "di_count": sum(1 for r in results if r.text_type == "DI"),
            "avg_confidence": sum(r.confidence for r in results) / len(results) if results else 0.0,
            "frame_range": f"{min(r.frame_number for r in results)} - {max(r.frame_number for r in results)}"
        }

        return stats
