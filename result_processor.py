"""
信息处理服务
负责结果过滤、合并、规范化
"""

import csv
import os
from typing import List, Dict, Any
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
            if not text.startswith('DI:'):
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

        # 2. 去重
        deduplicated = self.deduplicate_results(filtered)

        # 3. 合并相似文本
        final_results = self.merge_similar_texts(deduplicated)

        print(f"后处理完成: 最终 {len(final_results)} 个结果")
        return final_results

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
