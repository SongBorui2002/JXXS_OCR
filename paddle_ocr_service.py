"""
PaddleOCR服务
负责批量OCR处理
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from video_preprocessor import FrameData
from config import *

# 延迟导入PaddleOCR，避免在没有安装时导入失败
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("警告: PaddleOCR未安装，将使用模拟模式")

@dataclass
class OCRResult:
    """OCR结果数据结构"""
    frame_number: int
    timecode: str
    text: str
    pixel_count: int
    confidence: float
    text_type: str
    bbox: Tuple[int, int, int, int]  # 边界框坐标 (x1, y1, x2, y2)
    roi_png_path: str
    raw_ocr_data: Dict[str, Any]  # 保存原始OCR数据用于调试

class PaddleOCRService:
    """PaddleOCR服务类"""

    def __init__(self):
        """初始化PaddleOCR服务"""
        # 初始化PaddleOCR
        if PADDLEOCR_AVAILABLE:
            self.ocr = PaddleOCR(
                use_textline_orientation=OCR_USE_TEXTLINE_ORIENTATION,
                use_doc_unwarping=OCR_USE_DOC_UNWARPER,
                lang=OCR_LANG
            )
        else:
            self.ocr = None
            print("⚠️ PaddleOCR不可用，使用模拟模式")

        # 创建临时目录
        self.tmp_dir = TMP_DIR
        os.makedirs(self.tmp_dir, exist_ok=True)

        print("PaddleOCR服务初始化完成")

    def process_single_frame(self, frame_data: FrameData) -> Optional[OCRResult]:
        """处理单个帧的OCR"""
        try:
            # 从字节流重建图像
            image_array = np.frombuffer(frame_data.image_bytes, dtype=np.uint8)
            roi_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if roi_image is None:
                print(f"图像解码失败: 帧{frame_data.frame_number}")
                return None

            # 转换为RGB格式（PaddleOCR期望的格式）
            roi_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)

            # 调用PaddleOCR（直接用numpy数组）
            if PADDLEOCR_AVAILABLE and self.ocr:
                ocr_result = self.ocr.predict(roi_image)
            else:
                # 模拟OCR结果
                ocr_result = [{
                    'rec_texts': [f"模拟OCR结果_{frame_data.frame_number}"],
                    'rec_scores': [0.85]
                }]

            if not ocr_result:
                print(f"跳过帧 {frame_data.frame_number}: OCR返回空")
                return None

            # 解析OCR结果
            text_parts = []
            confidences = []
            bboxes = []
            raw_data = []

            for item in ocr_result:
                texts = item.get('rec_texts', [])
                scores = item.get('rec_scores', [])

                # 使用正确的bbox字段名
                boxes = item.get('rec_polys', [])

            for i, (t, s) in enumerate(zip(texts, scores)):
                if t:
                    text_parts.append(t.strip())
                    confidences.append(float(s))

                    # 提取bbox坐标
                    if i < len(boxes) and boxes[i] is not None:
                        box = boxes[i]
                        # PaddleOCR返回的bbox通常是4个点的坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                        # 或者numpy数组格式
                        try:
                            if isinstance(box, np.ndarray):
                                if box.shape == (4, 2):  # 4个点的坐标
                                    points = box
                                elif box.shape == (8,):  # 展平的8个坐标
                                    points = box.reshape(4, 2)
                                else:
                                    raise ValueError(f"Unexpected box shape: {box.shape}")
                            elif isinstance(box, list) and len(box) == 4:
                                # 列表格式 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                                points = np.array(box)
                            elif isinstance(box, list) and len(box) == 8:
                                # 展平的坐标 [x1,y1,x2,y2,x3,y3,x4,y4]
                                points = np.array(box).reshape(4, 2)
                            else:
                                raise ValueError(f"Unexpected box format: {box}")

                            # 计算边界框
                            x_coords = points[:, 0]
                            y_coords = points[:, 1]
                            x1, y1 = int(x_coords.min()), int(y_coords.min())
                            x2, y2 = int(x_coords.max()), int(y_coords.max())
                            bbox = (x1, y1, x2, y2)
                        except Exception as e:
                            bbox = (0, 0, 0, 0)
                    else:
                        bbox = (0, 0, 0, 0)  # 默认bbox
                        print(f"DEBUG: bbox不存在，使用默认值")

                    bboxes.append(bbox)
                    raw_data.append({
                        'text': t.strip(),
                        'score': float(s),
                        'bbox': bbox
                    })

            if not text_parts:
                print(f"跳过帧 {frame_data.frame_number}: 未识别到文本")
                return None

            # 合并文本，计算平均bbox（取最大的bbox作为代表）
            full_text = ''.join(filter(None, text_parts))
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            # 选择最大的bbox作为代表（适用于多行文本的情况）
            if bboxes:
                max_area = 0
                selected_bbox = bboxes[0]
                for bbox in bboxes:
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area > max_area:
                        max_area = area
                        selected_bbox = bbox
                representative_bbox = selected_bbox
            else:
                representative_bbox = (0, 0, 0, 0)

            # 创建OCR结果
            result = OCRResult(
                frame_number=frame_data.frame_number,
                timecode=frame_data.timecode,
                text=full_text,
                pixel_count=frame_data.pixel_count,
                confidence=avg_confidence,
                text_type=frame_data.text_type,
                bbox=representative_bbox,
                roi_png_path="",  # 字节流传递，无临时文件
                raw_ocr_data={'items': raw_data, 'avg_confidence': avg_confidence}
            )

            print(f"OCR成功 帧:{frame_data.frame_number} 类型:{frame_data.text_type} "
                  f"像素:{frame_data.pixel_count} 置信度:{avg_confidence:.2f} 文本:{full_text}")

            return result

        except Exception as e:
            print(f"OCR错误 在帧 {frame_data.frame_number}: {str(e)}")
            return None

    def process_batch(self, frame_batch: List[FrameData]) -> List[OCRResult]:
        """批量处理OCR"""
        results = []

        for frame_data in frame_batch:
            result = self.process_single_frame(frame_data)
            if result:
                results.append(result)

        print(f"批处理完成: 处理 {len(frame_batch)} 帧，成功识别 {len(results)} 帧")
        return results
