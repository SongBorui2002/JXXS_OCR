"""
PaddleOCR服务
负责批量OCR处理
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Any, Optional
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
            raw_data = []

            for item in ocr_result:
                texts = item.get('rec_texts', [])
                scores = item.get('rec_scores', [])

                for t, s in zip(texts, scores):
                    if t:
                        text_parts.append(t.strip())
                        confidences.append(float(s))
                        raw_data.append({'text': t.strip(), 'score': float(s)})

            if not text_parts:
                print(f"跳过帧 {frame_data.frame_number}: 未识别到文本")
                return None

            # 合并文本
            full_text = ''.join(filter(None, text_parts))
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            # 创建OCR结果
            result = OCRResult(
                frame_number=frame_data.frame_number,
                timecode=frame_data.timecode,
                text=full_text,
                pixel_count=frame_data.pixel_count,
                confidence=avg_confidence,
                text_type=frame_data.text_type,
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
