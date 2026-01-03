"""
视频预处理服务
负责视频解码、帧提取、颜色检测
"""

import cv2
import numpy as np
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
from config import *
import colour

@dataclass
class FrameData:
    """帧数据结构"""
    frame_number: int
    timecode: str
    image_bytes: bytes  # 已编码的图像字节流（可序列化）
    pixel_count: int
    text_type: str  # 'VFX' or 'DI'
    image_shape: tuple  # 图像形状信息 (height, width, channels)

@dataclass
class VideoInfo:
    """视频信息"""
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float

class VideoPreprocessor:
    """视频预处理服务"""

    def __init__(self, video_path: str, start_time: Optional[str] = None, end_time: Optional[str] = None, lut_path: Optional[str] = None):
        """初始化视频预处理器"""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        # 获取视频信息
        self.video_info = self._get_video_info()

        # 处理时间范围
        self.start_frame = self.time_to_frame(start_time) if start_time else 0
        self.end_frame = self.time_to_frame(end_time) if end_time else self.video_info.frame_count

        # 验证时间范围
        self._validate_time_range(start_time, end_time)

        # 计算处理范围
        self.total_frames_to_process = self.end_frame - self.start_frame

        # LUT路径设置
        self.lut_path = lut_path or DEFAULT_LUT_PATH
        self.lut_available = os.path.exists(self.lut_path) if self.lut_path else False

        if self.lut_path and not self.lut_available:
            print(f"⚠️ 警告: LUT文件不存在: {self.lut_path}，将不使用LUT处理")

        # ROI参数
        self.roi_top = int(self.video_info.height * ROI_TOP_RATIO)
        self.roi_right = int(self.video_info.width * ROI_RIGHT_RATIO)

        # 初始化检测历史
        self.pixel_history_green = []
        self.pixel_history_orange = []
        self.last_detected_count = {'VFX': 0, 'DI': 0}
        self.frames_since_last_detection = {'VFX': 0, 'DI': 0}

        print(f"视频预处理器初始化完成: {video_path}")
        print(f"视频信息: {self.video_info.fps}fps, {self.video_info.width}x{self.video_info.height}")
        if self.lut_available:
            print(f"LUT增强已启用: {self.lut_path}")
        print(f"处理范围: 帧 {self.start_frame} - {self.end_frame} (共 {self.total_frames_to_process} 帧)")

    def _get_video_info(self) -> VideoInfo:
        """获取视频基本信息"""
        fps = self.cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_seconds = frame_count / fps

        return VideoInfo(
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            duration_seconds=duration_seconds
        )

    def _validate_time_range(self, start_time: Optional[str], end_time: Optional[str]):
        """验证时间范围的有效性"""
        if self.start_frame >= self.video_info.frame_count:
            print(f"⚠️ 警告: 起始时间 {start_time} 超出视频时长 (视频总帧数: {self.video_info.frame_count})")
            self.start_frame = 0
        if self.end_frame > self.video_info.frame_count:
            print(f"⚠️ 警告: 结束时间 {end_time} 超出视频时长 (视频总帧数: {self.video_info.frame_count})")
            self.end_frame = self.video_info.frame_count
        if self.start_frame >= self.end_frame:
            raise ValueError(f"起始时间不能晚于或等于结束时间")

    def time_to_frame(self, time_str: Optional[str]) -> int:
        """将时间字符串转换为帧号"""
        if not time_str:
            return 0

        # 支持多种时间格式：HH:MM:SS、MM:SS、SS、HH:MM:SS:FF
        parts = time_str.split(':')
        if len(parts) == 4:  # HH:MM:SS:FF
            hours, minutes, seconds, frames = map(int, parts)
            total_seconds = hours * 3600 + minutes * 60 + seconds
            return int(total_seconds * self.video_info.fps) + frames
        elif len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(int, parts)
            total_seconds = hours * 3600 + minutes * 60 + seconds
            return int(total_seconds * self.video_info.fps)
        elif len(parts) == 2:  # MM:SS
            minutes, seconds = map(int, parts)
            total_seconds = minutes * 60 + seconds
            return int(total_seconds * self.video_info.fps)
        elif len(parts) == 1:  # SS
            seconds = int(parts[0])
            return int(seconds * self.video_info.fps)
        else:
            raise ValueError(f"不支持的时间格式: {time_str}")

    def frame_to_smpte(self, frame_number: int) -> str:
        """将帧号转换为SMPTE时间码"""
        total_seconds = frame_number / float(self.video_info.fps)
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        frames = int(round((total_seconds - int(total_seconds)) * self.video_info.fps))
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"

    @staticmethod
    def time_to_frame_static(time_str: str, fps: float = 25.0) -> int:
        """静态方法：将时间字符串转换为帧号"""
        if not time_str:
            return 0

        parts = time_str.split(':')
        if len(parts) == 4:  # HH:MM:SS:FF
            hours, minutes, seconds, frames = map(int, parts)
            total_seconds = hours * 3600 + minutes * 60 + seconds
            return int(total_seconds * fps) + frames
        elif len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(int, parts)
            total_seconds = hours * 3600 + minutes * 60 + seconds
            return int(total_seconds * fps)
        elif len(parts) == 2:  # MM:SS
            minutes, seconds = map(int, parts)
            total_seconds = minutes * 60 + seconds
            return int(total_seconds * fps)
        elif len(parts) == 1:  # SS
            seconds = int(parts[0])
            return int(seconds * fps)
        else:
            raise ValueError(f"不支持的时间格式: {time_str}")

    @staticmethod
    def frame_to_smpte_static(frame_number: int, fps: float = 25.0) -> str:
        """静态方法：将帧号转换为SMPTE时间码"""
        total_seconds = frame_number / float(fps)
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        frames = int(round((total_seconds - int(total_seconds)) * fps))
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"

    def apply_lut_processing(self, image_bgr: np.ndarray, lut_path: str) -> np.ndarray:
        """
        应用LUT处理到图像
        输入: BGR格式图像, LUT文件路径
        输出: 处理后的BGR格式图像
        """
        try:
            # 转换为RGB并标准化
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_normalized = image_rgb.astype(np.float32) / 255.0

            # 加载LUT
            lut_3d = colour.io.read_LUT(lut_path)

            # 应用LUT
            try:
                processed_image = lut_3d.apply(image_normalized)
            except:
                # 备用方法
                height, width, channels = image_normalized.shape
                image_reshaped = image_normalized.reshape(-1, channels)
                processed_reshaped = colour.algebra.table_interpolation_trilinear(
                    image_reshaped, lut_3d.table
                )
                processed_image = processed_reshaped.reshape(height, width, channels)

            # 裁剪到有效范围
            processed_image = np.clip(processed_image, 0.0, 1.0)

            # 转换回uint8格式并转为BGR
            processed_uint8 = (processed_image * 255).astype(np.uint8)
            processed_bgr = cv2.cvtColor(processed_uint8, cv2.COLOR_RGB2BGR)

            return processed_bgr

        except Exception as e:
            raise Exception(f"LUT处理失败: {str(e)}")

    def get_colored_pixel_count(self, frame: np.ndarray) -> List[Tuple[str, int, np.ndarray]]:
        """获取ROI区域中目标颜色像素并返回过滤后的ROI"""
        roi = frame[0:self.roi_top, self.roi_right:self.video_info.width]
        # 使用HLS颜色空间
        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
        green_mask = cv2.inRange(hls, LOWER_GREEN_HLS, UPPER_GREEN_HLS)
        orange_mask = cv2.inRange(hls, LOWER_ORANGE_HLS, UPPER_ORANGE_HLS)

        # 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)

        green_only = cv2.bitwise_and(roi, roi, mask=green_mask)
        orange_only = cv2.bitwise_and(roi, roi, mask=orange_mask)

        green_count = cv2.countNonZero(green_mask)
        orange_count = cv2.countNonZero(orange_mask)

        results = []
        if green_count > PIXEL_THRESHOLD:
            results.append(("VFX", green_count, green_only))
        if orange_count > PIXEL_THRESHOLD:
            results.append(("DI", orange_count, orange_only))
        return results

    def should_detect_ocr(self, text_type: str, pixel_count: int) -> bool:
        """判断是否应该进行OCR检测"""
        pixel_history = self.pixel_history_green if text_type == 'VFX' else self.pixel_history_orange

        if len(pixel_history) < FRAME_WINDOW:
            return False

        avg_history = sum(pixel_history) / len(pixel_history)
        ten_seconds_frames = int(10 * self.video_info.fps)

        should_detect = (
            pixel_count > PIXEL_THRESHOLD and
            (
                (pixel_count > avg_history * INCREASE_THRESHOLD and
                 all(pixel_count > x * INCREASE_THRESHOLD for x in pixel_history[-3:]))
                or
                (pixel_count < avg_history * 0.8 and
                 all(pixel_count < x * 0.8 for x in pixel_history[-3:]))
                or
                (self.frames_since_last_detection[text_type] >= MIN_DETECTION_INTERVAL and
                 (self.frames_since_last_detection[text_type] >= ten_seconds_frames or
                  abs(pixel_count - self.last_detected_count[text_type]) > self.last_detected_count[text_type] * 0.3))
            )
        )

        return should_detect

    def process_frames_batch(self, batch_frames: List[int]) -> List[FrameData]:
        """批量处理帧，返回需要OCR的帧数据"""
        ocr_frames = []

        for frame_number in batch_frames:
            # 设置视频位置
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()

            if not ret:
                continue

            # 颜色检测
            color_results = self.get_colored_pixel_count(frame)

            for text_type, pixel_count, filtered_roi in color_results:
                # 更新历史记录
                pixel_history = self.pixel_history_green if text_type == 'VFX' else self.pixel_history_orange
                if len(pixel_history) >= FRAME_WINDOW:
                    pixel_history.pop(0)
                pixel_history.append(pixel_count)

                # 判断是否需要OCR
                if self.should_detect_ocr(text_type, pixel_count):
                    # 应用LUT处理（如果可用）
                    processed_roi = filtered_roi
                    if self.lut_available and self.lut_path:
                        try:
                            processed_roi = self.apply_lut_processing(filtered_roi, self.lut_path)
                            print(f"\n已应用LUT处理: 帧 {frame_number} ({text_type})")
                        except Exception as e:
                            print(f"\nLUT处理失败，使用原图: {e}")

                    # 将处理后的图像编码为字节流（可序列化）
                    success, encoded_img = cv2.imencode('.png', processed_roi)
                    if success:
                        image_bytes = encoded_img.tobytes()
                        frame_data = FrameData(
                            frame_number=frame_number,
                            timecode=self.frame_to_smpte(frame_number),
                            image_bytes=image_bytes,  # 字节流
                            pixel_count=pixel_count,
                            text_type=text_type,
                            image_shape=processed_roi.shape  # 保存形状信息
                        )
                    else:
                        print(f"图像编码失败: 帧{frame_number}")
                        continue
                    ocr_frames.append(frame_data)

                    # 更新检测记录
                    self.last_detected_count[text_type] = pixel_count
                    self.frames_since_last_detection[text_type] = 0

            # 更新帧计数器
            self.frames_since_last_detection['VFX'] += 1
            self.frames_since_last_detection['DI'] += 1

        return ocr_frames

    def get_all_frames_to_process(self) -> List[int]:
        """获取所有需要处理的帧号列表"""
        return list(range(self.start_frame, self.end_frame))

    def get_progress_info(self, processed_frames: int) -> str:
        """获取处理进度信息"""
        if self.total_frames_to_process > 0:
            progress = (processed_frames / self.total_frames_to_process) * 100
            return f"{progress:.2f}%"
        return "进度计算中..."

    def __del__(self):
        """析构函数，确保释放视频资源"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
