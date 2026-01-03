import cv2
import numpy as np
import json
import csv
import os
from datetime import timedelta
import argparse
from paddleocr import PaddleOCR
import colour

class VideoOCRPaddle:
    def __init__(self, video_path, lut_path=None, use_gpu=False, start_time=None, end_time=None):
        """初始化视频和 PaddleOCR"""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        # 视频信息（需要先获取fps用于时间转换）
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 处理时间范围参数
        self.start_time = start_time
        self.end_time = end_time
        self.start_frame = self.time_to_frame(start_time) if start_time else 0
        self.end_frame = self.time_to_frame(end_time) if end_time else self.frame_count

        # 验证时间范围的有效性
        if self.start_frame >= self.frame_count:
            print(f"⚠️ 警告: 起始时间 {start_time} 超出视频时长 (视频总帧数: {self.frame_count})")
            self.start_frame = 0
        if self.end_frame > self.frame_count:
            print(f"⚠️ 警告: 结束时间 {end_time} 超出视频时长 (视频总帧数: {self.frame_count})")
            self.end_frame = self.frame_count
        if self.start_frame >= self.end_frame:
            raise ValueError(f"起始时间不能晚于或等于结束时间")

        # 计算实际处理范围
        self.total_frames_to_process = self.end_frame - self.start_frame

        # LUT路径设置（默认使用LUT）
        self.lut_path = "/Users/sbr/Desktop/contrast_1.EP24.cube"
        if not os.path.exists(self.lut_path):
            print(f"⚠️ 警告: 默认LUT文件不存在: {self.lut_path}，将不使用LUT处理")
            self.lut_path = None

        # 初始化 PaddleOCR
        # 关闭方向/文档矫正默认（可按需改为 True）
        self.ocr = PaddleOCR(use_textline_orientation=False, use_doc_unwarping=False, lang='ch')

        # ROI 区域参数（和原脚本保持一致）
        self.roi_top = int(self.height * 0.06)
        self.roi_right = int(self.width * 0.40)

        # HLS 颜色范围示例（OpenCV H:0-180, L:0-255, S:0-255）
        # 这些值是示例，根据你的达芬奇设置或样本图像应当调整
        self.lower_green_hls = np.array([45, 106, 138])   # H, L, S
        self.upper_green_hls = np.array([75, 195, 255])
        self.lower_orange_hls = np.array([10, 106, 75])
        self.upper_orange_hls = np.array([25, 160, 245])

        # 检测参数
        self.pixel_threshold = 680
        self.frame_window = 5
        self.increase_threshold = 2.0

    def time_to_frame(self, time_str):
        """将时间字符串转换为帧号"""
        if not time_str:
            return 0

        # 支持多种时间格式：HH:MM:SS、MM:SS、SS、HH:MM:SS:FF
        parts = time_str.split(':')
        if len(parts) == 4:  # HH:MM:SS:FF
            hours, minutes, seconds, frames = map(int, parts)
            total_seconds = hours * 3600 + minutes * 60 + seconds
            return int(total_seconds * self.fps) + frames
        elif len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(int, parts)
            total_seconds = hours * 3600 + minutes * 60 + seconds
            return int(total_seconds * self.fps)
        elif len(parts) == 2:  # MM:SS
            minutes, seconds = map(int, parts)
            total_seconds = minutes * 60 + seconds
            return int(total_seconds * self.fps)
        elif len(parts) == 1:  # SS
            seconds = int(parts[0])
            return int(seconds * self.fps)
        else:
            raise ValueError(f"不支持的时间格式: {time_str}")

    def frame_to_smpte(self, frame_number):
        """使用实际视频 FPS 将帧号转换为 SMPTE 时间码"""
        total_seconds = frame_number / float(self.fps)
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        frames = int(round((total_seconds - int(total_seconds)) * self.fps))
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"

    def get_colored_pixel_count(self, frame):
        """获取 ROI 区域中目标颜色像素并返回过滤后的 ROI（numpy array）"""
        roi = frame[0:self.roi_top, self.roi_right:self.width]
        # 使用 HLS（OpenCV 名称 HLS，即 H,L,S）
        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
        green_mask = cv2.inRange(hls, self.lower_green_hls, self.upper_green_hls)
        orange_mask = cv2.inRange(hls, self.lower_orange_hls, self.upper_orange_hls)

        # 可选形态学去噪，减少小斑点（更稳定）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)

        green_only = cv2.bitwise_and(roi, roi, mask=green_mask)
        orange_only = cv2.bitwise_and(roi, roi, mask=orange_mask)

        green_count = cv2.countNonZero(green_mask)
        orange_count = cv2.countNonZero(orange_mask)

        results = []
        if green_count > self.pixel_threshold:
            results.append(("VFX", green_count, roi))
        if orange_count > self.pixel_threshold:
            results.append(("DI", orange_count, roi))
        return results

    def process_text(self, text, text_type):
        """和原脚本一致的文本规范化处理"""
        if not text:
            return ""
        if text_type == "VFX":
            text = text.replace('VEX:', 'VFX:')
            text = text.replace('VFX;', 'VFX:')
            text = text.replace('VEX;', 'VFX:')
            if not text.startswith('VFX:'):
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
        else:
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

    def apply_lut_processing(self, image_bgr, lut_path):
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

    def analyze_video_with_ocr(self):
        """主循环：读取视频帧、颜色触发、内存传递 ROI 给 PaddleOCR"""
        pixel_history_green = []
        pixel_history_orange = []
        detected_frames = []
        frame_number = 0
        last_detected_count = {'VFX': 0, 'DI': 0}
        frames_since_last_detection = {'VFX': 0, 'DI': 0}

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # 检查是否在指定的时间范围内
                if frame_number < self.start_frame:
                    frame_number += 1
                    frames_since_last_detection['VFX'] += 1
                    frames_since_last_detection['DI'] += 1
                    continue
                if frame_number > self.end_frame:
                    break

                # 计算基于处理范围的进度
                processed_frames = frame_number - self.start_frame
                if processed_frames % 100 == 0 and self.total_frames_to_process > 0:
                    progress = (processed_frames / self.total_frames_to_process) * 100
                    print(f"\r处理进度: {progress:.2f}% ({processed_frames}/{self.total_frames_to_process} 帧)", end="")

                color_results = self.get_colored_pixel_count(frame)
                for text_type, pixel_count, filtered_roi in color_results:
                    pixel_history = pixel_history_green if text_type == 'VFX' else pixel_history_orange
                    if len(pixel_history) >= self.frame_window:
                        pixel_history.pop(0)
                    pixel_history.append(pixel_count)

                    if len(pixel_history) >= self.frame_window:
                        avg_history = sum(pixel_history) / len(pixel_history)
                        # 计算10秒对应的帧数（假设25fps）
                        ten_seconds_frames = int(10 * self.fps)

                        should_detect = (
                            pixel_count > self.pixel_threshold and
                            (
                                (pixel_count > avg_history * self.increase_threshold and
                                 all(pixel_count > x * self.increase_threshold for x in pixel_history[-3:]))
                                or
                                (pixel_count < avg_history * 0.8 and  # 下降到平均值的70%以下
                                 all(pixel_count < x * 0.8 for x in pixel_history[-3:]))  # 比最近3帧都小
                                or
                                (frames_since_last_detection[text_type] >= 25 and
                                (frames_since_last_detection[text_type] >= ten_seconds_frames or  # 超过10秒直接检测
                                  abs(pixel_count - last_detected_count[text_type]) > last_detected_count[text_type] * 0.3))
                            )
                        )

                        if should_detect:
                            # 以前直接把 numpy array 传给 PaddleOCR（内存传递）：
                            # try:
                            #     res = self.ocr.predict(filtered_roi)
                            # except Exception:
                            #     res = None
                            # 改为：把 ROI 保存为 PNG 到项目下的 tmp/ 目录，再用文件路径调用 OCR（保留 PNG，便于后续手动检查颜色取值）
                            try:
                                tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')
                                os.makedirs(tmp_dir, exist_ok=True)
                                fname = f"roi_{frame_number}_{text_type}.png"
                                tmp_path = os.path.join(tmp_dir, fname)

                                # 如果指定了LUT，应用LUT处理
                                if self.lut_path:
                                    processed_roi = self.apply_lut_processing(filtered_roi, self.lut_path)
                                    cv2.imwrite(tmp_path, processed_roi)
                                    # cv2.imwrite(tmp_path, filtered_roi)
                                    print(f"已应用LUT处理: {tmp_path}")
                                else:
                                    cv2.imwrite(tmp_path, filtered_roi)
                                # 使用文件路径调用 PaddleOCR
                                res = self.ocr.predict(tmp_path)
                                # 注意：不删除 tmp 下的文件，便于后续检查
                                if res:
                                    text_parts = []
                                    confidences = []
                                    for item in res:
                                        texts = item.get('rec_texts', [])
                                        scores = item.get('rec_scores', [])
                                        for t, s in zip(texts, scores):
                                            if t:
                                                text_parts.append(t.strip())
                                                confidences.append(float(s))
                                    if text_parts:
                                        full_text = ''.join(filter(None, text_parts))
                                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                                        processed_text = self.process_text(full_text, text_type)
                                        if processed_text.strip():
                                            detected_frames.append({
                                                "frame_number": frame_number,
                                                "timecode": self.frame_to_smpte(frame_number),
                                                "text": processed_text,
                                                "pixel_count": int(pixel_count),
                                                "confidence": float(avg_confidence),
                                                "type": text_type,
                                                "roi_png": tmp_path
                                            })
                                            last_detected_count[text_type] = pixel_count
                                            frames_since_last_detection[text_type] = 0
                                            print(f"\n检测到新文本 帧:{frame_number} 类型:{text_type} 像素:{int(pixel_count)} 置信度:{avg_confidence:.2f} 文本:{processed_text} png:{tmp_path}")
                                        else:
                                            print(f"\r跳过帧 {frame_number}: 未识别到文本", end="")
                                    else:
                                        print(f"\r跳过帧 {frame_number}: OCR无结果", end="")
                                else:
                                    print(f"\r跳过帧 {frame_number}: OCR返回空", end="")
                            except Exception as e:
                                print(f"\nOCR错误 在帧 {frame_number}: {str(e)}")
                frame_number += 1
                frames_since_last_detection['VFX'] += 1
                frames_since_last_detection['DI'] += 1

            print("\n分析完成")
            return detected_frames
        finally:
            # 释放资源
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()

    def save_results_csv(self, results):
        """保存结果为 CSV（与原脚本一致）"""
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        output_file = f"{video_name}_detected_frames_paddle.csv"
        with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['帧数', '时间码', '文本内容', '像素数量', '置信度', '类型'])
            for frame in results:
                writer.writerow([
                    frame['frame_number'],
                    frame['timecode'],
                    frame['text'],
                    frame['pixel_count'],
                    f"{frame['confidence']:.3f}",
                    frame['type']
                ])

def main():
    parser = argparse.ArgumentParser(description='视频字幕 OCR（Paddle）')
    parser.add_argument('--video_path', '-v', type=str, required=True, help='视频文件路径')
    parser.add_argument('--lut_path', '-l', type=str, help='LUT文件路径 (.cube格式，用于图像增强)')
    parser.add_argument('--start_time', '-s', type=str, help='处理起始时间 (格式: HH:MM:SS 或 MM:SS 或 SS 或 HH:MM:SS:FF)')
    parser.add_argument('--end_time', '-e', type=str, help='处理结束时间 (格式: HH:MM:SS 或 MM:SS 或 SS 或 HH:MM:SS:FF)')
    args = parser.parse_args()

    try:
        processor = VideoOCRPaddle(args.video_path, start_time=args.start_time, end_time=args.end_time)

        time_info = ""
        if args.start_time or args.end_time:
            actual_start_frame = processor.start_frame
            actual_end_frame = processor.end_frame
            start_time_display = args.start_time or "00:00:00"
            end_time_display = args.end_time or f"{processor.frame_to_smpte(processor.frame_count)}"
            time_info = f" (时间范围: {start_time_display} - {end_time_display}, 帧范围: {actual_start_frame}-{actual_end_frame})"

        print(f"开始分析视频: {args.video_path}{time_info}")
        results = processor.analyze_video_with_ocr()
        processor.save_results_csv(results)
        print(f"\n检测到 {len(results)} 个字幕，结果已保存为 CSV")
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main()


