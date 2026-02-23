"""
主协调器：协调整个视频处理流程
"""

import time
import argparse
import cv2
import numpy as np
import os
import glob
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor
from video_preprocessor import VideoPreprocessor, FrameData
from paddle_ocr_service import PaddleOCRService, OCRResult
from result_processor import ResultProcessor
from config import BATCH_SIZE, MAX_WORKERS, TMP_DIR


def process_ocr_batch_parallel(frame_data_batch: List[FrameData]) -> List[OCRResult]:
    """在子进程中处理单个OCR批次（模块级函数，避免序列化问题）"""
    try:
        # 为每个子进程创建独立的OCR服务实例
        ocr_service = PaddleOCRService()

        # OCR处理
        ocr_results = []
        if frame_data_batch:
            ocr_results = ocr_service.process_batch(frame_data_batch)

        return ocr_results

    except Exception as e:
        print(f"OCR子进程处理错误: {e}")
        return []


class MainCoordinator:
    """主进程协调器"""

    def __init__(self, video_path: str, lut_path: Optional[str] = None,
                 start_time: Optional[str] = None, end_time: Optional[str] = None):
        """初始化协调器"""
        self.video_path = video_path
        self.lut_path = lut_path
        self.start_time = start_time
        self.end_time = end_time

        # 初始化服务
        self.preprocessor = VideoPreprocessor(video_path, start_time, end_time, lut_path)
        self.ocr_service = PaddleOCRService()
        self.result_processor = ResultProcessor(video_path)

        print("主协调器初始化完成")

    def run(self, parallel: bool = True) -> str:
        """运行完整的处理流程"""
        start_time = time.time()

        try:
            # 选择处理模式
            if parallel and self.preprocessor.total_frames_to_process > 1000:  # 长视频使用并行
                print("检测到长视频，使用并行处理模式")
                results = self.process_video_parallel()
            else:
                print("使用顺序处理模式")
                results = self.process_video_sequential()

            # 完整的后处理流程（包括过滤和去重）
            filtered_results = self.result_processor.process_results(results)

            # 保存结果
            output_file = self.result_processor.save_to_csv(filtered_results)

            # 显示统计信息
            stats = self.result_processor.get_statistics(results)
            elapsed_time = time.time() - start_time

            print("\n=== 处理完成 ===")
            print(f"总耗时: {elapsed_time:.2f} 秒")
            print(f"检测到字幕: {stats['total_results']} 个")
            print(f"VFX字幕: {stats['vfx_count']} 个")
            print(f"DI字幕: {stats['di_count']} 个")
            print(f"帧范围: {stats['frame_range']}")
            print(f"结果文件: {output_file}")

            # 清理临时文件
            self._cleanup_tmp_files()

            return output_file

        except Exception as e:
            print(f"处理过程中发生错误: {e}")
            raise

    def _cleanup_tmp_files(self):
        """清理临时目录中的临时文件"""
        try:
            if os.path.exists(TMP_DIR):
                # 删除 tmp 目录下的所有 png 文件
                png_files = glob.glob(os.path.join(TMP_DIR, "*.png"))
                if png_files:
                    count = len(png_files)
                    for f in png_files:
                        try:
                            os.remove(f)
                        except Exception:
                            pass
                    print(f"\n已清理 {count} 个临时文件")
        except Exception as e:
            print(f"\n清理临时文件失败: {e}")

    def process_video_sequential(self) -> List[OCRResult]:
        """顺序处理视频（适合短视频或调试）"""
        print("使用顺序处理模式")

        # 顺序处理所有帧
        ocr_tasks = self._sequential_preprocess_frames()

        # 顺序OCR处理
        all_results = []
        for task in ocr_tasks:
            result = self.ocr_service.process_single_frame(task)
            if result:
                all_results.append(result)

        return all_results

    def process_video_parallel(self) -> List[OCRResult]:
        """并行处理视频（适合长视频）"""
        print("开始并行处理视频...")

        # 阶段1: 顺序预处理
        ocr_tasks = self._sequential_preprocess_frames()

        # 阶段2: 并发OCR处理
        results = self._concurrent_batch_ocr(ocr_tasks)

        return results

    def _sequential_preprocess_frames(self) -> List[FrameData]:
        """顺序读取视频帧，逐帧进行预处理和颜色检测，积累需要OCR的帧数据"""
        print("阶段1: 顺序预处理视频帧...")
        ocr_tasks: List[FrameData] = []

        # 使用预处理器的 VideoCapture，避免重复打开
        cap = self.preprocessor.cap
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")

        start_frame = self.preprocessor.start_frame
        end_frame = self.preprocessor.end_frame
        total_frames_to_process = self.preprocessor.total_frames_to_process

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_number = start_frame
        processed_count = 0

        try:
            while frame_number < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_data = self._preprocess_single_frame(frame, frame_number)
                if frame_data:
                    ocr_tasks.append(frame_data)

                processed_count += 1
                progress = self.preprocessor.get_progress_info(processed_count)
                print(f"\r预处理进度: {progress} ({processed_count}/{total_frames_to_process} 帧)", end="", flush=True)
                frame_number += 1
        finally:
            # 确保 VideoCapture 被释放（虽然预处理器的 __del__ 也会释放）
            pass  # cap 由 VideoPreprocessor 管理

        print(f"\n预处理完成，获得 {len(ocr_tasks)} 个OCR任务")
        return ocr_tasks

    def _preprocess_single_frame(self, frame: np.ndarray, frame_number: int) -> Optional[FrameData]:
        """预处理单帧：颜色检测，决定是否需要OCR"""
        # 使用预处理器的颜色检测逻辑
        color_results = self.preprocessor.get_colored_pixel_count(frame)

        for text_type, pixel_count, filtered_roi in color_results:
            # 检查是否应该进行OCR检测（已包含采样逻辑）
            if self.preprocessor.should_detect_ocr(text_type, pixel_count):
                # 应用LUT处理
                processed_roi = filtered_roi
                if self.preprocessor.lut_available and self.preprocessor.lut_path:
                    try:
                        processed_roi = self.preprocessor.apply_lut_processing(filtered_roi, self.preprocessor.lut_path)
                        print(f"\n已应用LUT处理: 帧 {frame_number} ({text_type})")
                    except Exception as e:
                        print(f"\nLUT处理失败，使用原图: {e}")

                # 将处理后的图像编码为字节流
                success, encoded_img = cv2.imencode('.png', processed_roi)
                if success:
                    image_bytes = encoded_img.tobytes()
                    frame_data = FrameData(
                        frame_number=frame_number,
                        timecode=self.preprocessor.frame_to_smpte(frame_number),
                        image_bytes=image_bytes,
                        pixel_count=pixel_count,
                        text_type=text_type,
                        image_shape=processed_roi.shape
                    )
                    return frame_data
                else:
                    print(f"图像编码失败: 帧{frame_number}")
                    return None

        return None

    def _concurrent_batch_ocr(self, ocr_tasks: List[FrameData]) -> List[OCRResult]:
        """并发处理OCR批次"""
        if not ocr_tasks:
            return []

        # 分批处理
        ocr_batches = []
        for i in range(0, len(ocr_tasks), BATCH_SIZE):
            batch = ocr_tasks[i:i + BATCH_SIZE]
            ocr_batches.append(batch)

        print(f"OCR任务分批: {len(ocr_tasks)} 个任务 → {len(ocr_batches)} 个批次")

        # 使用进程池并发处理OCR批次
        all_ocr_results = []
        with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(ocr_batches))) as executor:
            # 提交所有OCR批次任务
            future_to_batch = {}
            for batch in ocr_batches:
                future = executor.submit(process_ocr_batch_parallel, batch)
                future_to_batch[future] = batch

            # 收集结果
            completed_batches = 0
            for future in future_to_batch:
                try:
                    batch_results = future.result()
                    all_ocr_results.extend(batch_results)

                    completed_batches += 1
                    progress_percentage = (completed_batches / len(ocr_batches)) * 100
                    progress = f"OCR进度: {progress_percentage:.2f}% ({completed_batches}/{len(ocr_batches)} 批次)"
                    print(f"\r{progress}", end="", flush=True)

                except Exception as e:
                    batch = future_to_batch[future]
                    print(f"OCR批次处理失败: {e}")

        print(f"\rOCR进度: 100.00% ({len(ocr_batches)}/{len(ocr_batches)} 批次)")
        return all_ocr_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='重构版视频字幕OCR系统')
    parser.add_argument('--video_path', '-v', type=str, required=True, help='视频文件路径')
    parser.add_argument('--lut_path', '-l', type=str, help='LUT文件路径')
    parser.add_argument('--start_time', '-s', type=str, help='开始时间 (HH:MM:SS 或 MM:SS 或 SS)')
    parser.add_argument('--end_time', '-e', type=str, help='结束时间 (HH:MM:SS 或 MM:SS 或 SS)')
    parser.add_argument('--sequential', action='store_true', help='强制使用顺序处理模式')

    args = parser.parse_args()

    try:
        # 创建协调器
        coordinator = MainCoordinator(
            args.video_path,
            args.lut_path,
            args.start_time,
            args.end_time
        )

        # 显示处理信息
        time_info = ""
        if args.start_time or args.end_time:
            actual_start_frame = coordinator.preprocessor.start_frame
            actual_end_frame = coordinator.preprocessor.end_frame
            start_time_display = args.start_time or "00:00:00"
            end_time_display = args.end_time or f"{coordinator.preprocessor.frame_to_smpte(coordinator.preprocessor.frame_count)}"
            time_info = f" (时间范围: {start_time_display} - {end_time_display}, 帧范围: {actual_start_frame}-{actual_end_frame})"

        print(f"开始分析视频: {args.video_path}{time_info}")

        # 运行处理
        use_parallel = not args.sequential
        output_file = coordinator.run(parallel=use_parallel)

        print(f"\n成功完成！结果已保存至: {output_file}")

    except Exception as e:
        print(f"错误: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())