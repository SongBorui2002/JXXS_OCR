"""
配置文件：定义所有通用参数和设置
"""

import numpy as np

# 视频处理参数
DEFAULT_FPS = 25.0

# ROI 区域参数
ROI_TOP_RATIO = 0.06  # 视频高度的6%
ROI_RIGHT_RATIO = 0.40  # 视频宽度的40%

# HLS 颜色范围参数
# 绿色 (VFX)
LOWER_GREEN_HLS = np.array([45, 106, 138])  # H, L, S
UPPER_GREEN_HLS = np.array([75, 195, 255])

# 橙色 (DI)
LOWER_ORANGE_HLS = np.array([10, 106, 75])
UPPER_ORANGE_HLS = np.array([25, 160, 245])

# 检测参数
PIXEL_THRESHOLD = 680  # 像素阈值
FRAME_WINDOW = 5  # 滑动窗口大小
INCREASE_THRESHOLD = 2.0  # 增长阈值

# LUT文件路径
DEFAULT_LUT_PATH = "/Users/sbr/Desktop/JXXS_OCR/JXXS_OCR.cube"

# PaddleOCR 参数
OCR_LANG = 'ch'
OCR_USE_TEXTLINE_ORIENTATION = False
OCR_USE_DOC_UNWARPER = False

# 批处理参数
BATCH_SIZE = 20  # OCR批处理大小，根据测试结果调整
MAX_WORKERS = 3  # 并发PaddleOCR实例数量，根据并发测试结果调整

# 时间参数
MIN_DETECTION_INTERVAL = 25  # 最短检测间隔(帧)
MAX_DETECTION_INTERVAL = 10 * 25  # 最长检测间隔(10秒*25fps)

# 临时文件目录
TMP_DIR = "tmp"

# 输出参数
OUTPUT_CSV_HEADERS = ['帧数', '时间码', '文本内容', '像素数量', '置信度', '类型']
