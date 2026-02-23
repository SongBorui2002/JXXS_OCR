# JXXS Video Subtitle OCR System

🎬 基于深度学习的智能视频字幕OCR识别系统，专为影视后期制作设计，能够自动识别和提取VFX/DI字幕内容。

## 📁 项目文件说明

| 文件 | 类型 | 说明 |
|------|------|------|
| `main_coordinator.py` | 主模块 | 主协调器，协调整个处理流程 |
| `video_preprocessor.py` | 预处理 | 视频解码、颜色检测、ROI提取 |
| `paddle_ocr_service.py` | OCR服务 | PaddleOCR批量文本识别 |
| `result_processor.py` | 后处理 | 结果过滤、去重、规范化 |
| `config.py` | 配置 | 统一参数配置管理 |
| `videoOCR_Paddle.py` | 历史文件 | 单体架构版本，已废弃 |

---

## 🎯 快速开始

### 环境要求

- Python 3.7+
- OpenCV 4.5+
- PaddlePaddle 2.4+
- PaddleOCR 2.6+

### 安装步骤

```bash
# 克隆项目
cd /Users/sbr/Desktop/JXXS_OCR

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # macOS/Linux

# 安装依赖
pip install -r requirements.txt
```

### 基本使用

```bash
# 基本识别
python main_coordinator.py --video_path your_video.mp4

# 指定时间范围处理
python main_coordinator.py --video_path your_video.mp4 --start_time 00:10:00 --end_time 00:20:00

# 使用LUT增强图像质量
python main_coordinator.py --video_path your_video.mp4 --lut_path JXXS_OCR.cube

# 调试模式（顺序处理）
python main_coordinator.py --video_path your_video.mp4 --sequential
```

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         整体处理流程                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌───────────────┐                                                    │
│   │   视频输入     │                                                    │
│   └───────┬───────┘                                                    │
│           ▼                                                             │
│   ┌───────────────────────────┐                                         │
│   │   VideoPreprocessor       │◀━━━━━━━━━━┐                           │
│   │   • ROI区域提取           │           │                           │
│   │   • HLS颜色检测           │           │                           │
│   │   • LUT图像增强           │           │                           │
│   └───────────┬───────────────┘           │                           │
│               ▼                             │                           │
│   ┌───────────────────────────┐            │                           │
│   │   FrameData 数据结构       │            │                           │
│   │   (包含图像字节流)         │            │                           │
│   └───────────┬───────────────┘            │                           │
│               ▼                             │                           │
│   ┌───────────────────────────┐            │                           │
│   │   MainCoordinator         │            │                           │
│   │   • 批处理分组             │            │                           │
│   │   • 进程池并发调度         │            │                           │
│   └───────────┬───────────────┘            │                           │
│               ▼                             │                           │
│   ┌───────────────────────────┐            │                           │
│   │   PaddleOCR Service       │            │                           │
│   │   • 批量文本识别           │            │                           │
│   │   • 中文字符识别           │            │                           │
│   └───────────┬───────────────┘            │                           │
│               ▼                             │                           │
│   ┌───────────────────────────┐            │                           │
│   │   OCRResult 数据结构       │            │                           │
│   └───────────┬───────────────┘            │                           │
│               ▼                             │                           │
│   ┌───────────────────────────┐            │                           │
│   │   ResultProcessor          │            │                           │
│   │   • 文本规范化              │            │                           │
│   │   • IoU去重                │            │                           │
│   │   • 相似文本合并            │            │                           │
│   └───────────┬───────────────┘            │                           │
│               ▼                             │                           │
│   ┌───────────────────────────┐            │                           │
│   │   CSV 输出文件             │            │                           │
│   │   tmp/ 临时目录             │            │                           │
│   └───────────────────────────┘            │                           │
│                                          │                           │
└──────────────────────────────────────────┼───────────────────────────┘
                                           │
                           ┌───────────────┴───────────────┐
                           ▼                               ▼
┌───────────────────────────────┐         ┌───────────────────────────────┐
│   ProcessPoolExecutor          │         │   单进程顺序处理              │
│   (长视频 >1000帧 自动启用)    │         │   (短视频 或 --sequential)    │
│                               │         │                               │
│   ┌─────┐ ┌─────┐ ┌─────┐    │         │   顺序执行                    │
│   │Worker│ │Worker│ │Worker│...│         │                               │
│   │  1   │ │  2   │ │  3   │            │                               │
│   └─────┘ └─────┘ └─────┘              │                               │
└───────────────────────────────┘         └───────────────────────────────┘
```

---

## 📖 核心类详解

### 1. VideoPreprocessor 类

**职责**: 视频预处理服务，负责视频解码、帧提取、ROI区域提取、HLS颜色检测

#### 1.1 核心数据结构

```python
# 帧数据结构
@dataclass
class FrameData:
    frame_number: int      # 帧号
    timecode: str          # SMPTE时间码 (HH:MM:SS:FF)
    image_bytes: bytes     # PNG编码的图像字节流
    pixel_count: int       # 检测到的颜色像素数量
    text_type: str        # 'VFX' or 'DI'
    image_shape: tuple     # 图像形状 (height, width, channels)

# 视频信息结构
@dataclass  
class VideoInfo:
    fps: float            # 帧率
    frame_count: int      # 总帧数
    width: int            # 视频宽度
    height: int           # 视频高度
    duration_seconds: float  # 时长(秒)
```

#### 1.2 ROI区域定义

```python
# 配置文件 config.py
ROI_TOP_RATIO = 0.06      # 视频高度的6%
ROI_RIGHT_RATIO = 0.40    # 视频宽度的40%
```

**ROI区域位置示意图**:
```
┌─────────────────────────────────────────┐
│                                          │
│    ┌─────────────────────────────────┐    │  ← 顶部6% (roi_top)
│    │                                 │    │
│    │         ROI 区域                │    │  
│    │   (右上角 宽度的40%)           │    │
│    │                                 │    │
│    └─────────────────────────────────┘    │
│                                          │
└─────────────────────────────────────────┘
           ↑
      右边40% (roi_right)
```

**代码实现**:
```python
# video_preprocessor.py:211
def get_colored_pixel_count(self, frame: np.ndarray) -> List[Tuple[str, int, np.ndarray]]:
    """获取ROI区域中目标颜色像素并返回过滤后的ROI"""
    # 提取ROI区域
    roi = frame[0:self.roi_top, self.roi_right:self.video_info.width]
```

#### 1.3 HLS颜色空间检测算法

**为什么选择HLS而不是RGB？**

| 颜色空间 | 优点 | 缺点 |
|---------|------|------|
| **RGB** | 直观 | 受光照影响大，绿/橙难以区分 |
| **HSL** | **色相(Hue)分离度高**，不受亮度影响 | 需要转换计算 |

**HLS颜色空间可视化**:
```
     0°    30°    60°    90°    120°   ...   180°
      │      │      │      │      │           │
      ▼      ▼      ▼      ▼      ▼           ▼
   ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐
   │红色 │橙色 │黄色 │绿色 │青色 │蓝色 │品红 │红色 │
   └─────┴─────┴─────┴─────┴─────┴─────┴─────┘
               ▲                   ▲
               │                   │
          VFX绿色范围            DI橙色范围
          H:45-75              H:10-25
```

**HLS阈值参数**:
```python
# config.py:14-21

# VFX字幕（绿色）- 用于VFX合成信息标识
LOWER_GREEN_HLS = np.array([45, 106, 138])   # H:45, L:106, S:138
UPPER_GREEN_HLS = np.array([75, 195, 255])    # H:75, L:195, S:255

# DI字幕（橙色）- 用于调色信息标识
LOWER_ORANGE_HLS = np.array([10, 106, 75])    # H:10, L:106, S:75
UPPER_ORANGE_HLS = np.array([25, 160, 245])   # H:25, L:160, S:245

# 像素阈值
PIXEL_THRESHOLD = 680      # 超过680像素才触发检测
```

#### 1.4 颜色检测完整流程

```python
# video_preprocessor.py:209-233
def get_colored_pixel_count(self, frame: np.ndarray) -> List[Tuple[str, int, np.ndarray]]:
    """获取ROI区域中目标颜色像素并返回过滤后的ROI"""
    
    # 步骤1: 提取ROI区域
    roi = frame[0:self.roi_top, self.roi_right:self.video_info.width]
    
    # 步骤2: BGR转HLS颜色空间
    # OpenCV的HLS: H∈[0,180], L∈[0,255], S∈[0,255]
    hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
    
    # 步骤3: 颜色阈值分割（inRange生成二值掩码）
    green_mask = cv2.inRange(hls, LOWER_GREEN_HLS, UPPER_GREEN_HLS)
    orange_mask = cv2.inRange(hls, LOWER_ORANGE_HLS, UPPER_ORANGE_HLS)
    
    # 步骤4: 形态学去噪（去除小斑点）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
    
    # 步骤5: 应用掩码提取ROI
    green_only = cv2.bitwise_and(roi, roi, mask=green_mask)
    orange_only = cv2.bitwise_and(roi, roi, mask=orange_mask)
    
    # 步骤6: 像素计数
    green_count = cv2.countNonZero(green_mask)
    orange_count = cv2.countNonZero(orange_mask)
    
    # 步骤7: 返回超过阈值的结果
    results = []
    if green_count > PIXEL_THRESHOLD:
        results.append(("VFX", green_count, green_only))
    if orange_count > PIXEL_THRESHOLD:
        results.append(("DI", orange_count, orange_only))
    
    return results
```

**颜色检测流程图**:
```
输入帧
    │
    ▼
┌─────────────────────┐
│  提取ROI区域         │  ← frame[0:roi_top, roi_right:width]
│  (右上角40%×顶部6%) │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  BGR → HLS 转换     │  ← cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐    ┌─────────────────────────────┐
│  颜色阈值分割        │───▶│ green_mask = inRange(...)   │
│  cv2.inRange()      │    │ H:45-75, L:106-195, S:138-255│
└─────────┬───────────┘    └─────────────────────────────┘
          │
          ▼
┌─────────────────────┐    ┌─────────────────────────────┐
│  形态学去噪         │───▶│ kernel = (3,3)              │
│  MORPH_OPEN         │    │ 去除小斑点噪声               │
└─────────┬───────────┘    └─────────────────────────────┘
          │
          ▼
    ┌─────┴─────┐
    ▼           ▼
  绿色掩码    橙色掩码
    │           │
    ▼           ▼
像素计数    像素计数
    │           │
    └─────┬─────┘
          │
          ▼
    ┌─────┴─────┐
    ▼           ▼
 VFX>680?    DI>680?
    │           │
  ┌─┴─┐       ┌─┴─┐
  │是 │       │是 │
  ▼   ▼       ▼   ▼
返回      返回
(VFX,    (DI,
 count,  count,
 mask)   mask)
```

#### 1.5 采样控制策略

```python
# video_preprocessor.py:235-248
def should_detect_ocr(self, text_type: str, pixel_count: int) -> bool:
    """
    判断是否应该进行OCR检测
    策略: 隔帧检测，每2帧检测1帧
    """
    # 初始化计数器
    if not hasattr(self, '_sample_counters'):
        self._sample_counters = {'VFX': 0, 'DI': 0}

    # 超过像素阈值才考虑检测
    if pixel_count > PIXEL_THRESHOLD:
        # 隔帧逻辑：counter在0和1之间交替
        counter = self._sample_counters[text_type]
        self._sample_counters[text_type] = (counter + 1) % 2
        return counter == 0  # 只有counter=0时检测

    return False
```

**采样时序图**:
```
帧号:    0   1   2   3   4   5   6   7   8   9
─────────────────────────────────────────────
VFX颜色: █   ░   █   ░   █   ░   █   ░   █   ░
         ↑       ↑       ↑       ↑       ↑
        检测    跳过    检测    跳过    检测

counter: 0   1   0   1   0   1   0   1   0   1
检测:   ✅   ❌   ✅   ❌   ✅   ❌   ✅   ❌   ✅

█ = 超过680像素
░ = 超过680像素
```

#### 1.6 LUT图像增强

```python
# video_preprocessor.py:171-207
def apply_lut_processing(self, image_bgr: np.ndarray, lut_path: str) -> np.ndarray:
    """
    应用LUT处理到图像
    目的: 提升OCR识别准确率
    """
    try:
        # 步骤1: BGR → RGB 转换
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 步骤2: 归一化到 [0, 1]
        image_normalized = image_rgb.astype(np.float32) / 255.0

        # 步骤3: 加载LUT文件
        lut_3d = colour.io.read_LUT(lut_path)

        # 步骤4: 应用LUT
        try:
            processed_image = lut_3d.apply(image_normalized)
        except:
            # 备用方法：手动三线性插值
            height, width, channels = image_normalized.shape
            image_reshaped = image_normalized.reshape(-1, channels)
            processed_reshaped = colour.algebra.table_interpolation_trilinear(
                image_reshaped, lut_3d.table
            )
            processed_image = processed_reshaped.reshape(height, width, channels)

        # 步骤5: 裁剪到有效范围
        processed_image = np.clip(processed_image, 0.0, 1.0)

        # 步骤6: 转换回 uint8 → BGR
        processed_uint8 = (processed_image * 255).astype(np.uint8)
        processed_bgr = cv2.cvtColor(processed_uint8, cv2.COLOR_RGB2BGR)

        return processed_bgr

    except Exception as e:
        raise Exception(f"LUT处理失败: {str(e)}")
```

---

### 2. PaddleOCRService 类

**职责**: PaddleOCR批量文本识别服务

#### 2.1 OCRResult 数据结构

```python
# paddle_ocr_service.py:22-33
@dataclass
class OCRResult:
    """OCR结果数据结构"""
    frame_number: int                      # 帧号
    timecode: str                          # SMPTE时间码
    text: str                              # 识别出的文本
    pixel_count: int                       # 颜色像素数量
    confidence: float                       # 置信度 (0.0-1.0)
    text_type: str                         # 'VFX' or 'DI'
    bbox: Tuple[int, int, int, int]       # 边界框 (x1, y1, x2, y2)
    roi_png_path: str                       # ROI图像路径（空=字节流传递）
    raw_ocr_data: Dict[str, Any]          # 原始数据用于调试
```

#### 2.2 单帧OCR处理流程

```python
# paddle_ocr_service.py:57-185
def process_single_frame(self, frame_data: FrameData) -> Optional[OCRResult]:
    """处理单个帧的OCR"""
    try:
        # 步骤1: 从字节流重建图像
        # FrameData中存储的是PNG编码的字节流
        image_array = np.frombuffer(frame_data.image_bytes, dtype=np.uint8)
        roi_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if roi_image is None:
            print(f"图像解码失败: 帧{frame_data.frame_number}")
            return None

        # 步骤2: BGR → RGB 转换
        # PaddleOCR期望RGB格式输入
        roi_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)

        # 步骤3: 调用PaddleOCR
        ocr_result = self.ocr.predict(roi_image)

        if not ocr_result:
            print(f"跳过帧 {frame_data.frame_number}: OCR返回空")
            return None

        # 步骤4: 解析OCR结果
        text_parts = []      # 识别出的文本片段
        confidences = []     # 各片段置信度
        bboxes = []          # 各片段边界框
        raw_data = []        # 原始数据

        for item in ocr_result:
            texts = item.get('rec_texts', [])   # 文本列表
            scores = item.get('rec_scores', []) # 置信度列表
            boxes = item.get('rec_polys', [])   # 多边形坐标列表

            for i, (t, s) in enumerate(zip(texts, scores)):
                if t:
                    text_parts.append(t.strip())
                    confidences.append(float(s))

                    # 提取bbox坐标
                    box = boxes[i]
                    bbox = self._parse_bbox(box)  # 统一格式解析
                    bboxes.append(bbox)

                    raw_data.append({
                        'text': t.strip(),
                        'score': float(s),
                        'bbox': bbox
                    })

        if not text_parts:
            print(f"跳过帧 {frame_data.frame_number}: 未识别到文本")
            return None

        # 步骤5: 合并结果
        full_text = ''.join(filter(None, text_parts))
        avg_confidence = sum(confidences) / len(confidences)

        # 选择最大面积的bbox作为代表（适用于多行文本）
        representative_bbox = max(bboxes, key=lambda b: self._bbox_area(b))

        # 步骤6: 创建OCRResult
        result = OCRResult(
            frame_number=frame_data.frame_number,
            timecode=frame_data.timecode,
            text=full_text,
            pixel_count=frame_data.pixel_count,
            confidence=avg_confidence,
            text_type=frame_data.text_type,
            bbox=representative_bbox,
            raw_ocr_data={'items': raw_data}
        )

        print(f"OCR成功 帧:{frame_data.frame_number} "
              f"类型:{frame_data.text_type} "
              f"像素:{frame_data.pixel_count} "
              f"置信度:{avg_confidence:.2f} "
              f"文本:{full_text}")

        return result

    except Exception as e:
        print(f"OCR错误 在帧 {frame_data.frame_number}: {str(e)}")
        return None
```

#### 2.3 BBox格式解析

```python
# paddle_ocr_service.py:103-136
def _parse_bbox(self, box) -> Tuple[int, int, int, int]:
    """
    解析PaddleOCR返回的bbox格式为标准 (x1, y1, x2, y2)
    
    PaddleOCR可能返回:
    1. numpy array shape=(4, 2): [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    2. numpy array shape=(8,): [x1,y1,x2,y2,x3,y3,x4,y4]
    3. list of lists: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    4. flat list: [x1,y1,x2,y2,x3,y3,x4,y4]
    """
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

        # 计算边界框 (x1, y1, x2, y2)
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        x1, y1 = int(x_coords.min()), int(y_coords.min())
        x2, y2 = int(x_coords.max()), int(y_coords.max())
        
        return (x1, y1, x2, y2)
        
    except Exception as e:
        return (0, 0, 0, 0)
```

#### 2.4 批量OCR处理

```python
# paddle_ocr_service.py:187-197
def process_batch(self, frame_batch: List[FrameData]) -> List[OCRResult]:
    """批量处理OCR"""
    results = []

    for frame_data in frame_batch:
        result = self.process_single_frame(frame_data)
        if result:
            results.append(result)

    print(f"批处理完成: 处理 {len(frame_batch)} 帧，成功识别 {len(results)} 帧")
    return results
```

---

### 3. ResultProcessor 类

**职责**: 结果处理服务，负责过滤、去重、规范化

#### 3.1 字符串规范化算法

```python
# result_processor.py:23-84
def process_text(self, text: str, text_type: str) -> str:
    """
    规范化处理OCR识别结果
    
    处理OCR的各种误识别:
    - 字符混淆: V↔X, D↔I↔1↔l↔|, O↔0
    - 符号混淆: :↔;↔.
    - 漏字/多字: VVFX, 漏V
    """
    if not text:
        return ""

    if text_type == "VFX":
        # 步骤1: 直接替换各种混淆变体
        replacements = [
            ('VEX:', 'VFX:'),  # E↔F 混淆
            ('VFX;', 'VFX:'),  # ;↔: 混淆
            ('VEX;', 'VFX:'),
            ('VFX.', 'VFX:'),  # .↔: 混淆
            ('VEX.', 'VFX:'),
            ('FX:', 'VFX:'),   # 漏V
            ('VVFX:', 'VFX:'), # 多V
        ]
        for old, new in replacements:
            text = text.replace(old, new)

        # 步骤2: 处理不以VFX:开头的情况
        if not text.startswith('VFX:') and text.startswith('VFX'):
            # 形如 "VFX内容" → "VFX:内容"
            text = 'VFX:' + text[3:]
        
        elif not text.startswith('VFX:'):
            # 检测各种变体
            variants = ['vfx', 'vex', 'vpx', 'vix']
            if text[:4].lower().replace(' ', '') in variants:
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

    else:  # DI 类型
        # DI变体更多，因为 I↔1↔l↔|↔O↔0 都有混淆
        replacements = [
            ('D1:', 'DI:'), ('D1;', 'DI:'),  # 1↔I
            ('Di', 'DI:'), ('Di;', 'DI:'),  # i↔I (大小写)
            ('Di:', 'DI:'),
            ('Dl:', 'DI:'), ('D|:', 'DI:'), ('DL:', 'DI:'),  # l↔|, L↔I
            ('01:', 'DI:'), ('01;', 'DI:'),  # 0↔O
            ('DI;', 'DI:'), ('D1;', 'DI:'), ('Dl;', 'DI:'),
            ('D|;', 'DI:'), ('DL;', 'DI:'),
        ]
        for old, new in replacements:
            text = text.replace(old, new)

        if not text.startswith('DI:') and text.startswith('DI'):
            text = 'DI:' + text[3:]
        elif not text.startswith('DI:'):
            variants = ['di', 'd1', 'dl', 'ol', 'oi', '01']
            if text[:3].lower().replace(' ', '') in variants:
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
```

**OCR误识别示例对照表**:

| 原始字幕 | OCR识别 | 规范化后 |
|---------|--------|---------|
| `VFX:Comp A` | `VEX:Comp A` | `VFX:Comp A` |
| `VFX:Matte` | `VFX;Matte` | `VFX:Matte` |
| `VFX:Light` | `VVFX:Light` | `VFX:Light` |
| `VFX:Render` | `FX:Render` | `VFX:Render` |
| `DI:Grade` | `D1:Grade` | `DI:Grade` |
| `DI:Cube` | `Dl:Cube` | `DI:Cube` |
| `DI:LUT` | `01:LUT` | `DI:LUT` |

#### 3.2 文本相似度计算

```python
# result_processor.py:359-374
def _text_similarity(self, text1: str, text2: str) -> float:
    """
    计算两个文本的相似度
    方法: Jaccard相似系数 = |A∩B| / |A∪B|
    
    示例:
    text1 = "VFX:Comp A" → {'v', 'f', 'x', ':', 'c', 'o', 'm', 'p', ' ', 'a'}
    text2 = "VFX:Comp B" → {'v', 'f', 'x', ':', 'c', 'o', 'm', 'p', ' ', 'b'}
    
    intersection = 9  (v,f,x,:,c,o,m,p, )
    union = 10         (v,f,x,:,c,o,m,p, ,a,b)
    similarity = 0.9
    """
    if not text1 or not text2:
        return 0.0

    # 转换为小写，去重
    set1 = set(text1.lower())
    set2 = set(text2.lower())

    if not set1 or not set2:
        return 0.0

    intersection = set1.intersection(set2)
    union = set1.union(set2)

    return len(intersection) / len(union)
```

#### 3.3 IoU（交并比）计算

```python
# result_processor.py:312-341
def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                   bbox2: Tuple[int, int, int, int]) -> float:
    """
    计算两个边界框的IoU（Intersection over Union）
    
    IoU = 交集面积 / 并集面积
    
    范围: 0.0 (完全不重叠) ~ 1.0 (完全重合)
    
    示例:
    bbox1 = (10, 10, 50, 50)  # 面积 = 40×40 = 1600
    bbox2 = (30, 30, 70, 70)  # 面积 = 40×40 = 1600
    
    交集 = (30-50) × (30-50) = 20×20 = 400
    并集 = 1600 + 1600 - 400 = 2800
    IoU = 400 / 2800 ≈ 0.143
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # 计算交集区域
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y2_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    # 交集面积
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # 各框面积
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # 并集面积
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0
```

**IoU可视化**:

```
    ┌───────────────────────┐
    │        bbox1           │
    │   ┌───────────────┐   │
    │   │   ┌─────────┐  │   │
    │   │   │  交集   │  │   │
    │   │   └─────────┘  │   │
    │   └───────────────┘   │
    │        bbox2           │
    └───────────────────────┘
    
    IoU = 交集面积 / 并集面积
    
    IoU = 0.0  → 完全不重叠
    IoU = 0.5  → 重叠50%
    IoU = 1.0  → 完全重合
```

#### 3.4 基于连续帧和IoU的去重

```python
# result_processor.py:141-231
def deduplicate_by_continuous_frames_iou(
    self, 
    ocr_results: List[OCRResult], 
    max_frame_gap: int = 12,      # 最大帧间隔
    iou_threshold: float = 0.8    # IoU阈值
) -> List[OCRResult]:
    """
    基于连续帧和IoU的去重处理
    
    策略:
    1. 按帧号排序
    2. 查找连续帧组（帧号差距≤max_frame_gap）
    3. 组内判断：IoU≥阈值 或 文本相似度≥0.8 → 合并
    4. 只保留≥10帧的连续组
    5. 从组内选择置信度最高的结果
    
    示例:
    帧序列: [100, 101, 102, ..., 118] 共19帧
            │
            ├── 帧间隔 ≤12 ✓
            ├── IoU ≥ 0.8 ✓ 或 文本相似度 ≥ 0.8 ✓
            └── 帧数 ≥ 10 ✓
            │
            └── 合并为1个结果（保留第一帧100）
    """
    if not ocr_results:
        return ocr_results

    # 步骤1: 按帧号排序
    sorted_results = sorted(ocr_results, key=lambda x: x.frame_number)

    deduplicated = []
    i = 0

    while i < len(sorted_results):
        current_result = sorted_results[i]
        continuous_group = [current_result]
        j = i + 1

        # 步骤2: 查找连续帧组
        while j < len(sorted_results):
            next_result = sorted_results[j]

            # 检查帧号连续性和类型
            frame_gap = next_result.frame_number - continuous_group[-1].frame_number
            type_match = next_result.text_type == current_result.text_type

            if frame_gap <= max_frame_gap and type_match:
                # 计算IoU
                current_bbox = self._bbox_from_paddle_points(current_result.bbox)
                next_bbox = self._bbox_from_paddle_points(next_result.bbox)
                iou = self._calculate_iou(current_bbox, next_bbox)

                # 计算文本相似度
                text_similarity = self._text_similarity(
                    continuous_group[-1].text, next_result.text
                )

                # 合并条件：IoU达标 或 文本相似度高
                merge_condition = (iou >= iou_threshold) or (text_similarity >= 0.8)

                if merge_condition:
                    continuous_group.append(next_result)
                    j += 1
                else:
                    break  # 不满足条件，结束当前组
            else:
                break  # 帧号不连续或类型不同

        # 步骤3: 处理连续组
        if len(continuous_group) >= 10:  # 只保留长连续组
            # 选择最佳结果
            best_result = self._select_best_from_continuous_group(continuous_group)
            # 保持第一帧的时间码
            best_result.frame_number = continuous_group[0].frame_number
            best_result.timecode = continuous_group[0].timecode
            deduplicated.append(best_result)
            print(f"连续帧组去重: {len(continuous_group)} 帧 -> 1 帧")
        elif len(continuous_group) > 1:
            print(f"跳过短连续组: {len(continuous_group)} 帧 - 长度不足10帧")
        else:
            print(f"删除单帧结果: 帧 {continuous_group[0].frame_number}")

        i = j  # 移动到下一组

    return deduplicated
```

#### 3.5 完整的后处理流程

```python
# result_processor.py:376-392
def process_results(self, ocr_results: List[OCRResult]) -> List[OCRResult]:
    """
    完整的后处理流程
    
    处理步骤:
    1. 过滤低质量结果 (置信度 < 0.1)
    2. 连续帧IoU去重 (max_gap=12, iou≥0.8)
    3. 相似文本合并 (相似度>0.8, 同类型, ≤1秒)
    """
    print(f"开始后处理 {len(ocr_results)} 个OCR结果")

    # 步骤1: 过滤低质量结果
    filtered = self.filter_results(ocr_results, min_confidence=0.1)
    print(f"过滤后: {len(filtered)} 个结果")

    # 步骤2: 连续帧IoU去重
    continuous_deduplicated = self.deduplicate_by_continuous_frames_iou(
        filtered, max_frame_gap=12, iou_threshold=0.8
    )
    print(f"连续帧去重后: {len(continuous_deduplicated)} 个结果")

    # 步骤3: 相似文本合并
    final_results = self.merge_similar_texts(continuous_deduplicated)

    print(f"后处理完成: 最终 {len(final_results)} 个结果")
    return final_results
```

---

### 4. MainCoordinator 类

**职责**: 主进程协调器，协调整个视频处理流程

#### 4.1 批处理与并发设计

```python
# main_coordinator.py:189-228
def _concurrent_batch_ocr(self, ocr_tasks: List[FrameData]) -> List[OCRResult]:
    """
    并发处理OCR批次
    
    并发策略:
    1. 将OCR任务分批 (BATCH_SIZE=20)
    2. 使用进程池 (ProcessPoolExecutor)
    3. 最多3个并发Worker (MAX_WORKERS=3)
    
    为什么用进程池而非线程池？
    - PaddleOCR是CPU密集型计算
    - Python GIL限制多线程效率
    - 进程池可突破GIL限制
    """
    if not ocr_tasks:
        return []

    # 步骤1: 分批处理
    ocr_batches = []
    for i in range(0, len(ocr_tasks), BATCH_SIZE):
        batch = ocr_tasks[i:i + BATCH_SIZE]
        ocr_batches.append(batch)

    print(f"OCR任务分批: {len(ocr_tasks)} 个任务 → {len(ocr_batches)} 个批次")

    # 步骤2: 进程池并发处理
    all_ocr_results = []
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(ocr_batches))) as executor:
        future_to_batch = {}
        for batch in ocr_batches:
            # 提交批次任务
            future = executor.submit(process_ocr_batch_parallel, batch)
            future_to_batch[future] = batch

        # 收集结果
        completed_batches = 0
        for future in future_to_batch:
            try:
                batch_results = future.result()
                all_ocr_results.extend(batch_results)

                completed_batches += 1
                progress = (completed_batches / len(ocr_batches)) * 100
                print(f"\rOCR进度: {progress:.2f}%", end="", flush=True)

            except Exception as e:
                print(f"OCR批次处理失败: {e}")

    print(f"\rOCR进度: 100.00%")
    return all_ocr_results
```

**分批示例**:
```
假设有 67 个OCR任务，BATCH_SIZE = 20

原始任务: [0, 1, 2, 3, 4, 5, 6, ... , 66]  共67个
                │
                ▼ 分批
               
批次1: [0, 1, 2, 3, ... , 19]  ← 20个任务 → Worker 1
批次2: [20, 21, 22, ... , 39]  ← 20个任务 → Worker 2 (并行)
批次3: [40, 41, 42, ... , 59]  ← 20个任务 → Worker 3 (并行)
批次4: [60, 61, 62, ... , 66]  ← 7个任务  → Worker 1 (空闲后处理)

共 4 个批次，3个Worker
```

**进程池 vs 线程池**:

```
┌─────────────────────────────────────────────────────────────┐
│                    ProcessPoolExecutor                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │Worker 1 │  │Worker 2 │  │Worker 3 │  │Worker...│       │
│  │ PID:xxx │  │ PID:yyy │  │ PID:zzz │  │          │       │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       │
│       │             │             │             │              │
│       └─────────────┴─────────────┴─────────────┘              │
│                     独立内存空间                                 │
│            GIL锁不阻塞，计算密集型任务高效                       │
└─────────────────────────────────────────────────────────────┘

vs

┌─────────────────────────────────────────────────────────────┐
│                    ThreadPoolExecutor                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │ Thread1 │  │ Thread2 │  │ Thread3 │  │ Thread...│       │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       │
│       │             │             │             │              │
│       └─────────────┴─────────────┴─────────────┘              │
│                     共享内存空间                                │
│        GIL锁限制，Python中多线程效率低（IO密集型除外）          │
└─────────────────────────────────────────────────────────────┘
```

#### 4.2 进程隔离与模块级函数

```python
# main_coordinator.py:17-32
def process_ocr_batch_parallel(frame_data_batch: List[FrameData]) -> List[OCRResult]:
    """
    在子进程中处理单个OCR批次
    
    必须是模块级函数，因为:
    1. ProcessPoolExecutor 使用 pickle 序列化
    2. 类方法无法被 pickle（隐含self参数）
    3. 每个子进程独立创建OCR服务实例
    """
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
```

**为什么必须是模块级函数？**:

```python
# ❌ 错误写法
class MyClass:
    def inner_method(self, data):  # 无法被pickle序列化
        pass

# ✅ 正确写法
def process_ocr_batch_parallel(data):  # 可以被pickle序列化
    pass
```

#### 4.3 完整处理流程

```python
# main_coordinator.py:53-88
def run(self, parallel: bool = True) -> str:
    """
    运行完整的处理流程
    
    处理模式选择:
    - 长视频 (>1000帧): 自动使用并行模式
    - 短视频 (≤1000帧): 使用顺序模式
    - 强制顺序: --sequential 参数
    """
    start_time = time.time()

    try:
        # 选择处理模式
        if parallel and self.preprocessor.total_frames_to_process > 1000:
            print("检测到长视频，使用并行处理模式")
            results = self.process_video_parallel()
        else:
            print("使用顺序处理模式")
            results = self.process_video_sequential()

        # 后处理（过滤、去重、规范化）
        filtered_results = self.result_processor.process_results(results)

        # 保存结果
        output_file = self.result_processor.save_to_csv(filtered_results)

        # 统计信息
        stats = self.result_processor.get_statistics(results)
        elapsed_time = time.time() - start_time

        print("\n=== 处理完成 ===")
        print(f"总耗时: {elapsed_time:.2f} 秒")
        print(f"检测到字幕: {stats['total_results']} 个")
        print(f"VFX字幕: {stats['vfx_count']} 个")
        print(f"DI字幕: {stats['di_count']} 个")
        print(f"帧范围: {stats['frame_range']}")
        print(f"结果文件: {output_file}")

        return output_file

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        raise
```

**处理流程图**:
```
┌─────────────────────────────────────────────────────────────────┐
│                     MainCoordinator.run()                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  模式选择                                                         │
│      │                                                          │
│      ▼                                                          │
│  ┌───────────────────────────┐                                  │
│  │ total_frames > 1000 ?     │                                  │
│  └───────────┬───────────────┘                                  │
│      ┌──────┴──────┐                                           │
│      ▼             ▼                                           │
│   是            否                                              │
│      │             │                                           │
│      ▼             ▼                                           │
│  ┌────────┐   ┌────────────┐                                    │
│  │ 并行   │   │ 顺序处理   │                                    │
│  │ 模式   │   │  模式      │                                    │
│  └───┬────┘   └────┬─────┘                                    │
│      │             │                                           │
│      ▼             ▼                                           │
│  ┌───────────────────────────┐                                  │
│  │ _sequential_preprocess_frames()│                              │
│  │ • 顺序读取视频帧             │                              │
│  │ • 颜色检测                   │                              │
│  │ • 生成 FrameData 列表        │                              │
│  └───────────┬───────────────┘                                  │
│              │                                                  │
│              ▼                                                  │
│  ┌───────────────────────────┐                                  │
│  │ _concurrent_batch_ocr()   │                                  │
│  │ • 分批 (BATCH_SIZE=20)    │                              │
│  │ • 进程池并发 (3 workers)   │                              │
│  │ • 返回 OCRResult 列表      │                              │
│  └───────────┬───────────────┘                                  │
│              │                                                  │
│              ▼                                                  │
│  ┌───────────────────────────┐                                  │
│  │ result_processor.process_results()│                           │
│  │ • 过滤 (置信度<0.1)        │                              │
│  │ • IoU去重 (max_gap=12)    │                              │
│  │ • 相似文本合并             │                              │
│  └───────────┬───────────────┘                                  │
│              │                                                  │
│              ▼                                                  │
│  ┌───────────────────────────┐                                  │
│  │ save_to_csv()             │                              │
│  │ 输出 CSV 结果文件          │                              │
│  └───────────────────────────┘                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ 配置参数完整参考

```python
# config.py

# ==================== 视频处理参数 ====================
DEFAULT_FPS = 25.0                    # 默认帧率 (当无法读取时使用)

# ==================== ROI 区域参数 ====================
ROI_TOP_RATIO = 0.06                  # ROI高度占视频高度的比例 (6%)
ROI_RIGHT_RATIO = 0.40                # ROI宽度占视频宽度的比例 (40%)

# ==================== HLS 颜色阈值参数 ====================
# OpenCV HLS: H∈[0,180], L∈[0,255], S∈[0,255]

# VFX字幕（绿色）- H:45-75, L:106-195, S:138-255
LOWER_GREEN_HLS = np.array([45, 106, 138])
UPPER_GREEN_HLS = np.array([75, 195, 255])

# DI字幕（橙色）- H:10-25, L:106-160, S:75-245
LOWER_ORANGE_HLS = np.array([10, 106, 75])
UPPER_ORANGE_HLS = np.array([25, 160, 245])

# ==================== 检测参数 ====================
PIXEL_THRESHOLD = 680                  # 像素阈值（超过此值才触发检测）
FRAME_WINDOW = 5                       # 滑动窗口大小
INCREASE_THRESHOLD = 2.0              # 像素增长阈值

# ==================== LUT 文件参数 ====================
DEFAULT_LUT_PATH = "/Users/sbr/Desktop/JXXS_OCR/JXXS_OCR.cube"
                                        # 默认LUT文件路径

# ==================== PaddleOCR 参数 ====================
OCR_LANG = 'ch'                       # OCR语言 ('ch'=中文, 'en'=英文)
OCR_USE_TEXTLINE_ORIENTATION = False  # 是否使用文本行方向检测
OCR_USE_DOC_UNWARPER = False          # 是否使用文档展平

# ==================== 批处理参数 ====================
BATCH_SIZE = 20                       # OCR批处理大小（每批处理帧数）
MAX_WORKERS = 3                       # 最大并发进程数

# ==================== 时间参数 ====================
MIN_DETECTION_INTERVAL = 25            # 最短检测间隔（帧）
MAX_DETECTION_INTERVAL = 250           # 最长检测间隔（10秒×25fps）

# ==================== 临时文件参数 ====================
TMP_DIR = "tmp"                       # 临时文件目录

# ==================== 输出参数 ====================
OUTPUT_CSV_HEADERS = [
    '帧数',        # frame_number
    '时间码',      # timecode
    '文本内容',    # text
    '像素数量',    # pixel_count
    '置信度',      # confidence
    '类型'         # text_type (VFX/DI)
]
```

---

## 📋 输出格式

### CSV 结果文件

```csv
帧数,时间码,文本内容,像素数量,置信度,类型
100,00:00:04:00,VFX:Comp A,800,0.950,VFX
200,00:00:08:00,DI:Grade,750,0.880,DI
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| 帧数 | int | 帧号 |
| 时间码 | str | SMPTE时间码 (HH:MM:SS:FF) |
| 文本内容 | str | OCR识别的文本内容 |
| 像素数量 | int | 检测到的颜色像素数量 |
| 置信度 | float | OCR置信度 (0.0-1.0) |
| 类型 | str | VFX 或 DI |

### 临时文件

- `tmp/roi_{帧数}_{类型}.png` - 保存检测到的ROI图像，便于调试验证

---

## 📈 性能表现

### 并发优化成果

| 优化策略 | 效果 |
|---------|------|
| 批处理优化 | 20张/批，吞吐量 4.71张/秒 |
| 并发提升 | 3实例并发，吞吐量 7.36张/秒 (**+56%**) |
| 时间效率 | 41分钟视频约2小时完成 |

### 智能调度

- **短视频**：自动使用顺序处理模式（≤1000帧）
- **长视频**：自动切换并行模式（>1000帧）
- **内存优化**：分批处理，避免内存溢出

---

## 🔧 高级用法

### 命令行参数

| 参数 | 简写 | 说明 | 示例 |
|------|------|------|------|
| `--video_path` | `-v` | 视频文件路径 (必需) | `--video_path video.mp4` |
| `--lut_path` | `-l` | LUT文件路径 | `--lut_path enhance.cube` |
| `--start_time` | `-s` | 开始时间 | `--start_time 00:10:00` |
| `--end_time` | `-e` | 结束时间 | `--end_time 00:20:00` |
| `--sequential` | - | 强制顺序处理 | `--sequential` |

### 时间格式支持

```bash
# 各种时间格式都支持
--start_time 00:10:00        # HH:MM:SS
--start_time 10:00           # MM:SS
--start_time 600             # SS (秒)
--start_time 00:10:00:15     # HH:MM:SS:FF (包含帧号)
```

---

## 🐛 故障排除

### 常见问题

**PaddleOCR安装失败**
```bash
# 手动安装
pip install paddlepaddle paddleocr --upgrade
```

**视频文件无法打开**
- 检查文件路径是否正确
- 确认视频格式支持（MP4, MOV, AVI等）
- 检查文件权限

**LUT文件不存在**
```bash
# 检查LUT文件路径
ls -la JXXS_OCR.cube

# 或跳过LUT处理
python main_coordinator.py --video_path video.mp4
```

**内存不足错误**
```python
# 在config.py中调整
BATCH_SIZE = 10      # 减小批处理大小
MAX_WORKERS = 2      # 减少并发实例
```

### 调试技巧

1. **顺序模式调试**：使用 `--sequential` 查看详细处理过程
2. **临时文件检查**：查看 `tmp/` 目录验证ROI检测结果
3. **日志分析**：观察控制台输出定位问题

---

## 🔄 更新日志

### v2.0.0 (2025-01-02) - 并发性能优化

- ✅ **架构重构**：分离预处理和OCR处理职责
- ✅ **并发优化**：3实例并发，性能提升56%
- ✅ **批处理调优**：20张/批，最优吞吐量4.71张/秒
- ✅ **LUT迁移**：预处理阶段完成图像增强

### v1.0.0 - 微服务架构重构

- ✅ **模块化设计**：拆分为5个独立服务模块
- ✅ **智能调度**：自动选择处理模式
- ✅ **结果优化**：规范化、去重、质量过滤
- ✅ **配置统一**：集中参数管理

---

## 📄 许可证

本项目仅供学习和研究使用，请遵守相关法律法规。

---

**技术栈**: Python • OpenCV • PaddleOCR • NumPy • Colour Science
