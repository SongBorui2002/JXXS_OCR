# 重构版视频字幕OCR系统

基于原有`videoOCR_Paddle.py`的重构版本，将单体架构拆分为微服务架构，支持并发处理41分钟长视频。

## 架构设计

### 核心服务模块

1. **`config.py`** - 统一配置管理
   - 所有参数和阈值定义
   - 颜色范围、检测参数等

2. **`video_preprocessor.py`** - 视频预处理服务
   - 视频解码和帧提取
   - 颜色检测和智能触发
   - LUT图像增强处理
   - 输出需要OCR的帧数据

3. **`paddle_ocr_service.py`** - PaddleOCR服务
   - 批量OCR文本识别
   - 并行处理支持

4. **`result_processor.py`** - 结果处理服务
   - 文本规范化
   - 结果过滤、去重、合并
   - CSV输出

5. **`main_coordinator.py`** - 主进程协调器
   - 协调各服务间的通信
   - 支持顺序/并行处理模式
   - 进度监控和错误处理

## 安装和使用

### 1. 创建虚拟环境

```bash
cd /Users/sbr/Desktop/JXXS_OCR
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或在Windows上: venv\Scripts\activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行系统

```bash
# 基本用法
python main_coordinator.py --video_path your_video.mp4

# 指定时间范围
python main_coordinator.py --video_path your_video.mp4 --start_time 00:10:00 --end_time 00:20:00

# 强制顺序处理（调试用）
python main_coordinator.py --video_path your_video.mp4 --sequential

# 使用LUT文件增强图像
python main_coordinator.py --video_path your_video.mp4 --lut_path /path/to/lut.cube
```

### 4. 测试架构

```bash
python test_architecture.py
```

## 技术特性

### 并发处理
- **自动模式选择**：短视频使用顺序处理，长视频自动切换并行模式
- **多进程架构**：预处理、OCR、结果处理各司其职
- **批处理优化**：减少进程间通信开销

### 智能检测
- **颜色触发**：基于HLS颜色空间的像素阈值检测
- **历史分析**：5帧滑动窗口 + 多条件智能判断
- **防抖动**：避免频繁重复检测

### 图像增强
- **预处理LUT**：在视频预处理阶段应用LUT增强，提高OCR准确性
- **形态学处理**：减少颜色斑点干扰

### 结果优化
- **文本规范化**：统一VFX/DI字幕格式
- **去重合并**：消除时间相近的重复结果
- **质量过滤**：基于置信度阈值过滤

## 输出格式

### CSV文件格式
```csv
帧数,时间码,文本内容,像素数量,置信度,类型
100,00:00:04:00,VFX:测试字幕,800,0.950,VFX
200,00:00:08:00,DI:测试字幕,750,0.880,DI
```

### 中间文件
- `tmp/`目录：保存检测到的ROI图像，便于调试和验证

## 配置参数

主要参数位于`config.py`：

- **视频参数**：`DEFAULT_FPS`, `ROI_TOP_RATIO`, `ROI_RIGHT_RATIO`
- **颜色检测**：`LOWER_GREEN_HLS`, `UPPER_GREEN_HLS` 等
- **处理参数**：`PIXEL_THRESHOLD`, `BATCH_SIZE`, `MAX_WORKERS`
- **时间控制**：`MIN_DETECTION_INTERVAL`, `MAX_DETECTION_INTERVAL`

## 性能优化

### 长视频处理
- **分批处理**：将视频分成小批次，避免内存溢出
- **并行OCR**：多个PaddleOCR实例并发处理
- **智能调度**：根据视频长度自动选择处理策略

### 内存管理
- **帧复用**：及时释放不需要的帧数据
- **批处理传输**：减少进程间数据拷贝

## 故障排除

### 常见问题

1. **PaddleOCR未安装**
   ```
   pip install paddlepaddle paddleocr
   ```

2. **视频文件无法打开**
   - 检查文件路径是否正确
   - 确认视频格式支持

3. **LUT文件不存在**
   - 检查LUT文件路径
   - 或者移除`--lut_path`参数

4. **内存不足**
   - 减小`BATCH_SIZE`
   - 使用顺序处理模式

### 调试模式

```bash
# 查看详细处理过程
python main_coordinator.py --video_path video.mp4 --sequential
```

## 架构优势

相比原单体架构，新架构具有：

1. **模块化**：各服务职责单一，便于维护和扩展
2. **并发性**：充分利用多核CPU，提升处理速度
3. **可扩展性**：可独立升级各服务组件
4. **容错性**：单个服务失败不影响整体流程
5. **测试性**：各模块可独立测试
6. **预处理优化**：LUT增强在预处理阶段完成，提高整体效率

## 最新更新

### 并发性能优化 (2025-01-02)

**测试结果**：
- ✅ **批大小优化**：20张图片每批，吞吐量4.71张/秒
- ✅ **并发测试**：3个PaddleOCR实例并发，吞吐量提升至7.36张/秒 (+56%)
- ✅ **性能预期**：41分钟视频约2小时完成 (之前预计4小时)

**架构重构**：
- ✅ **职责分离**：paddle_ocr_service.py专注OCR处理，移除并发逻辑
- ✅ **并发统一**：main_coordinator.py统一管理所有并发策略
- ✅ **两阶段处理**：预处理阶段 → OCR并发处理阶段

**配置更新**：
```python
BATCH_SIZE = 20    # 批处理大小
MAX_WORKERS = 3    # 并发实例数量
```

### LUT服务重构 (2025-01-02)

**变更说明**：
- ✅ **LUT处理迁移**：将LUT图像增强从`paddle_ocr_service.py`移动到`video_preprocessor.py`
- ✅ **预处理优化**：在检测到需要OCR的帧时立即应用LUT处理
- ✅ **职责分离**：预处理器负责图像增强，OCR服务专注文本识别
- ✅ **性能提升**：减少重复的图像处理开销

## 兼容性

- **Python**: 3.7+
- **OpenCV**: 4.5+
- **PaddlePaddle**: 2.4+
- **PaddleOCR**: 2.6+
- **操作系统**: macOS, Linux, Windows
# JXXS_OCR
