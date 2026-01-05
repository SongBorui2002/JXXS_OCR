# JXXS Video Subtitle OCR System

🎬 基于深度学习的智能视频字幕OCR识别系统，专为影视后期制作设计，能够自动识别和提取VFX/DI字幕内容。

## 📁 项目文件说明

- **`main_coordinator.py`** - 主协调器（当前使用版本）
- **`video_preprocessor.py`** - 视频预处理服务
- **`paddle_ocr_service.py`** - OCR服务
- **`result_processor.py`** - 结果处理服务
- **`config.py`** - 统一配置管理
- **`videoOCR_Paddle.py`** - 🔶 **历史文件**（单体架构版本，已废弃）

## ✨ 核心特性

- **🎯 智能识别**：基于HLS颜色空间的像素阈值检测，自动识别绿色(VFX)和橙色(DI)字幕
- **⚡ 高性能并发**：多进程架构，支持41分钟长视频并发处理，处理速度提升56%
- **🔍 精确检测**：5帧滑动窗口智能判断，避免频繁重复检测
- **🎨 图像增强**：支持LUT文件预处理，提升OCR准确性
- **📊 智能优化**：文本规范化、去重合并、质量过滤
- **🛠️ 模块化设计**：微服务架构，便于维护和扩展

## 🚀 快速开始

### 环境要求

- Python 3.7+
- OpenCV 4.5+
- PaddlePaddle 2.4+
- PaddleOCR 2.6+

### 安装步骤

1. **克隆项目**
   ```bash
   cd /Users/sbr/Desktop/JXXS_OCR
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   # 或在Windows上: venv\Scripts\activate
   ```

3. **安装依赖**
   ```bash
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

## 🏗️ 系统架构

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  MainCoordinator │───▶│ VideoPreprocessor│───▶│ PaddleOCR Service│
│   (协调调度)     │    │   (智能检测)     │    │   (文本识别)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ ResultProcessor │◀───│   Color Detection│◀───│   Batch OCR     │
│   (结果优化)     │    │   (HLS阈值)     │    │   (并发处理)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 核心模块

| 模块 | 职责 | 特性 |
|------|------|------|
| **main_coordinator.py** | 主进程协调器 | 自动选择处理模式，进度监控 |
| **video_preprocessor.py** | 视频预处理 | 智能颜色检测，LUT增强 |
| **paddle_ocr_service.py** | OCR服务 | 批量文本识别，并发优化 |
| **result_processor.py** | 结果处理 | 文本规范化，去重合并 |
| **config.py** | 配置管理 | 统一参数配置 |

## 📋 输出格式

### CSV结果文件
```csv
帧数,时间码,文本内容,像素数量,置信度,类型
100,00:00:04:00,VFX:测试字幕,800,0.950,VFX
200,00:00:08:00,DI:测试字幕,750,0.880,DI
```

### 临时文件
- `tmp/` - 保存检测到的ROI图像，便于调试验证

## ⚙️ 配置参数

### 颜色检测参数
```python
# VFX字幕 (绿色)
LOWER_GREEN_HLS = [45, 106, 138]
UPPER_GREEN_HLS = [75, 195, 255]

# DI字幕 (橙色)
LOWER_ORANGE_HLS = [10, 106, 75]
UPPER_ORANGE_HLS = [25, 160, 245]
```

### 处理参数
```python
PIXEL_THRESHOLD = 680      # 像素阈值
BATCH_SIZE = 20           # OCR批处理大小
MAX_WORKERS = 3           # 并发实例数量
FRAME_WINDOW = 5          # 滑动窗口大小
```

## 📈 性能表现

### 并发优化成果
- **批处理优化**：20张/批，吞吐量4.71张/秒
- **并发提升**：3实例并发，吞吐量7.36张/秒 (**+56%**)
- **时间效率**：41分钟视频约2小时完成

### 智能调度
- **短视频**：自动使用顺序处理模式
- **长视频**：自动切换并行模式 (>1000帧)
- **内存优化**：分批处理，避免内存溢出

## 🔧 高级用法

### 命令行参数

| 参数 | 简写 | 说明 | 示例 |
|------|------|------|------|
| `--video_path` | `-v` | 视频文件路径 (必需) | `--video_path video.mp4` |
| `--lut_path` | `-l` | LUT文件路径 | `--lut_path enhance.cube` |
| `--start_time` | `-s` | 开始时间 | `--start_time 00:10:00` |
| `--end_time` | `-e` | 结束时间 | `--end_time 00:20:00` |
| `--sequential` | - | 强制顺序处理 | `--sequential` |

### 自定义配置

编辑 `config.py` 调整参数：

```python
# 调整检测灵敏度
PIXEL_THRESHOLD = 500      # 降低阈值提高灵敏度
INCREASE_THRESHOLD = 1.5   # 调整增长检测阈值

# 优化性能
BATCH_SIZE = 30           # 增大批处理大小
MAX_WORKERS = 4           # 增加并发实例
```

## 🐛 故障排除

### 常见问题

**❌ PaddleOCR安装失败**
```bash
# 手动安装
pip install paddlepaddle paddleocr --upgrade
```

**❌ 视频文件无法打开**
- 检查文件路径是否正确
- 确认视频格式支持（MP4, MOV, AVI等）
- 检查文件权限

**❌ LUT文件不存在**
```bash
# 检查LUT文件路径
ls -la JXXS_OCR.cube

# 或跳过LUT处理
python main_coordinator.py --video_path video.mp4
```

**❌ 内存不足错误**
```python
# 在config.py中调整
BATCH_SIZE = 10      # 减小批处理大小
MAX_WORKERS = 2      # 减少并发实例
```

### 调试技巧

1. **顺序模式调试**：使用 `--sequential` 查看详细处理过程
2. **临时文件检查**：查看 `tmp/` 目录验证ROI检测结果
3. **日志分析**：观察控制台输出定位问题

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

## 📄 许可证

本项目仅供学习和研究使用，请遵守相关法律法规。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

---

**技术栈**: Python • OpenCV • PaddleOCR • NumPy • Colour Science
# JXXS_OCR
