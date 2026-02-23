# Resolve_Marker

DaVinci Resolve 标记点管理工具，用于与 Resolve 时间线交互，管理 Marker（标记点）。

## 项目结构

```
Resolve_Marker/
├── resolveConnector.py     # Resolve 连接器
├── markerManager.py         # 标记点管理器（核心）
├── test_marker_manager.py   # 功能测试
├── test_add_markers.py      # 批量添加测试
└── test_markers.json        # 标记点数据示例
```

## 核心类

### 1. ResolveConnector

Resolve 连接器，负责与 DaVinci Resolve 建立连接。

**主要方法：**

| 方法 | 说明 |
|------|------|
| `__init__()` | 初始化并连接到 Resolve |
| `get_project_name()` | 获取当前项目名称 |
| `get_timeline()` | 获取当前时间线 |
| `get_media_pool()` | 获取媒体池 |

**连接逻辑：**
- 自动查找 `fusionscript.so` 库路径
- 支持 Resolve 19.0 / 20.2 版本
- 通过 `DaVinciResolveScript` 模块连接

---

### 2. MarkerManager

标记点管理器，封装了所有 Marker 操作。

**颜色映射：**

| 输入类型 | Resolve 颜色 |
|----------|-------------|
| VFX | Green |
| DI | Yellow |
| orange | Orange |
| red | Red |
| blue | Blue |
| cyan | Cyan |
| magenta | Magenta |

---

**主要方法：**

| 方法 | 说明 |
|------|------|
| `get_all_markers()` | 获取所有标记点，返回 `{frame_id: MarkerInfo}` |
| `get_markers_list()` | 获取标记点列表（按帧号排序） |
| `add_marker(frame_id, color, name, note, duration)` | 添加单个标记点 |
| `add_markers_from_csv(csv_path)` | 从 CSV 批量导入 |
| `delete_marker_at_frame(frame_id)` | 删除指定帧的标记点 |
| `delete_markers_by_color(color)` | 删除指定颜色的所有标记点 |
| `delete_all_markers()` | 删除所有标记点 |
| `get_markers_by_color(color)` | 按颜色筛选 |
| `export_markers_to_json(path)` | 导出为 JSON |

---

### 3. MarkerInfo

标记点数据结构。

**属性：**

| 属性 | 类型 | 说明 |
|------|------|------|
| `frame_id` | float | 帧号 |
| `color` | str | 颜色 |
| `name` | str | 名称 |
| `note` | str | 注释（OCR 识别的文本内容） |
| `duration` | float | 持续时间（帧） |
| `custom_data` | str | 自定义数据 |

---

## 数据流转

### 1. CSV → Resolve Timeline

```
CSV 文件
┌─────────────────────────────────────────────────────┐
│ 帧数   │ 时间码     │ 文本内容      │ ... │ 类型 │
├─────────────────────────────────────────────────────┤
│ 6106  │ 00:04:04:06 │ VFX:擦重庆    │ ... │ VFX │
│ 7722  │ 00:05:08:22 │ DI:背景黄衣...│ ... │ DI  │
│ ...   │ ...        │ ...           │ ... │ ... │
└─────────────────────────────────────────────────────┘
              │
              ▼
    add_markers_from_csv()
              │
              ▼
    ┌────────────────────────────┐
    │ MarkerManager 处理逻辑     │
    │ • 解析 CSV 行              │
    │ • 根据类型映射颜色          │
    │ • 调用 Resolve API 添加    │
    └────────────────────────────┘
              │
              ▼
    Resolve Timeline Markers
    ┌─────────────────────────────────────┐
    │ 帧 6106: Green, VFX, "擦重庆"       │
    │ 帧 7722: Yellow, DI, "背景黄衣..."  │
    └─────────────────────────────────────┘
```

### 2. Resolve → JSON

```
Resolve Timeline
┌─────────────────────────────────────┐
│ 帧 6106: Green, VFX, "擦重庆"       │
│ 帧 7722: Yellow, DI, "背景黄衣..."  │
└─────────────────────────────────────┘
              │
              ▼
    get_all_markers() / export_markers_to_json()
              │
              ▼
    JSON 文件
    ┌─────────────────────────────────────┐
    │ {                                   │
    │   "total_count": 44,               │
    │   "colors": {"Green": 35, ...},    │
    │   "markers": [{...}, {...}, ...]    │
    │ }                                   │
    └─────────────────────────────────────┘
```

---

## 使用示例

### 添加单个标记点

```python
from markerManager import MarkerManager

manager = MarkerManager()
manager.add_marker(
    frame_id=1000.0,
    color='Green',
    name='VFX',
    note='VFX:擦除穿帮',
    duration=1.0
)
```

### 从 CSV 批量导入

```python
from markerManager import MarkerManager

manager = MarkerManager()
result = manager.add_markers_from_csv('../EP25_detected_frames_paddle_refactored.csv')
print(f"成功: {result['success']}, 失败: {result['failed']}")
```

### 导出标记点

```python
from markerManager import MarkerManager

manager = MarkerManager()
manager.export_markers_to_json('markers.json')
```

### 查询标记点

```python
from markerManager import MarkerManager

manager = MarkerManager()

# 获取所有
all_markers = manager.get_all_markers()

# 按颜色筛选
vfx_markers = manager.get_markers_by_color('Green')
di_markers = manager.get_markers_by_color('Yellow')

# 获取汇总
summary = manager.get_markers_summary()
```

---

## CSV 格式要求

列索引（默认）：

| 列索引 | 内容 | 示例 |
|--------|------|------|
| 0 | 帧数 | 6106 |
| 2 | 文本内容 | VFX:擦重庆 |
| 5 | 类型 | VFX / DI |

```csv
帧数,时间码,文本内容,像素数量,置信度,类型
6106,00:04:04:06,VFX:擦重庆,1234,0.95,VFX
7722,00:05:08:22,DI:背景黄衣...,980,0.88,DI
```

---

## 运行测试

```bash
# 测试标记点管理
python test_marker_manager.py

# 测试添加功能（需要 Resolve 运行）
python test_add_markers.py
```

---

## 与 JXXS_OCR 的关系

```
JXXS_OCR                          Resolve_Marker
   │                                   │
   │  输出 CSV                         │  输入 CSV
   ▼                                   ▼
┌─────────┐    CSV    ┌─────────────────────┐
│ 视频处理 │ ────────▶ │ 时间线标记点导入     │
│ 字幕识别 │           │ (add_markers_from_csv) │
└─────────┘           └─────────────────────┘
```

JXXS_OCR 识别视频中的 VFX/DI 字幕 → 输出 CSV → Resolve_Marker 导入到 DaVinci Resolve 时间线作为标记点。
