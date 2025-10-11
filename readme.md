# 机器人视觉抓取系统 (Robot Vision Grasping System)

## 项目概述

本项目实现了一个基于视觉的机器人抓取系统，集成了 Intel RealSense 深度相机、YOLO 目标检测、SAM 分割和 Dobot 机械臂控制。系统能够自动识别、定位和抓取目标物体。

## 系统架构

```
相机采集 → 目标检测与分割 → 位姿估计 → 机械臂控制 → 抓取执行
   ↓              ↓              ↓           ↓          ↓
Camera       YOLO+SAM      mask2pose    Robot      Grasping
```

## 核心模块

### 1. 相机模块 (`lib/camera.py`)

#### `Camera` 类
智能的 RealSense 相机控制类，支持多种型号的自动配置。

**主要特性:**
- 自动检测相机型号（D435, D435i, D455, D415, L515, SR305）
- 根据设备自动优化分辨率和帧率配置
- 内参标定和深度对齐
- 3D 点云计算
- 图像捕获和保存

**使用示例:**
```python
from lib.camera import Camera

# 自动检测相机型号
cam = Camera(camera_model='AUTO')

# 获取RGB-D图像
color_image, depth_image = cam.get_frames()

# 获取3D坐标
x, y, z = cam.get_3d_point(pixel_x, pixel_y, depth_image)

# 获取相机内参
K = cam.get_camera_matrix('color')
```


### 2. 机器人控制模块 (`lib/robot.py`)

#### `Robot` 类
高级的 Dobot 机械臂控制类，封装了底层 API 并提供简洁接口。

**主要特性:**
- 自动连接和启用机器人
- 位姿控制（关节空间和笛卡尔空间）
- 错误处理和报警管理
- 状态监控和反馈
- 上下文管理器支持

**使用示例:**
```python
from lib.robot import Robot

# 创建机器人实例
robot = Robot("192.168.1.6")

# 使用上下文管理器
with robot:
    # 移动到指定位置
    robot.move_to(x=300, y=200, z=100, r=0)
    
    # 获取当前位姿
    pose_matrix = robot.get_pose()
    
    # 执行归零操作
    robot.home()
```

**API 结构:**
- **控制端口**: 29999 (DobotApiDashboard)
- **反馈端口**: 30004 (DobotApiFeedBack)
- **错误处理**: 自动解析返回码和错误信息

### 3. 视觉检测模块 (`lib/yolo_and_sam.py`)

#### `YOLOSegmentator` 类
集成 YOLO 目标检测和 SAM 实例分割的视觉处理类。

**主要特性:**
- YOLO-World 开放词汇目标检测
- FastSAM 快速分割
- 多目标同时检测
- 结果可视化和保存

**使用示例:**
```python
from lib.yolo_and_sam import YOLOSegmentator

segmentator = YOLOSegmentator()

# 检测和分割指定类别
result = segmentator.detect_and_segment_all(
    image_path="image.jpg",
    categories=['cup', 'spoon'],
    save_result=True
)
```

### 4. 位姿估计模块 (`lib/mask2pose.py`)

#### `mask2pose` 函数
从分割掩码和深度信息估计物体的 6DoF 位姿。

**主要特性:**
- 基于点云的位姿估计
- PCA 主成分分析计算方向
- 多种物体类型的适配
- 相机坐标系到机器人坐标系转换

**使用示例:**
```python
from lib.mask2pose import mask2pose

pose, transform = mask2pose(
    mask=obj_mask,
    depth_image=depth_meters,
    color_image=color_image,
    intrinsics=cam.intrinsics,
    T_cam2base=calibration_matrix,
    object_class='cup'
)
```

### 5. 物体位姿检测模块 (`lib/object_pose.py`)

完整的物体检测和位姿估计流水线。

## 配置文件

### 相机配置 (CAMERA_CONFIGS)
```python
'D435': {
    'color': {'width': 640, 'height': 480, 'fps': 30},
    'depth': {'width': 640, 'height': 480, 'fps': 30},
    'description': 'Intel RealSense D435'
}
```

### 预训练模型
- **YOLO-World**: `weights/yolov8s-world.pt`
- **FastSAM**: `weights/FastSAM-s.pt`

## 使用指南

### 环境要求
```bash
# 安装依赖
pip install -r requirements.txt

或者

uv sync

# 主要依赖包
- pyrealsense2  # RealSense 相机支持
- opencv-python # 图像处理
- ultralytics   # YOLO 模型
- numpy         # 数值计算
- scipy         # 科学计算
```

### 项目结构
```
capture202510/
├── lib/                    # 核心模块
│   ├── camera.py          # 相机控制
│   ├── robot.py           # 机器人控制
│   ├── yolo_and_sam.py    # 视觉检测
│   ├── mask2pose.py       # 位姿估计
│   └── object_pose.py     # 物体位姿检测
├── weights/               # 预训练模型
│   ├── yolov8s-world.pt
│   └── FastSAM-s.pt
├── docs/                  # 文档
├── images/                # 测试图像
├── result/                # 结果输出
├── test.py               # 测试脚本
├── main.py               # 主程序
└── calibration_eye_hand.py # 手眼标定
```


## 开发日志

### 已完成功能
- [-] RealSense 相机多型号支持和自动配置
- [-] Dobot 机械臂控制 API 封装
- [-] YOLO+SAM 视觉检测流水线
- [ ] 6DoF 位姿估计算法
- [ ] 手眼标定功能
- [ ] 完整的抓取演示程序