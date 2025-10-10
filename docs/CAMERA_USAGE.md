# RealSense Camera 使用说明

## 📷 支持的相机型号

- **D435** / **D435I** (D435i with IMU)
- **D455**
- **D415**
- **L515** (LiDAR)
- **SR305**
- **AUTO** (自动检测)

## 🚀 快速开始

### 1. 最简单的用法（自动检测型号）

```python
from lib.camera import Camera

# 自动检测相机型号并使用最佳配置
with Camera() as cam:
    color_image, depth_image = cam.get_frames()
```

### 2. 指定相机型号

```python
# 明确指定使用 D435
with Camera(camera_model='D435') as cam:
    color_image, depth_image = cam.get_frames()

# 使用 D455
with Camera(camera_model='D455') as cam:
    color_image, depth_image = cam.get_frames()

# 使用 L515
with Camera(camera_model='L515') as cam:
    color_image, depth_image = cam.get_frames()
```

### 3. 自定义配置（覆盖默认配置）

```python
# 自定义分辨率和帧率
custom_config = {
    'color': {'width': 1920, 'height': 1080, 'fps': 30},
    'depth': {'width': 1280, 'height': 720, 'fps': 30}
}

with Camera(camera_model='D435', custom_config=custom_config) as cam:
    color_image, depth_image = cam.get_frames()
```

### 4. 只使用彩色相机

```python
with Camera(enable_depth=False) as cam:
    color_image = cam.get_color_image()
```

### 5. 多相机场景（指定序列号）

```python
# 首先列出所有相机
Camera.print_devices()

# 使用特定序列号的相机
with Camera(camera_model='D435', serial_number='123456789') as cam:
    color_image, depth_image = cam.get_frames()
```

## 📋 默认配置

### D435 / D435I / D455 / D415
- **彩色**: 1280x720 @ 30fps
- **深度**: 1280x720 @ 30fps

### L515
- **彩色**: 1920x1080 @ 30fps
- **深度**: 1024x768 @ 30fps

### SR305
- **彩色**: 1920x1080 @ 30fps
- **深度**: 640x480 @ 30fps

## 🔧 常用功能

### 获取图像

```python
# 获取对齐的彩色和深度图像
color_image, depth_image = cam.get_frames()

# 只获取彩色图像
color_image = cam.get_color_image()

# 只获取深度图像
depth_image = cam.get_depth_image()

# 获取彩色深度图（用于可视化）
depth_colormap = cam.get_depth_colormap(depth_image)
```

### 捕获并保存图像

```python
# 保存彩色、深度和彩色深度图
paths = cam.capture(
    save_dir="images",
    prefix="capture",
    save_color=True,
    save_depth=True,
    save_depth_colormap=True
)

# paths 包含保存的文件路径
print(paths['color'])           # 彩色图像路径
print(paths['depth'])           # 深度数据路径 (.npy)
print(paths['depth_colormap'])  # 彩色深度图路径
```

### 获取 3D 坐标

```python
# 根据像素坐标和深度值计算 3D 坐标
color_image, depth_image = cam.get_frames()
x, y = 320, 240  # 图像中心
point_3d = cam.get_3d_point(x, y, depth_image)

if point_3d:
    X, Y, Z = point_3d
    print(f"3D 坐标: X={X:.3f}m, Y={Y:.3f}m, Z={Z:.3f}m")
```

### 获取相机内参

```python
# 获取彩色相机内参矩阵
K_color = cam.get_camera_matrix('color')

# 获取深度相机内参矩阵
K_depth = cam.get_camera_matrix('depth')

# 获取详细内参信息
intrinsics = cam.get_intrinsics_dict()
print(intrinsics['color']['fx'])  # 焦距 x
print(intrinsics['color']['fy'])  # 焦距 y
print(intrinsics['color']['ppx']) # 主点 x
print(intrinsics['color']['ppy']) # 主点 y
```

### 列出所有连接的相机

```python
# 静态方法，无需实例化
Camera.print_devices()

# 或获取设备列表
devices = Camera.list_devices()
for dev in devices:
    print(f"型号: {dev['name']}")
    print(f"序列号: {dev['serial_number']}")
```

## 💡 完整示例

```python
from lib.camera import Camera
import cv2

# 列出所有相机
Camera.print_devices()

# 创建相机实例（自动检测型号）
with Camera(camera_model='AUTO') as cam:
    print(f"使用相机: {cam.camera_model}")
    print(f"分辨率: {cam.width}x{cam.height}@{cam.fps}fps")
    
    # 实时显示
    while True:
        color_image, depth_image = cam.get_frames()
        
        if color_image is not None:
            cv2.imshow("Color", color_image)
        
        if depth_image is not None:
            depth_colormap = cam.get_depth_colormap(depth_image)
            cv2.imshow("Depth", depth_colormap)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
```

## ⚠️ 注意事项

1. **相机型号自动检测**: 使用 `camera_model='AUTO'` 时，程序会自动识别连接的相机型号
2. **配置兼容性**: 确保自定义配置的分辨率和帧率是相机支持的
3. **深度对齐**: 默认启用深度对齐到彩色图像，可通过 `align_to_color=False` 禁用
4. **资源释放**: 使用 `with` 语句确保资源正确释放，或手动调用 `cam.release()`

## 🔍 调试

如果遇到问题，可以检查：

```python
# 获取设备信息
device_info = cam.get_device_info_dict()
print(device_info)

# 获取内参信息
intrinsics = cam.get_intrinsics_dict()
print(intrinsics)

# 检查深度比例
print(f"深度比例: {cam.depth_scale}")
```
