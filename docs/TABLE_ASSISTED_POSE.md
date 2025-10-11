# 桌面辅助的杯子位姿估计

## 📋 概述

这是一个改进的杯子3D位姿估计方法，利用桌面信息来提高估计的鲁棒性和准确性。

## 🎯 核心改进

### **原始方法 vs 新方法**

| 特性 | 原始方法 | 桌面辅助方法 |
|------|---------|-------------|
| 检测目标 | 杯口平面 | 桌面平面 |
| 位置参考点 | 杯口中心 | 杯子底部中心（贴桌面） |
| 朝向确定 | 杯口法向量 | 桌面法向量 |
| 高度信息 | 无 | ✅ 杯子实际高度 |
| 遮挡鲁棒性 | 低（需要看到杯口） | ✅ 高（依赖桌面） |

## 🔧 实现原理

### **步骤1: 点云提取**

```
检测框 (bbox) → 点云 A (杯子 + 桌面)
分割掩码 (mask) → 点云 B (只有杯子)
```

### **步骤2: 桌面检测**

```python
# 从检测框点云中使用RANSAC检测最大平面
plane_model, inliers = bbox_point_cloud.segment_plane(
    distance_threshold=0.008,  # 8mm容差
    ransac_n=3,
    num_iterations=1000
)
```

- 提取桌面法向量（向上）
- 计算桌面高度（Z坐标）
- 验证：桌面应该在杯子下方

### **步骤3: 位姿计算**

```python
# 1. 杯子XY平面中心
cup_xy_center = mean(cup_points[:, :2])

# 2. 底部中心（投影到桌面）
bottom_center = [cup_xy_center[0], cup_xy_center[1], table_height]

# 3. 杯子高度
cup_height = max(cup_points[:, 2]) - table_height

# 4. 构建坐标系（Z轴 = 桌面法向量）
```

## 📊 返回值

### **mask2pose_with_table() 返回值**

```python
pose, T, extra_info = mask2pose_with_table(...)

# pose: [x, y, z, roll, pitch, yaw]
#   - x, y, z: 杯子底部中心位置（米）
#   - roll, pitch, yaw: 姿态角（度）

# T: 4x4变换矩阵

# extra_info: {
#     'height': 杯子高度（米）,
#     'table_height': 桌面高度（米）,
#     'bottom_center': 底部中心坐标,
#     'method': 'table_assisted'
# }
```

## 🚀 使用方法

### **自动使用（已集成到main.py）**

运行 `main.py` 时，会自动使用新的桌面辅助方法：

```bash
python main.py
```

按空格键触发检测，系统会：
1. 检测杯子的边界框和掩码
2. 提取检测框内的点云（包含桌面）
3. 自动检测桌面平面
4. 计算基于桌面的杯子位姿

### **手动调用**

```python
from lib.mask2pose import mask2pose_with_table

# 准备数据
mask = ...          # 杯子掩码 (H, W)
bbox = [x1, y1, x2, y2]  # 检测框
depth_image = ...   # 深度图（米）
color_image = ...   # 彩色图
intrinsics = ...    # 相机内参

# 调用新方法
pose, T, extra_info = mask2pose_with_table(
    mask=mask,
    bbox=bbox,
    depth_image=depth_image,
    color_image=color_image,
    intrinsics=intrinsics,
    T_cam2base=None,
    object_class="cup"
)

# 使用结果
if pose is not None:
    x, y, z = pose[:3]
    roll, pitch, yaw = pose[3:]
    height = extra_info['height']
    
    print(f"杯子底部位置: ({x:.3f}, {y:.3f}, {z:.3f})")
    print(f"杯子高度: {height:.3f}m")
```

## ✅ 优势

1. **更准确的位置**
   - 位置在杯子底部（贴桌面），更适合机器人抓取规划
   - 不受杯口遮挡影响

2. **更鲁棒的朝向**
   - 利用桌面法向量确定"上"方向
   - 即使杯口部分遮挡也能正确估计

3. **额外的高度信息**
   - 可以计算杯子实际高度
   - 有助于抓取点规划和碰撞检测

4. **智能回退机制**
   - 如果桌面检测失败，自动回退到原始方法
   - 保证系统鲁棒性

## ⚠️ 限制条件

1. **检测框必须包含部分桌面**
   - 建议检测框略大于物体（已在代码中实现）
   
2. **桌面必须是平面**
   - 不支持不规则表面（如布料、沙发）
   
3. **杯子必须放在桌面上**
   - 不支持手持或悬空的杯子

## 🔍 调试信息

运行时会输出详细的调试信息：

```
============================================================
🔲 使用桌面辅助的位姿估计方法 (物体: cup)
============================================================

🔍 开始检测桌面...
✅ 桌面检测成功:
   桌面高度: 0.450m
   桌面法向量: [0.001, 0.002, 0.999]
   桌面点数: 1234

🔍 杯子位姿计算（基于桌面）:
   底部中心: [0.123, 0.456, 0.450]
   杯子高度: 0.105m (10.5cm)
   杯子顶部Z: 0.555m
   桌面高度: 0.450m

✅ cup位姿估计成功 (桌面辅助法):
   底部位置: [0.123, 0.456, 0.450]
   姿态: [0.5°, -1.2°, 45.3°]
   杯子高度: 0.105m (10.5cm)
```

## 📝 代码位置

- **核心函数**: `lib/mask2pose.py`
  - `mask2pose_with_table()` - 主入口函数
  - `detect_table_from_bbox()` - 桌面检测
  - `calculate_cup_pose_with_table()` - 位姿计算

- **调用代码**: `main.py` (第169-207行)

## 🎨 可视化

可以使用 `v` 键进行3D可视化，坐标系位置表示杯子底部中心。

---

**作者**: AI Assistant  
**日期**: 2025-10-11  
**版本**: 1.0

