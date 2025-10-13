# ChArUco 标定板制作指南

## 📋 什么是 ChArUco 标定板？

ChArUco（Chessboard + ArUco）是一种结合了**棋盘格**和 **ArUco 标记**的混合标定板，具有以下优点：

- ✅ **更鲁棒的检测**：即使部分遮挡也能检测
- ✅ **亚像素精度**：结合了棋盘格的高精度
- ✅ **唯一ID识别**：每个标记有唯一编号
- ✅ **适合手眼标定**：特别适合机器人视觉应用

---

## 🎯 快速生成标定板

### 方法1：使用生成脚本（推荐）

#### 生成你代码中的配置（14x9, 20mm方块）：

```bash
python generate_charuco_board.py
```

这会生成 `charuco_board.png` 文件，参数与你代码中的配置完全匹配：
- 14 x 9 方块
- 方块边长：20mm
- ArUco标记边长：15mm
- 物理尺寸：280mm x 180mm

#### 交互式选择配置：

```bash
python generate_charuco_board.py --interactive
```

#### 自定义参数：

```bash
# 生成 10x7, 30mm方块的标定板
python generate_charuco_board.py --squares-x 10 --squares-y 7 --square-length 30 --marker-length 22

# 指定输出文件名
python generate_charuco_board.py -o my_board.png

# 设置更高的打印分辨率（600 DPI）
python generate_charuco_board.py --dpi 600
```

---

## 🖨️ 打印标定板

### 打印准备

1. **选择纸张**（重要！）
   - ✅ 推荐：哑光相纸（200g以上）
   - ✅ 推荐：厚卡纸（250g以上）
   - ❌ 不推荐：普通A4纸（容易弯曲）

2. **打印设置**
   ```
   ✓ 实际尺寸打印（100%，不缩放）
   ✓ 高质量/最佳质量模式
   ✓ 彩色或黑白均可
   ✓ 关闭"适合页面"或"自动缩放"
   ```

3. **打印测试**
   - 先打印一小块测试
   - 用尺子测量方块实际尺寸
   - 确保误差在 ±0.5mm 以内

### 标定板安装

1. **平整处理**
   - 将打印的标定板贴在硬质平板上
   - 推荐材料：亚克力板、铝板、木板
   - 使用双面胶或喷胶均匀粘贴

2. **过塑保护**（可选）
   - 可以过塑保护，防止磨损
   - 注意过塑可能影响平整度

3. **验证平整度**
   - 将标定板放在平面上
   - 确保没有翘曲或凹凸
   - 用直尺检查边缘是否平直

---

## ✅ 验证标定板质量

运行验证脚本检查标定板是否能被正确检测：

```bash
python test_charuco_detection.py
```

或手动测试：

```python
import cv2
from lib.camera import Camera

# 初始化相机
cam = Camera(camera_model='d405')

# 参数（与你的代码一致）
squaresX = 14
squaresY = 9
squareLength = 0.02
markerLength = 0.015
aruco_type = cv2.aruco.DICT_5X5_1000

# 创建检测器
dictionary = cv2.aruco.getPredefinedDictionary(aruco_type)
board = cv2.aruco.CharucoBoard((squaresX, squaresY), squareLength, markerLength, dictionary)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
charuco_detector = cv2.aruco.CharucoDetector(board)

while True:
    color_image, _ = cam.get_frames()
    
    # 检测 ChArUco 角点
    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(color_image)
    
    if charuco_corners is not None:
        cv2.aruco.drawDetectedCornersCharuco(color_image, charuco_corners, charuco_ids)
        print(f"检测到 {len(charuco_corners)} 个角点")
    
    cv2.imshow("ChArUco Detection Test", color_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```

---

## 📐 尺寸参考

### 你的配置 (14x9, 20mm)

```
方块尺寸: 20mm x 20mm
ArUco标记: 15mm x 15mm
总尺寸: 280mm x 180mm (约 A4 纸大小)
适用距离: 30cm - 80cm
```

### 其他常用尺寸

| 配置 | 方块大小 | 总尺寸 | 适用距离 | 用途 |
|------|---------|--------|---------|------|
| 14x9, 20mm | 20mm | 280x180mm | 30-80cm | 标准手眼标定 |
| 14x9, 30mm | 30mm | 420x270mm | 50-120cm | 远距离标定 |
| 10x7, 15mm | 15mm | 150x105mm | 20-50cm | 近距离/小场景 |
| 11x8, 25mm | 25mm | 275x200mm | 40-100cm | A4纸适配 |

---

## 🔧 故障排除

### 检测不到标定板？

1. **检查光照**
   - 确保光照均匀充足
   - 避免强烈反光
   - 避免阴影遮挡

2. **检查距离**
   - 标定板应在相机焦距范围内
   - 标定板在图像中占比 30%-70% 为佳

3. **检查打印质量**
   - ArUco 标记边缘清晰
   - 黑白对比度高
   - 方块边缘锐利

### 检测不稳定？

1. **增加曝光时间**（减少运动模糊）
2. **提高打印质量**
3. **检查标定板平整度**
4. **调整检测参数**

---

## 📝 在线生成器（备选）

如果无法运行脚本，可以使用在线工具：

1. **OpenCV ChArUco Board Generator**
   - 网址：https://calib.io/pages/camera-calibration-pattern-generator
   - 选择 "ChArUco" 类型
   - 输入参数：14x9, 20mm, DICT_5X5_1000

2. **Calibration Pattern Generator**
   - 搜索 "ChArUco board generator online"
   - 输入你的参数

---

## 💡 最佳实践

1. **多准备几个标定板**
   - 不同尺寸适应不同距离
   - 备用板以防损坏

2. **定期检查标定板**
   - 检查是否有污渍、磨损
   - 检查是否仍然平整

3. **存储建议**
   - 平放存储，避免弯曲
   - 避免阳光直射（防止褪色）
   - 保持清洁干燥

---

## 🎯 使用标定板进行手眼标定

生成并打印好标定板后，按照以下步骤：

1. **运行你的标定程序**：
   ```bash
   python eye_in_hand2.0_charuco.py
   ```

2. **移动机器人**：
   - 使标定板在相机视野内
   - 移动到不同位姿（距离、角度）
   - 按空格键采集数据

3. **采集建议**：
   - 至少采集 10-15 组数据
   - 覆盖不同的距离和角度
   - 确保每次检测到足够的角点

---

需要帮助？查看：
- `generate_charuco_board.py` - 生成脚本
- `test_charuco_detection.py` - 检测测试脚本
- `eye_in_hand2.0_charuco.py` - 你的标定程序
