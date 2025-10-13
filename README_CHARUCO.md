# 如何获得 ChArUco 标定板

你的代码使用的是 **ChArUco 标定板**，参数如下：
- 14 x 9 方块
- 方块边长：20mm
- ArUco标记边长：15mm
- 总尺寸：280mm x 180mm（约A4纸大小）

---

## 🚀 快速开始（3步）

### 1️⃣ 生成标定板图像

```bash
python generate_charuco_board.py
```

这会生成 `charuco_board.png` 文件（与你代码参数完全匹配）

### 2️⃣ 打印标定板

- 使用**哑光相纸**或**厚卡纸**打印
- 打印设置：**实际尺寸**（100%，不缩放）
- 高质量模式
- 贴在硬质平板上（亚克力板、铝板等）

### 3️⃣ 验证检测

```bash
python test_charuco_detection.py
```

将标定板放在相机前，检查是否能稳定检测到绿色角点和坐标轴

---

## 📋 详细步骤

### 方法A：使用脚本生成（推荐）

1. **生成默认配置**（14x9, 20mm）：
   ```bash
   python generate_charuco_board.py
   ```

2. **交互式选择**：
   ```bash
   python generate_charuco_board.py --interactive
   ```
   可选择不同尺寸配置

3. **自定义参数**：
   ```bash
   python generate_charuco_board.py --squares-x 14 --squares-y 9 --square-length 20 --marker-length 15
   ```

### 方法B：在线生成

访问在线工具：
- https://calib.io/pages/camera-calibration-pattern-generator
- 选择 "ChArUco" 类型
- 输入参数：
  - Squares X: 14
  - Squares Y: 9
  - Square Length: 20mm
  - Marker Length: 15mm
  - Dictionary: DICT_5X5_1000

---

## 🖨️ 打印要点

### ✅ 必做
1. **纸张**：使用哑光相纸（200g+）或厚卡纸（250g+）
2. **尺寸**：实际尺寸打印，不缩放
3. **质量**：最高质量打印模式
4. **测量**：打印后用尺子验证方块大小（应为20mm±0.5mm）
5. **平整**：贴在硬质平板上

### ❌ 禁忌
1. 不要用普通A4纸（容易弯曲）
2. 不要缩放或"适应页面"
3. 不要让标定板有褶皱或弯曲
4. 不要在光滑表面打印（容易反光）

---

## ✅ 验证标定板

运行测试脚本：
```bash
python test_charuco_detection.py
```

**期望结果**：
- ✓ 能看到绿色圆点（ChArUco角点）
- ✓ 能看到彩色方框（ArUco标记）
- ✓ 能看到RGB坐标轴（位姿估计成功）
- ✓ 检测率 > 90%

**如果检测失败**：
1. 检查光照（避免强光反射、阴影）
2. 调整距离（建议30-80cm）
3. 检查打印质量（边缘应清晰锐利）
4. 检查平整度（不应有翘曲）

---

## 📐 参数对照表

| 参数 | 代码中的值 | 说明 |
|------|-----------|------|
| `squaresX` | 14 | 横向方块数 |
| `squaresY` | 9 | 纵向方块数 |
| `squareLength` | 0.02 | 方块边长（米）= 20mm |
| `markerLength` | 0.015 | ArUco标记边长（米）= 15mm |
| `aruco_type` | DICT_5X5_1000 | ArUco字典类型 |

**物理尺寸**：280mm x 180mm（宽 x 高）

---

## 🎯 使用标定板

标定板准备好后，运行你的手眼标定程序：

```bash
python eye_in_hand2.0_charuco.py
```

**操作步骤**：
1. 将标定板放在相机视野内
2. 移动机器人到不同位姿
3. 按**空格键**采集数据
4. 重复采集10-15组数据
5. 程序会自动完成标定

---

## 📚 相关文档

- `generate_charuco_board.py` - 标定板生成脚本
- `test_charuco_detection.py` - 检测测试脚本  
- `docs/CHARUCO_BOARD_GUIDE.md` - 详细制作指南
- `eye_in_hand2.0_charuco.py` - 手眼标定程序

---

## 💡 提示

1. **准备多个尺寸**：不同距离可能需要不同大小的标定板
2. **定期检查**：标定板使用一段时间后检查是否有磨损
3. **妥善保存**：平放存储，避免弯曲和污损

---

## 🆘 需要帮助？

查看完整文档：`docs/CHARUCO_BOARD_GUIDE.md`

或运行测试查看检测效果：
```bash
python test_charuco_detection.py
```
