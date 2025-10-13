# 手眼标定快速开始指南

## 问题：端口占用冲突

当 `robot_ui.py` 运行时，它会占用机器人的控制端口(29999)，导致 `calibration_eye_hand.py` 无法同时连接。

## 解决方案（3种方法）

---

### 方法1：使用示教器手动控制（最简单）✅ 推荐

#### 步骤：
1. **关闭所有机器人连接程序**（包括 robot_ui.py）
2. **运行标定程序**：
   ```bash
   python calibration_eye_hand.py
   ```
3. **选择操作1**（采集数据）
4. **使用示教器移动机器人**到不同位姿
5. **在每个位姿按 'c' 键**采集数据
6. 采集完成后，**选择操作2**进行标定计算

#### 优点：
- ✅ 不需要修改代码
- ✅ 操作简单
- ✅ 完全避免端口冲突

---

### 方法2：使用轻量级移动脚本（半自动）

#### 步骤：
1. **移动机器人到目标位姿**：
   ```bash
   # 使用预设位姿1-10
   python move_robot_preset.py --preset 1
   
   # 或自定义位姿
   python move_robot_preset.py --x 600 --y -260 --z 380 --rx 170 --ry 12 --rz 140
   ```
   脚本会自动断开连接并释放端口

2. **运行标定程序采集数据**：
   ```bash
   python calibration_eye_hand.py
   ```
   选择操作1，按 'c' 采集当前位姿数据

3. **重复步骤1-2**，直到采集足够数据（建议10-15组）

4. **执行标定计算**：选择操作2

#### 优点：
- ✅ 可以通过脚本精确控制位姿
- ✅ 自动释放端口
- ✅ 适合需要特定位姿的场景

---

### 方法3：全自动采集（最快速）⚡

**警告：机器人会自动移动，确保安全！**

#### 步骤：
1. **编辑 `auto_calibration.py`**，根据需要调整预设位姿：
   ```python
   AUTO_CALIBRATION_POSES = [
       {"x": 600, "y": -260, "z": 380, "rx": 170, "ry": 12, "rz": 140},
       # ... 更多位姿
   ]
   ```

2. **运行自动采集**：
   ```bash
   python auto_calibration.py
   ```
   
3. **确认安全后输入 'yes'** 开始自动采集

4. **运行标定计算**：
   ```bash
   python calibration_eye_hand.py
   ```
   选择操作2

#### 优点：
- ✅ 完全自动化
- ✅ 快速采集大量数据
- ✅ 一次性完成所有位姿

#### 注意：
- ⚠️ 确保工作空间安全
- ⚠️ 测试时建议先用少量位姿

---

## 推荐工作流程

### 对于新手用户：
```
方法1（使用示教器）
```

### 对于有经验用户：
```
方法2（半自动）或 方法3（全自动）
```

---

## 配置说明

### 修改机器人IP地址：

**calibration_eye_hand.py**:
```python
robot_ip = "192.168.5.2"  # 第247行
```

**move_robot_preset.py**:
```python
ROBOT_IP = "192.168.5.2"  # 第12行
```

**auto_calibration.py**:
```python
ROBOT_IP = "192.168.5.2"  # 第14行
```

### 修改棋盘格参数：

在所有脚本中找到并修改：
```python
CHESSBOARD_CORNERS_NUM_X = 9  # 棋盘格列数
CHESSBOARD_CORNERS_NUM_Y = 6  # 棋盘格行数
CHESSBOARD_SQUARE_SIZE_MM = 20  # 方格边长(mm)
```

---

## 常见问题

### Q: 仍然提示 "Connection refused"？
**A:** 确保没有其他程序占用端口：
```bash
# Windows检查端口占用
netstat -ano | findstr :29999

# 如果有进程占用，关闭该进程
```

### Q: 棋盘格检测不到？
**A:** 检查：
- 棋盘格是否清晰可见
- 光照是否充足
- 棋盘格是否平整
- 相机焦距是否对准

### Q: 采集多少组数据合适？
**A:** 
- 最少：3组（勉强可用）
- 推荐：10-15组（良好精度）
- 更多：20+组（最佳精度）

### Q: 如何验证标定结果？
**A:** 运行标定程序，选择操作4（验证标定结果）

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `calibration_eye_hand.py` | 主标定程序（手动采集） |
| `move_robot_preset.py` | 轻量级移动工具 |
| `auto_calibration.py` | 全自动采集脚本 |
| `HAND_EYE_CALIBRATION_GUIDE.md` | 详细技术说明 |
| `QUICKSTART_HAND_EYE.md` | 本快速指南 |

---

## 支持

如有问题，请查看：
1. `HAND_EYE_CALIBRATION_GUIDE.md` - 详细技术文档
2. `docs/` 目录下的其他文档
3. 代码注释

祝标定顺利！🎉
