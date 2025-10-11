import cv2
import numpy as np

# ==== 棋盘格参数 ====
CHESSBOARD_CORNERS_NUM_X = 9     # 内角点的列数
CHESSBOARD_CORNERS_NUM_Y = 6     # 内角点的行数
CHESSBOARD_SQUARE_SIZE_MM = 20   # 单个方格的物理边长 (mm)

# ==== A4 页面参数 ====
A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297
DPI = 300                         # 打印分辨率（可改为 150, 600 等）

# ==== 计算像素换算 ====
PIXELS_PER_MM = DPI / 25.4
A4_WIDTH_PX = int(A4_WIDTH_MM * PIXELS_PER_MM)
A4_HEIGHT_PX = int(A4_HEIGHT_MM * PIXELS_PER_MM)

# ==== 棋盘格尺寸计算 ====
num_squares_x = CHESSBOARD_CORNERS_NUM_X + 1
num_squares_y = CHESSBOARD_CORNERS_NUM_Y + 1
square_size_px = int(CHESSBOARD_SQUARE_SIZE_MM * PIXELS_PER_MM)
board_width_px = num_squares_x * square_size_px
board_height_px = num_squares_y * square_size_px

# ==== 创建白底画布（A4 尺寸） ====
canvas = np.ones((A4_HEIGHT_PX, A4_WIDTH_PX), dtype=np.uint8) * 255

# ==== 居中打印棋盘 ====
offset_x = (A4_WIDTH_PX - board_width_px) // 2
offset_y = (A4_HEIGHT_PX - board_height_px) // 2

for i in range(num_squares_y):
    for j in range(num_squares_x):
        if (i + j) % 2 == 0:
            x0 = offset_x + j * square_size_px
            y0 = offset_y + i * square_size_px
            x1 = x0 + square_size_px
            y1 = y0 + square_size_px
            cv2.rectangle(canvas, (x0, y0), (x1, y1), 0, -1)  # 黑方格

# ==== 输出 ====
SAVE_PATH = "chessboard_A4_300DPI.png"
cv2.imwrite(SAVE_PATH, canvas)

print(f"✅ 标定板已保存到 {SAVE_PATH}")
print(f"棋盘格 {num_squares_x}×{num_squares_y}，单格 {CHESSBOARD_SQUARE_SIZE_MM} mm")
print(f"总棋盘尺寸 {num_squares_x*CHESSBOARD_SQUARE_SIZE_MM} mm × {num_squares_y*CHESSBOARD_SQUARE_SIZE_MM} mm")
print(f"A4 页面像素尺寸: {A4_WIDTH_PX}×{A4_HEIGHT_PX} @ {DPI} DPI")
