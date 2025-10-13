"""
生成 ChArUco 标定板
ChArUco 标定板结合了棋盘格和 ArUco 标记，提供更好的检测鲁棒性

使用方法:
    python generate_charuco_board.py
    
生成的图片可以用于打印或显示在屏幕上
"""

import cv2
import numpy as np

def generate_charuco_board(
    squares_x=14,           # 横向方块数
    squares_y=9,            # 纵向方块数
    square_length=0.02,     # 方块边长（米）
    marker_length=0.015,    # ArUco标记边长（米）
    aruco_dict=cv2.aruco.DICT_5X5_1000,  # ArUco字典类型
    output_path="charuco_board.png",
    dpi=300,                # 打印分辨率
    margin_size=20          # 边距（像素）
):
    """
    生成 ChArUco 标定板图像
    
    参数:
        squares_x: 横向方块数
        squares_y: 纵向方块数
        square_length: 方块边长（米）
        marker_length: ArUco标记边长（米）
        aruco_dict: ArUco字典类型
        output_path: 输出文件路径
        dpi: 打印分辨率（每英寸点数）
        margin_size: 白色边距（像素）
    """
    
    # 创建字典
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
    
    # 创建 ChArUco 标定板
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length,
        marker_length,
        dictionary
    )
    
    # 计算图像尺寸（像素）
    # 将米转换为英寸，再转换为像素
    meters_to_inches = 39.3701  # 1米 = 39.3701英寸
    pixels_per_meter = dpi * meters_to_inches
    
    img_width = int(squares_x * square_length * pixels_per_meter) + 2 * margin_size
    img_height = int(squares_y * square_length * pixels_per_meter) + 2 * margin_size
    
    print(f"生成 ChArUco 标定板:")
    print(f"  - 尺寸: {squares_x} x {squares_y} 方块")
    print(f"  - 方块边长: {square_length*1000} mm")
    print(f"  - ArUco标记边长: {marker_length*1000} mm")
    print(f"  - 物理尺寸: {squares_x*square_length*1000} x {squares_y*square_length*1000} mm")
    print(f"  - 图像尺寸: {img_width} x {img_height} 像素")
    print(f"  - DPI: {dpi}")
    
    # 生成标定板图像
    board_image = board.generateImage((img_width, img_height), marginSize=margin_size)
    
    # 保存图像
    cv2.imwrite(output_path, board_image)
    print(f"\n✓ 标定板已保存到: {output_path}")
    
    # 显示预览
    display_image = cv2.resize(board_image, (0, 0), fx=0.3, fy=0.3)
    cv2.imshow("ChArUco Board Preview (30% size)", display_image)
    print("\n按任意键关闭预览窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return board_image


def generate_multiple_sizes():
    """生成多个不同尺寸的标定板供选择"""
    
    print("="*60)
    print("ChArUco 标定板生成器")
    print("="*60)
    
    configs = [
        {
            "name": "默认配置 (14x9, 20mm方块)",
            "squares_x": 14,
            "squares_y": 9,
            "square_length": 0.02,
            "marker_length": 0.015,
            "output": "charuco_board_14x9_20mm.png"
        },
        {
            "name": "大尺寸 (14x9, 30mm方块)",
            "squares_x": 14,
            "squares_y": 9,
            "square_length": 0.03,
            "marker_length": 0.022,
            "output": "charuco_board_14x9_30mm.png"
        },
        {
            "name": "小尺寸 (10x7, 15mm方块)",
            "squares_x": 10,
            "squares_y": 7,
            "square_length": 0.015,
            "marker_length": 0.011,
            "output": "charuco_board_10x7_15mm.png"
        },
        {
            "name": "A4纸适配 (11x8, 25mm方块)",
            "squares_x": 11,
            "squares_y": 8,
            "square_length": 0.025,
            "marker_length": 0.019,
            "output": "charuco_board_A4_25mm.png"
        }
    ]
    
    print("\n可用配置:")
    for i, config in enumerate(configs, 1):
        print(f"  {i}. {config['name']}")
    print(f"  5. 全部生成")
    print(f"  0. 退出")
    
    choice = input("\n请选择 (0-5): ").strip()
    
    if choice == "0":
        print("退出程序")
        return
    elif choice == "5":
        print("\n生成所有配置...")
        for config in configs:
            print(f"\n--- {config['name']} ---")
            generate_charuco_board(
                squares_x=config["squares_x"],
                squares_y=config["squares_y"],
                square_length=config["square_length"],
                marker_length=config["marker_length"],
                output_path=config["output"]
            )
    elif choice in ["1", "2", "3", "4"]:
        config = configs[int(choice) - 1]
        print(f"\n生成: {config['name']}")
        generate_charuco_board(
            squares_x=config["squares_x"],
            squares_y=config["squares_y"],
            square_length=config["square_length"],
            marker_length=config["marker_length"],
            output_path=config["output"]
        )
    else:
        print("无效选择")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='生成 ChArUco 标定板')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='交互式选择配置')
    parser.add_argument('--squares-x', type=int, default=14,
                        help='横向方块数 (默认: 14)')
    parser.add_argument('--squares-y', type=int, default=9,
                        help='纵向方块数 (默认: 9)')
    parser.add_argument('--square-length', type=float, default=20,
                        help='方块边长(mm) (默认: 20)')
    parser.add_argument('--marker-length', type=float, default=15,
                        help='ArUco标记边长(mm) (默认: 15)')
    parser.add_argument('--output', '-o', type=str, default='charuco_board.png',
                        help='输出文件路径 (默认: charuco_board.png)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='打印分辨率 (默认: 300)')
    
    args = parser.parse_args()
    
    if args.interactive:
        generate_multiple_sizes()
    else:
        generate_charuco_board(
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_length=args.square_length / 1000,  # 转换为米
            marker_length=args.marker_length / 1000,  # 转换为米
            output_path=args.output,
            dpi=args.dpi
        )
    
    print("\n" + "="*60)
    print("打印说明:")
    print("="*60)
    print("1. 使用高质量纸张（推荐：哑光相纸或厚卡纸）")
    print("2. 打印设置:")
    print("   - 实际尺寸打印（不要缩放）")
    print("   - 高质量/最佳质量模式")
    print("   - 关闭边距自动调整")
    print("3. 打印后:")
    print("   - 将标定板平整地贴在硬质平面上")
    print("   - 确保标定板没有弯曲或褶皱")
    print("   - 可以过塑保护")
    print("4. 验证尺寸:")
    print("   - 使用尺子测量实际方块大小")
    print("   - 确保与设定值一致（误差<1mm）")
    print("="*60)


if __name__ == "__main__":
    main()
