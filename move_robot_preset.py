"""
轻量级机器人移动脚本
用于手眼标定时快速移动机器人到预设位姿后立即释放端口

使用方法：
    python move_robot_preset.py --preset 1
    python move_robot_preset.py --x 600 --y -260 --z 380 --rx 170 --ry 12 --rz 140
"""

import argparse
import time
from lib.robot import Robot

# 机器人IP配置
ROBOT_IP = "192.168.5.2"

# 预设位姿（用于手眼标定的推荐位姿）
# 这些位姿应该覆盖不同的角度和高度
PRESET_POSES = {
    1: {"x": 600, "y": -260, "z": 380, "rx": 170, "ry": 12, "rz": 140},
    2: {"x": 550, "y": -200, "z": 400, "rx": 180, "ry": 0, "rz": 130},
    3: {"x": 650, "y": -300, "z": 350, "rx": 160, "ry": 20, "rz": 150},
    4: {"x": 600, "y": -150, "z": 420, "rx": 175, "ry": 5, "rz": 120},
    5: {"x": 500, "y": -250, "z": 360, "rx": 170, "ry": 15, "rz": 145},
    6: {"x": 700, "y": -280, "z": 390, "rx": 165, "ry": 10, "rz": 135},
    7: {"x": 580, "y": -220, "z": 410, "rx": 180, "ry": 0, "rz": 140},
    8: {"x": 620, "y": -270, "z": 370, "rx": 170, "ry": 18, "rz": 138},
    9: {"x": 550, "y": -300, "z": 400, "rx": 160, "ry": 20, "rz": 150},
    10: {"x": 650, "y": -200, "z": 380, "rx": 175, "ry": 8, "rz": 125},
}


def move_to_pose(robot, x, y, z, rx, ry, rz, mode='joint'):
    """
    移动机器人到指定位姿
    
    Args:
        robot: Robot对象
        x, y, z: 位置坐标 (mm)
        rx, ry, rz: 姿态角度 (度)
        mode: 移动模式 ('joint' 或 'linear')
    """
    print(f"\n准备移动到位姿: X={x}, Y={y}, Z={z}, Rx={rx}, Ry={ry}, Rz={rz}")
    print(f"移动模式: {mode}")
    
    try:
        success = robot.move_to(x, y, z, rx, ry, rz, mode=mode)
        if success:
            print("移动指令已发送，等待机器人到达目标位姿...")
            # 等待机器人完成移动（根据实际情况调整等待时间）
            time.sleep(5)
            print("✓ 机器人已到达目标位姿")
            return True
        else:
            print("✗ 移动指令发送失败")
            return False
    except Exception as e:
        print(f"✗ 移动过程中发生错误: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='轻量级机器人移动工具')
    
    # 预设位姿选项
    parser.add_argument('--preset', type=int, choices=range(1, 11),
                        help='使用预设位姿 (1-10)')
    
    # 自定义位姿选项
    parser.add_argument('--x', type=float, help='X坐标 (mm)')
    parser.add_argument('--y', type=float, help='Y坐标 (mm)')
    parser.add_argument('--z', type=float, help='Z坐标 (mm)')
    parser.add_argument('--rx', type=float, help='Rx角度 (度)')
    parser.add_argument('--ry', type=float, help='Ry角度 (度)')
    parser.add_argument('--rz', type=float, help='Rz角度 (度)')
    
    # 移动模式
    parser.add_argument('--mode', type=str, default='joint',
                        choices=['joint', 'linear'],
                        help='移动模式 (默认: joint)')
    
    # 机器人IP
    parser.add_argument('--ip', type=str, default=ROBOT_IP,
                        help=f'机器人IP地址 (默认: {ROBOT_IP})')
    
    args = parser.parse_args()
    
    # 确定目标位姿
    if args.preset:
        pose = PRESET_POSES[args.preset]
        print(f"使用预设位姿 #{args.preset}")
    elif all([args.x is not None, args.y is not None, args.z is not None,
              args.rx is not None, args.ry is not None, args.rz is not None]):
        pose = {
            "x": args.x, "y": args.y, "z": args.z,
            "rx": args.rx, "ry": args.ry, "rz": args.rz
        }
        print("使用自定义位姿")
    else:
        parser.print_help()
        print("\n错误: 请指定 --preset 或完整的自定义位姿参数")
        return
    
    # 连接并移动机器人
    print(f"\n正在连接机器人 ({args.ip})...")
    robot = Robot(args.ip)
    
    try:
        if robot.connect():
            print("✓ 机器人连接成功")
            
            # 启用机器人
            if robot.enable():
                print("✓ 机器人已启用")
                
                # 移动到目标位姿
                success = move_to_pose(
                    robot,
                    pose["x"], pose["y"], pose["z"],
                    pose["rx"], pose["ry"], pose["rz"],
                    mode=args.mode
                )
                
                if success:
                    print("\n✓ 任务完成！机器人已到达目标位姿")
                    print("提示: 现在可以运行 calibration_eye_hand.py 进行数据采集")
                else:
                    print("\n✗ 移动失败")
                
            else:
                print("✗ 机器人启用失败")
        else:
            print("✗ 机器人连接失败")
            print("请检查:")
            print("  1. 机器人IP地址是否正确")
            print("  2. 网络连接是否正常")
            print("  3. 是否有其他程序占用端口")
    
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        # 断开连接，释放端口
        robot.disconnect()
        print("\n✓ 已断开连接，端口已释放")
        print("现在可以运行标定程序了")


if __name__ == "__main__":
    main()
