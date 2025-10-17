import numpy as np
import os
from lib.dobot import DobotRobot

def Drag():
    robot = DobotRobot(robot_ip='192.168.5.1')  # 初始化机械臂
    robot.r_inter.StartDrag()



def txt_to_npy(txt_path: str, npy_path: str = None, delimiter: str = ' ', dtype: type = np.float32) -> np.ndarray:

    # 检查输入文件是否存在
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"错误: 输入文件 '{txt_path}' 不存在。")

    # 自动生成 .npy 输出路径
    if npy_path is None:
        # 将文件扩展名从 .txt 替换为 .npy
        base_path, _ = os.path.splitext(txt_path)
        npy_path = base_path + '.npy'

    try:

        print(f"正在读取文件: '{txt_path}'...")
        array_data = np.loadtxt(txt_path, delimiter=delimiter, dtype=dtype)
        # 移除常见的非数字字符
        chars_to_remove = "[](),"
        for char in chars_to_remove:
            content = content.replace(char, '')
        np.save(npy_path, array_data)
        return array_data

    except Exception as e:
        print(f"处理文件时发生错误: {e}")


def main():
    key = input('输入key:')
    if key == '1':
        Drag()
    elif key== '2':
        txt_to_npy('./best_hand_eye_calibration.txt')

    
if __name__ == "__main__":
    main()