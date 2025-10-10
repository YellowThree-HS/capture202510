import numpy as np
from scipy.spatial.transform import Rotation
import re
import DobotApi
import time

class Robot:
    """
    一个更高级的 Dobot 机械臂控制类，封装了底层的 DobotApi。
    提供了更简洁的接口和状态管理。
    """
    def __init__(self, ip="192.168.1.6", control_port=29999, feedback_port=30003):
        """
        初始化机器人对象。
        :param ip: 机械臂的 IP 地址。
        :param control_port: 控制端口。
        :param feedback_port: 反馈端口。
        """
        self.ip = ip
        self.control_port = control_port
        self.feedback_port = feedback_port
        self.api = None
        self.connected = False
        self.enabled = False

    def connect(self):
        """
        连接到机器人。
        :return: bool, True表示成功，False表示失败。
        """
        if self.connected:
            print("机器人已连接。")
            return True
        try:
            self.api = DobotApi.DobotApi(self.ip, self.control_port, self.feedback_port)
            self.connected = True
            print(f"成功连接到机器人 {self.ip}")
            # 连接后建议清除一下错误状态
            self.clear_alarms()
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """
        断开与机器人的连接。
        """
        if self.api and self.connected:
            if self.enabled:
                self.disable() # 断开前先禁用机器人
            self.api.close()
            self.connected = False
            self.enabled = False
            print("与机器人的连接已关闭。")

    def enable(self):
        """
        启用机器人。
        """
        if not self.connected:
            raise ConnectionError("机器人未连接，请先调用 connect()。")
        self.api.EnableRobot()
        # 等待一小段时间确保机器人状态切换完成
        time.sleep(0.5)
        self.enabled = True
        print("机器人已启用。")

    def disable(self):
        """
        禁用机器人。
        """
        if not self.connected:
            raise ConnectionError("机器人未连接，请先调用 connect()。")
        self.api.DisableRobot()
        self.enabled = False
        print("机器人已禁用。")

    def home(self):
        """
        让机器人执行归零（Home）操作。
        """
        if not self.enabled:
            raise RuntimeError("机器人未启用，请先调用 enable()。")
        # Home 指令的格式，具体参数可能依赖于型号，(0)通常是默认选项
        self.api.send_data("Home(0)") 
        print("正在执行归零操作...")
        # 归零操作需要较长时间，这里仅发送指令
        # 在实际应用中可能需要轮询状态直到操作完成

    def clear_alarms(self):
        """
        清除机器人的报警状态。
        """
        if not self.connected:
            raise ConnectionError("机器人未连接，请先调用 connect()。")
        self.api.ClearError()
        print("尝试清除报警状态。")
        
    def get_alarms(self):
        """
        获取并返回当前机械臂的报警信息列表。
        :return: list or None, 报警信息字符串列表，如果没有报警则返回空列表。
        """
        if not self.connected:
            raise ConnectionError("机器人未连接，请先调用 connect()。")
        alarm_str = self.api.GetErrorID()
        # 例子: {0,0,[[1,[]],[2,[]],...]} or {1,0,[[1,[[id,level,type,'description']]],...]}
        if "[[1,[]]" in alarm_str: # 简单判断没有报警
            print("当前无报警。")
            return []
        
        # 使用正则表达式解析报警信息
        alarms = re.findall(r"\[(\d+),(\d+),(\d+),'(.*?)'\]", alarm_str)
        if not alarms:
            return []
        
        alarm_list = [f"ID:{a[0]}, Level:{a[1]}, Type:{a[2]}, Desc:'{a[3]}'" for a in alarms]
        print("检测到报警:", alarm_list)
        return alarm_list

    def move_to(self, x, y, z, r, mode='joint'):
        """
        移动机器人到指定位置。
        :param x, y, z, r: 目标点位坐标。
        :param mode: 移动模式。'joint' (关节移动, MovJ) 或 'linear' (直线移动, MovL)。
        """
        if not self.enabled:
            raise RuntimeError("机器人未启用，请先调用 enable()。")
        
        print(f"准备以 {mode} 模式移动到: ({x}, {y}, {z}, {r})")
        if mode.lower() == 'joint':
            self.api.MovJ(x, y, z, r)
        elif mode.lower() == 'linear':
            self.api.MovL(x, y, z, r)
        else:
            raise ValueError("无效的移动模式。请选择 'joint' 或 'linear'。")
        # 同样，移动需要时间，这里只发送指令

    def get_pose(self):
        """
        获取当前末端工具的位姿，并返回一个4x4的齐次变换矩阵。
        位置单位为毫米 (mm)。
        :return: numpy.ndarray or None, 4x4齐次变换矩阵。
        """
        if not self.connected:
            raise ConnectionError("机器人未连接，请先调用 connect()。")
            
        pose_str = self.api.GetPose()
        match = re.search(r'\{0,([\d.-]+),([\d.-]+),([\d.-]+),([\d.-]+),([\d.-]+),([\d.-]+)', pose_str)
        if not match:
            print("错误: 无法解析位姿字符串。")
            return None
            
        x, y, z, rx, ry, rz = map(float, match.groups())
        
        T = np.eye(4)
        T[:3, 3] = [x, y, z] # 位置 (mm)
        rot_matrix = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
        T[:3, :3] = rot_matrix
        
        return T

    # --- 上下文管理器，用于 with 语句 ---
    def __enter__(self):
        """`with`语句进入时调用: 连接并启用机器人。"""
        self.connect()
        if self.connected:
            self.enable()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """`with`语句退出时调用: 禁用并断开机器人。"""
        print("程序块结束，自动断开连接...")
        self.disconnect()

# --- 主程序：使用示例 ---
if __name__ == '__main__':
    # 请将 IP 替换为你的机器人 IP
    ROBOT_IP = "192.168.1.6"

    # --- 示例1: 标准用法 ---
    print("--- 示例1: 标准用法 ---")
    robot = Robot(ROBOT_IP)
    if robot.connect():
        try:
            robot.enable()
            
            # 获取当前报警
            robot.get_alarms()
            
            # 获取当前位姿
            current_pose_matrix = robot.get_pose()
            if current_pose_matrix is not None:
                print("当前机器人位姿矩阵:\n", np.round(current_pose_matrix, 2))
            
            # 移动机器人
            print("\n正在进行关节移动 (MovJ)...")
            robot.move_to(250, 0, 50, 0, mode='joint')
            time.sleep(5) # 等待移动完成

            print("\n正在进行直线移动 (MovL)...")
            robot.move_to(250, 100, 50, 0, mode='linear')
            time.sleep(5) # 等待移动完成
            
            # 最终禁用机器人
            robot.disable()
            
        except (ConnectionError, RuntimeError, ValueError) as e:
            print(f"发生错误: {e}")
        finally:
            # 无论如何，最后都断开连接
            robot.disconnect()
            
    # --- 示例2: 使用 `with` 语句进行优雅管理 ---
    print("\n\n--- 示例2: 使用 `with` 语句 ---")
    try:
        # with语句会自动处理连接、启用、禁用和断开
        with Robot(ROBOT_IP) as robot:
            print("机器人已在 'with' 块中自动连接并启用。")
            
            # 获取当前位姿
            pose = robot.get_pose()
            if pose is not None:
                print("当前位置 (x, y, z) in mm:", np.round(pose[:3, 3], 2))
            
            # 执行移动
            robot.move_to(200, -100, 20, 0, mode='joint')
            time.sleep(5)
            
            robot.move_to(200, 100, 20, 0, mode='linear')
            time.sleep(5)
            
        # 当代码离开 'with' 块时，__exit__ 会被自动调用，机器人被禁用和断开
        print("'with' 块执行完毕。")
        
    except Exception as e:
        print(f"在 'with' 块中发生错误: {e}")

