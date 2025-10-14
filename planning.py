import time
import numpy as np
from lib.dobot import DobotRobot
from lib.inspire_hand import InspireHand
from lib.dobot_api import MyType


class DobotPlanning(object):
    def __init__(self, robotip = "192.168.1.1", no_gripper = False):
        self.dobot = DobotRobot(robotip, no_gripper=no_gripper)
        self.in_init = False
        self.gripper_on = False

        # 运动检测相关参数
        self.last_pose = None
        self.last_joints = None
        self.motion_threshold = 1.0  # 位置变化阈值(mm)
        self.rotation_threshold = 1.0  # 旋转变化阈值(度)
        self.joint_threshold = 1.0  # 关节角度变化阈值(度)
        self.stable_count = 0  # 稳定计数
        self.stable_threshold = 3  # 连续稳定次数


    def moveto_init(self):
        init_joint_state = np.array([-90.0, 0.0, -90.0, 0.0, 90.0, 90.0, 1.0])  # left ram
        # init_joint_state = np.array([90.0, -0.0, 70.0, 0.0, -90.0, -90.0, 1.0])  # right arm
        print(f"move to init config {init_joint_state}.")
        self.dobot.moveJ(init_joint_state)
        self.wait_motion_complete_by_status(timeout=30)
        self.in_init = True
        self.gripper_on = True

    def grasp(self, grasp_pose: np.ndarray):
        """
        grasp_pose: xyz(mm), rxryrz(deg))
        """
        if not self.in_init:
            self.moveto_init()

        self.dobot.moveL(grasp_pose)

        self.moveto_init()
        self.in_init = True
        self.gripper_on = True

    def place(self, place_pose: np.ndarray):
        """
        place_pose: xyz(mm), rxryrz(deg)
        """
        if not self.in_init:
            self.moveto_init()

        self.dobot.moveL(place_pose)

        self.moveto_init()
        self.in_init = True
        self.gripper_on = True

    def test(self):
        """
        按键控制机械臂运动
        """
        trans_step = 100  # 平移步长100mm
        rot_step = 10  # 旋转度数10deg

        print("=" * 50)
        print("Dobot 测试程序")
        print("=" * 50)
        print("控制说明:")
        print("平移 - I/K(X轴), J/L(Y轴), W/S(Z轴)")
        print("旋转 - Q/E(rx), A/D(ry), Z/C(rz)")
        print("退出 - P")
        print("=" * 50)

        while True:
            try:
                key = input("请输入指令后按回车: ").strip().lower()

                if not key:
                    continue

                if key == 'p':
                    print("程序退出")
                    break

                curr_pose = self.dobot.get_XYZrxryrz_state()
                moved = False

                if key == "i":
                    curr_pose[0] += trans_step
                    print(f"X轴正向移动: +{trans_step}mm")
                    moved = True
                elif key == "k":
                    curr_pose[0] -= trans_step
                    print(f"X轴负向移动: -{trans_step}mm")
                    moved = True
                elif key == "j":
                    curr_pose[1] += trans_step
                    print(f"Y轴正向移动: +{trans_step}mm")
                    moved = True
                elif key == "l":
                    curr_pose[1] -= trans_step
                    print(f"Y轴负向移动: -{trans_step}mm")
                    moved = True
                elif key == "w":
                    curr_pose[2] += trans_step
                    print(f"Z轴正向移动: +{trans_step}mm")
                    moved = True
                elif key == "s":
                    curr_pose[2] -= trans_step
                    print(f"Z轴负向移动: -{trans_step}mm")
                    moved = True
                elif key == "q":
                    curr_pose[3] += rot_step
                    print(f"rx轴正向旋转: +{rot_step}deg")
                    moved = True
                elif key == "e":
                    curr_pose[3] -= rot_step
                    print(f"rx轴负向旋转: -{rot_step}deg")
                    moved = True
                elif key == "a":
                    curr_pose[4] += rot_step
                    print(f"ry轴正向旋转: +{rot_step}deg")
                    moved = True
                elif key == "d":
                    curr_pose[4] -= rot_step
                    print(f"ry轴负向旋转: -{rot_step}deg")
                    moved = True
                elif key == "z":
                    curr_pose[5] += rot_step
                    print(f"rz轴正向旋转: +{rot_step}deg")
                    moved = True
                elif key == "c":
                    curr_pose[5] -= rot_step
                    print(f"rz轴负向旋转: -{rot_step}deg")
                    moved = True
                elif key == "r":
                    self.moveto_init()
                    print(f"回到原位")
                else:
                    print("无效指令，请重新输入")
                    continue

                if moved:
                    self.dobot.moveL(curr_pose)

            except KeyboardInterrupt:
                print("\n程序被用户中断")
                break
            except Exception as e:
                print(f"发生错误: {e}")
                continue

    def is_moving(self, use_joints=True, check_interval=0.1):
        """
        判断机械臂是否在运动中
        参数:
        use_joints: True使用关节角度判断，False使用末端位姿判断
        check_interval: 检查间隔(秒)
        返回:
        True: 正在运动, False: 已停止运动
        """
        # 获取当前状态
        if use_joints:
            current_state = self.dobot.get_joint_state()
            current_state_deg = np.array([np.rad2deg(rad) for rad in current_state[:6]])
        else:
            current_state = self.dobot.get_XYZrxryrz_state()
            current_state_deg = np.array(current_state[:6])  # x,y,z,rx,ry,rz

        # 第一次调用，初始化参考状态
        if self.last_pose is None:
            self.last_pose = current_state_deg.copy()
            if use_joints:
                self.last_joints = current_state_deg.copy()
            time.sleep(check_interval)
            return True  # 第一次调用认为在运动中

        # 计算变化量
        if use_joints:
            delta = np.abs(current_state_deg - self.last_joints)
            max_delta = np.max(delta)
            # 更新参考状态
            self.last_joints = current_state_deg.copy()
        else:
            # 分别处理位置和旋转
            pos_delta = np.abs(current_state_deg[:3] - self.last_pose[:3])
            rot_delta = np.abs(current_state_deg[3:] - self.last_pose[3:])
            max_delta = max(np.max(pos_delta), np.max(rot_delta))
            # 更新参考状态
            self.last_pose = current_state_deg.copy()

        # 判断是否在运动
        if max_delta > (self.joint_threshold if use_joints else max(self.motion_threshold, self.rotation_threshold)):
            self.stable_count = 0
            return True
        else:
            self.stable_count += 1
            # 连续多次稳定才认为真正停止
            return self.stable_count < self.stable_threshold

    # def wait_for_motion_completion(self, timeout=1000, check_interval=0.1, use_joints=True):
    #     """
    #         等待机械臂运动完成

    #         参数:
    #         timeout: 最大等待时间(秒)
    #         check_interval: 检查间隔(秒)
    #         use_joints: 使用关节角度还是末端位姿进行判断

    #         返回:
    #         True: 运动完成, False: 超时
    #     """
    #     # start_time = time.time()
    #     t = 0
    #     while t < timeout:
    #         print("is_moving:", self.is_moving())
    #         if not self.is_moving(use_joints=use_joints, check_interval=check_interval):
    #             return True
    #         time.sleep(check_interval)
    #         t += 1
    #         print("t:", t)

    #     print("警告：机械臂运动等待超时")
    #     return False

    def get_robot_status_data(self):
        """从30004端口获取完整的机器人状态数据"""
        try:
            data = bytes()
            hasRead = 0
            self.dobot.robot_status.socket_dobot.settimeout(0.1)
            
            while hasRead < 1440:
                temp = self.dobot.robot_status.socket_dobot.recv(1440 - hasRead)
                if len(temp) > 0:
                    hasRead += len(temp)
                    data += temp
            
            # 解析二进制数据
            feedInfo = np.frombuffer(data, dtype=MyType)
            return feedInfo
        except Exception as e:
            print(f"[get_robot_status_data] 错误: {e}")
            return None

    def wait_motion_complete_by_status(self, timeout=30):
        """使用状态端口(30004)等待运动完成"""
        start_time = time.time()
        time.sleep(0.1)  # 确保运动已开始
        
        stable_count = 0
        required_stable = 5
        
        while time.time() - start_time < timeout:
            try:
                status_data = self.get_robot_status_data()
                
                if status_data is not None:
                    # 获取运行状态
                    # running_status: 1=暂停, 2=运行中, 3=空闲, 4=报警
                    running_status = status_data['running_status'][0]
                    robot_mode = status_data['robot_mode'][0]
                    
                    print(f"[wait] robot_mode={robot_mode}, running_status={running_status}")
                    
                    # 判断是否空闲 (running_status==3 表示空闲)
                    if running_status == 3:
                        stable_count += 1
                        if stable_count >= required_stable:
                            print(f"[wait] 运动完成,耗时: {time.time()-start_time:.2f}秒")
                            return True
                    else:
                        stable_count = 0
                
                time.sleep(0.05)
                
            except Exception as e:
                print(f"[wait] 状态读取异常: {e}")
                time.sleep(0.1)
        
        print(f"[wait] 超时")
        return False


    def moveto_grasp(self):
        if not self.in_init:
            self.moveto_init()
        # time.sleep(30)
        curr_joints_rad = self.dobot.get_joint_state()
        print(curr_joints_rad)
        curr_joints_angle = [np.rad2deg(rad) for rad in curr_joints_rad]
        print(curr_joints_angle)
        curr_joints_angle[3] += 90
        print(curr_joints_angle)
        self.dobot.moveJ(np.array(curr_joints_angle))

        # curr_pose = self.dobot.get_XYZrxryrz_state()
        # curr_pose[4] -= 90
        # self.dobot.moveL(curr_pose)

        # curr_pose[2] -= 100
        # self.dobot.moveL(curr_pose)


class HandManipulation(object):
    def __init__(self):
        self.hand = InspireHand(port='/dev/ttyUSB0')
        self.hand.open_hand([800, 800, 800, 800, 1000, 0])
        time.sleep(1)

    def grasp_cup(self):
        self.hand.open_hand([800, 800, 800, 800, 1000, 0])



if __name__ == "__main__":
    dobot_planning = DobotPlanning(robotip="192.168.5.1", no_gripper=True)
    # hand_manipulation = HandManipulation()

    dobot_planning.moveto_init()
    dobot_planning.moveto_grasp()
    # hand_manipulation.grasp_cup()
    # print("1:", dobot_planning.dobot.get_XYZrxryrz_state())
    #
    # joint_rad = dobot_planning.dobot.get_joint_state()
    # joint_angle = [np.rad2deg(rad) for rad in joint_rad]
    # print("2:", dobot_planning.dobot.r_inter.PositiveSolution(offset1=joint_angle[0],
    #                                                          offset2=joint_angle[1],
    #                                                          offset3=joint_angle[2],
    #                                                          offset4=joint_angle[3],
    #                                                          offset5=joint_angle[4],
    #                                                          offset6=joint_angle[5],
    #                                                          user=0,
    #                                                          tool=1))

    # hand = InspireHand(port='/dev/ttyUSB0')
    # hand.close_hand([100, 0, 0, 0, 0, 0])
    # hand.open_hand([1000, 1000, 1000, 1000, 1000, 1000])
    # hand.open_hand([0, 0, 0, 0, 1000, 1000])
    # hand.open_hand()
    # hand.close_hand()
    # print(dobot_planning.dobot.get_joint_state())
    # dobot_planning.test()




