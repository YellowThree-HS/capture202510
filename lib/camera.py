"""
Intel RealSense Camera Class
支持多种 RealSense 相机型号（D435, D455, D415, L515 等）
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime
import os
from typing import Tuple, Optional, Dict, List


class Camera:
    """
    RealSense 相机类，支持多种型号

    支持的相机型号：
    - D435, D435i
    - D455
    - D415
    - L515
    - SR305
    等所有 Intel RealSense 相机
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = True,
        enable_color: bool = True,
        serial_number: Optional[str] = None,
        align_to_color: bool = True
    ):
        """
        初始化 RealSense 相机

        参数:
            width: 图像宽度，默认 640
            height: 图像高度，默认 480
            fps: 帧率，默认 30
            enable_depth: 是否启用深度流，默认 True
            enable_color: 是否启用彩色流，默认 True
            serial_number: 相机序列号，用于多相机场景，默认 None（使用第一个相机）
            align_to_color: 是否将深度对齐到彩色图像，默认 True
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth
        self.enable_color = enable_color
        self.serial_number = serial_number
        self.align_to_color = align_to_color

        # RealSense 对象
        self.pipeline = None
        self.config = None
        self.align = None
        self.profile = None

        # 相机信息
        self.device_info = {}
        self.depth_scale = None
        self.intrinsics = {}

        # 初始化相机
        self._initialize()

    def _initialize(self):
        """初始化相机配置"""
        try:
            # 创建 pipeline 和 config
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            # 如果指定了序列号，则使用特定相机
            if self.serial_number:
                self.config.enable_device(self.serial_number)

            # 配置彩色流
            if self.enable_color:
                self.config.enable_stream(
                    rs.stream.color,
                    self.width,
                    self.height,
                    rs.format.bgr8,
                    self.fps
                )

            # 配置深度流
            if self.enable_depth:
                self.config.enable_stream(
                    rs.stream.depth,
                    self.width,
                    self.height,
                    rs.format.z16,
                    self.fps
                )

            # 启动 pipeline
            self.profile = self.pipeline.start(self.config)

            # 创建对齐对象（将深度对齐到彩色）
            if self.align_to_color and self.enable_depth and self.enable_color:
                self.align = rs.align(rs.stream.color)

            # 获取设备信息
            self._get_device_info()

            # 获取深度比例
            if self.enable_depth:
                depth_sensor = self.profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()

            # 获取相机内参
            self._get_intrinsics()

            # 预热相机
            self._warmup()

            print(f"✓ RealSense 相机初始化成功")
            print(f"  型号: {self.device_info.get('name', 'Unknown')}")
            print(f"  序列号: {self.device_info.get('serial_number', 'Unknown')}")
            print(f"  分辨率: {self.width}x{self.height}@{self.fps}fps")
            if self.depth_scale:
                print(f"  深度比例: {self.depth_scale:.6f}")

        except Exception as e:
            raise RuntimeError(f"相机初始化失败: {str(e)}")

    def _get_device_info(self):
        """获取设备信息"""
        try:
            device = self.profile.get_device()
            self.device_info = {
                'name': device.get_info(rs.camera_info.name),
                'serial_number': device.get_info(rs.camera_info.serial_number),
                'firmware_version': device.get_info(rs.camera_info.firmware_version),
                'product_id': device.get_info(rs.camera_info.product_id),
                'usb_type': device.get_info(rs.camera_info.usb_type_descriptor),
            }
        except Exception as e:
            print(f"警告：无法获取设备信息: {e}")

    def _get_intrinsics(self):
        """获取相机内参"""
        try:
            if self.enable_color:
                color_profile = self.profile.get_stream(rs.stream.color)
                color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
                self.intrinsics['color'] = {
                    'width': color_intrinsics.width,
                    'height': color_intrinsics.height,
                    'fx': color_intrinsics.fx,
                    'fy': color_intrinsics.fy,
                    'ppx': color_intrinsics.ppx,
                    'ppy': color_intrinsics.ppy,
                    'coeffs': color_intrinsics.coeffs,
                    'model': str(color_intrinsics.model)
                }

            if self.enable_depth:
                depth_profile = self.profile.get_stream(rs.stream.depth)
                depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
                self.intrinsics['depth'] = {
                    'width': depth_intrinsics.width,
                    'height': depth_intrinsics.height,
                    'fx': depth_intrinsics.fx,
                    'fy': depth_intrinsics.fy,
                    'ppx': depth_intrinsics.ppx,
                    'ppy': depth_intrinsics.ppy,
                    'coeffs': depth_intrinsics.coeffs,
                    'model': str(depth_intrinsics.model)
                }
        except Exception as e:
            print(f"警告：无法获取相机内参: {e}")

    def _warmup(self, frames: int = 30):
        """
        预热相机，跳过前几帧

        参数:
            frames: 跳过的帧数，默认 30
        """
        for _ in range(frames):
            self.pipeline.wait_for_frames()

    def get_frames(self, aligned: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        获取一帧图像

        参数:
            aligned: 是否使用对齐的帧（深度对齐到彩色），默认 True

        返回:
            (color_image, depth_image): 彩色图像和深度图像的元组
            - color_image: BGR 格式的彩色图像 (H, W, 3)
            - depth_image: 深度图像 (H, W)，单位为毫米
            - 如果未启用对应流，则返回 None
        """
        try:
            # 等待新的一帧
            frames = self.pipeline.wait_for_frames()

            # 对齐深度到彩色
            if aligned and self.align and self.enable_depth and self.enable_color:
                frames = self.align.process(frames)

            # 获取彩色帧
            color_image = None
            if self.enable_color:
                color_frame = frames.get_color_frame()
                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())

            # 获取深度帧
            depth_image = None
            if self.enable_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())

            return color_image, depth_image

        except Exception as e:
            print(f"获取帧失败: {e}")
            return None, None

    def get_color_image(self) -> Optional[np.ndarray]:
        """
        仅获取彩色图像

        返回:
            color_image: BGR 格式的彩色图像
        """
        color_image, _ = self.get_frames()
        return color_image

    def get_depth_image(self) -> Optional[np.ndarray]:
        """
        仅获取深度图像

        返回:
            depth_image: 深度图像，单位为毫米
        """
        _, depth_image = self.get_frames()
        return depth_image

    def get_depth_colormap(self, depth_image: np.ndarray) -> np.ndarray:
        """
        将深度图像转换为彩色显示

        参数:
            depth_image: 深度图像

        返回:
            depth_colormap: 彩色深度图像
        """
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        return depth_colormap

    def get_3d_point(self, x: int, y: int, depth_image: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        根据像素坐标和深度值计算 3D 坐标

        参数:
            x: 像素 x 坐标
            y: 像素 y 坐标
            depth_image: 深度图像

        返回:
            (X, Y, Z): 3D 坐标（米），如果深度无效则返回 None
        """
        if not self.enable_depth or 'depth' not in self.intrinsics:
            print("深度流未启用或内参未获取")
            return None

        try:
            # 获取深度值（毫米）
            depth_value = depth_image[y, x]
            if depth_value == 0:
                return None

            # 转换为米
            depth_meters = depth_value * self.depth_scale

            # 获取内参
            intr = self.intrinsics['depth']

            # 计算 3D 坐标
            X = (x - intr['ppx']) / intr['fx'] * depth_meters
            Y = (y - intr['ppy']) / intr['fy'] * depth_meters
            Z = depth_meters

            return (X, Y, Z)

        except Exception as e:
            print(f"计算 3D 坐标失败: {e}")
            return None

    def capture(
        self,
        save_dir: str = "images",
        prefix: str = "capture",
        save_color: bool = True,
        save_depth: bool = True,
        save_depth_colormap: bool = True
    ) -> Dict[str, str]:
        """
        捕获并保存图像

        参数:
            save_dir: 保存目录
            prefix: 文件名前缀
            save_color: 是否保存彩色图像
            save_depth: 是否保存原始深度图像（.npy 格式）
            save_depth_colormap: 是否保存彩色深度图像

        返回:
            保存的文件路径字典 {'color': path, 'depth': path, 'depth_colormap': path}
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 获取图像
        color_image, depth_image = self.get_frames()

        # 生成时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        saved_paths = {}

        # 保存彩色图像
        if save_color and color_image is not None:
            color_path = os.path.join(save_dir, f"{prefix}_color_{timestamp}.jpg")
            cv2.imwrite(color_path, color_image)
            saved_paths['color'] = color_path
            print(f"✓ 彩色图像已保存: {color_path}")

        # 保存原始深度图像
        if save_depth and depth_image is not None:
            depth_path = os.path.join(save_dir, f"{prefix}_depth_{timestamp}.npy")
            np.save(depth_path, depth_image)
            saved_paths['depth'] = depth_path
            print(f"✓ 深度数据已保存: {depth_path}")

        # 保存彩色深度图像
        if save_depth_colormap and depth_image is not None:
            depth_colormap = self.get_depth_colormap(depth_image)
            depth_colormap_path = os.path.join(save_dir, f"{prefix}_depth_colormap_{timestamp}.jpg")
            cv2.imwrite(depth_colormap_path, depth_colormap)
            saved_paths['depth_colormap'] = depth_colormap_path
            print(f"✓ 彩色深度图像已保存: {depth_colormap_path}")

        return saved_paths

    def get_camera_matrix(self, stream_type: str = 'color') -> Optional[np.ndarray]:
        """
        获取相机内参矩阵

        参数:
            stream_type: 'color' 或 'depth'

        返回:
            3x3 相机内参矩阵
        """
        if stream_type not in self.intrinsics:
            return None

        intr = self.intrinsics[stream_type]
        K = np.array([
            [intr['fx'], 0, intr['ppx']],
            [0, intr['fy'], intr['ppy']],
            [0, 0, 1]
        ])
        return K

    def get_device_info_dict(self) -> Dict:
        """获取设备信息字典"""
        return self.device_info.copy()

    def get_intrinsics_dict(self) -> Dict:
        """获取相机内参字典"""
        return self.intrinsics.copy()

    def release(self):
        """释放相机资源"""
        if self.pipeline:
            self.pipeline.stop()
            print("✓ 相机资源已释放")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.release()

    def __del__(self):
        """析构函数"""
        self.release()

    @staticmethod
    def list_devices() -> List[Dict]:
        """
        列出所有连接的 RealSense 设备

        返回:
            设备信息列表
        """
        ctx = rs.context()
        devices = ctx.query_devices()

        device_list = []
        for i, device in enumerate(devices):
            device_info = {
                'index': i,
                'name': device.get_info(rs.camera_info.name),
                'serial_number': device.get_info(rs.camera_info.serial_number),
                'firmware_version': device.get_info(rs.camera_info.firmware_version),
                'product_id': device.get_info(rs.camera_info.product_id),
            }
            device_list.append(device_info)

        return device_list

    @staticmethod
    def print_devices():
        """打印所有连接的 RealSense 设备"""
        devices = Camera.list_devices()

        if not devices:
            print("未检测到 RealSense 设备")
            return

        print(f"\n检测到 {len(devices)} 个 RealSense 设备:")
        print("=" * 80)
        for dev in devices:
            print(f"设备 {dev['index']}:")
            print(f"  名称: {dev['name']}")
            print(f"  序列号: {dev['serial_number']}")
            print(f"  固件版本: {dev['firmware_version']}")
            print(f"  产品 ID: {dev['product_id']}")
            print("-" * 80)


if __name__ == "__main__":
    # 测试代码
    print("RealSense Camera Class Test")
    print("=" * 50)

    # 列出所有设备
    Camera.print_devices()

    # 创建相机实例
    try:
        with Camera(width=640, height=480, fps=30) as cam:
            print("\n开始捕获图像...")

            # 捕获并保存
            paths = cam.capture(save_dir="test_images")

            # 显示图像
            color_image, depth_image = cam.get_frames()

            if color_image is not None:
                cv2.imshow("Color", color_image)

            if depth_image is not None:
                depth_colormap = cam.get_depth_colormap(depth_image)
                cv2.imshow("Depth", depth_colormap)

            print("\n按任意键退出...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"错误: {e}")
