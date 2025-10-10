"""
Intel RealSense Camera Class
支持多种 RealSense 相机型号（D435, D455, D415, L515 等）
根据相机型号自动配置最佳参数
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime
import os
from typing import Tuple, Optional, Dict, List


# 预定义各型号相机的最佳配置
CAMERA_CONFIGS = {
    'D435': {
        'color': {'width': 1280, 'height': 720, 'fps': 30, 'format': rs.format.bgr8},
        'depth': {'width': 1280, 'height': 720, 'fps': 30, 'format': rs.format.z16},
        'description': 'Intel RealSense D435'
    },
    'D435I': {
        'color': {'width': 1280, 'height': 720, 'fps': 30, 'format': rs.format.bgr8},
        'depth': {'width': 1280, 'height': 720, 'fps': 30, 'format': rs.format.z16},
        'description': 'Intel RealSense D435i (with IMU)'
    },
    'D455': {
        'color': {'width': 1280, 'height': 720, 'fps': 30, 'format': rs.format.bgr8},
        'depth': {'width': 1280, 'height': 720, 'fps': 30, 'format': rs.format.z16},
        'description': 'Intel RealSense D455'
    },
    'D415': {
        'color': {'width': 1280, 'height': 720, 'fps': 30, 'format': rs.format.bgr8},
        'depth': {'width': 1280, 'height': 720, 'fps': 30, 'format': rs.format.z16},
        'description': 'Intel RealSense D415'
    },
    'L515': {
        'color': {'width': 1920, 'height': 1080, 'fps': 30, 'format': rs.format.bgr8},
        'depth': {'width': 1024, 'height': 768, 'fps': 30, 'format': rs.format.z16},
        'description': 'Intel RealSense L515 (LiDAR)'
    },
    'SR305': {
        'color': {'width': 1920, 'height': 1080, 'fps': 30, 'format': rs.format.bgr8},
        'depth': {'width': 640, 'height': 480, 'fps': 30, 'format': rs.format.z16},
        'description': 'Intel RealSense SR305'
    },
    'DEFAULT': {
        'color': {'width': 640, 'height': 480, 'fps': 30, 'format': rs.format.bgr8},
        'depth': {'width': 640, 'height': 480, 'fps': 30, 'format': rs.format.z16},
        'description': 'Default configuration'
    }
}


class Camera:
    """
    RealSense 相机类，根据型号自动配置最佳参数

    支持的相机型号：
    - D435, D435i (D435I)
    - D455
    - D415
    - L515
    - SR305
    - 或使用 'AUTO' 自动检测
    """

    def __init__(
        self,
        camera_model: str = 'AUTO',
        enable_depth: bool = True,
        enable_color: bool = True,
        serial_number: Optional[str] = None,
        align_to_color: bool = True,
        custom_config: Optional[Dict] = None
    ):
        """
        初始化 RealSense 相机

        参数:
            camera_model: 相机型号，支持 'D435', 'D435I', 'D455', 'D415', 'L515', 'SR305', 'AUTO'
                         'AUTO' 会自动检测连接的相机型号
            enable_depth: 是否启用深度流，默认 True
            enable_color: 是否启用彩色流，默认 True
            serial_number: 相机序列号，用于多相机场景，默认 None（使用第一个相机）
            align_to_color: 是否将深度对齐到彩色图像，默认 True
            custom_config: 自定义配置字典，会覆盖默认配置
                          格式: {'color': {'width': 1280, 'height': 720, 'fps': 30}, 
                                'depth': {...}}
        
        示例:
            # 自动检测型号
            cam = Camera()
            
            # 指定型号
            cam = Camera(camera_model='D435')
            
            # 自定义配置
            cam = Camera(camera_model='D455', custom_config={
                'color': {'width': 1920, 'height': 1080, 'fps': 30}
            })
        """
        self.camera_model = camera_model.upper()
        self.enable_depth = enable_depth
        self.enable_color = enable_color
        self.serial_number = serial_number
        self.align_to_color = align_to_color
        self.custom_config = custom_config

        # 配置参数（初始化后确定）
        self.width = None
        self.height = None
        self.fps = None
        self.config_info = {}

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

    def _detect_camera_model(self) -> str:
        """
        自动检测连接的相机型号
        
        返回:
            相机型号名称
        """
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            raise RuntimeError("未找到 RealSense 设备")
        
        # 选择设备
        device = None
        if self.serial_number:
            for dev in devices:
                if dev.get_info(rs.camera_info.serial_number) == self.serial_number:
                    device = dev
                    break
            if device is None:
                raise RuntimeError(f"未找到序列号为 {self.serial_number} 的设备")
        else:
            device = devices[0]
        
        # 获取产品线（型号）
        product_line = device.get_info(rs.camera_info.product_line)
        name = device.get_info(rs.camera_info.name)
        
        print(f"检测到相机: {name} (产品线: {product_line})")
        
        # 根据产品线判断型号
        if 'D435I' in name.upper() or 'D435 I' in name.upper():
            return 'D435I'
        elif 'D435' in name.upper():
            return 'D435'
        elif 'D455' in name.upper():
            return 'D455'
        elif 'D415' in name.upper():
            return 'D415'
        elif 'L515' in name.upper():
            return 'L515'
        elif 'SR305' in name.upper():
            return 'SR305'
        else:
            print(f"警告: 未识别的型号 '{name}'，使用默认配置")
            return 'DEFAULT'

    def _get_camera_config(self) -> Dict:
        """
        获取相机配置
        
        返回:
            相机配置字典
        """
        # 如果是 AUTO，自动检测
        if self.camera_model == 'AUTO':
            self.camera_model = self._detect_camera_model()
        
        # 获取基础配置
        if self.camera_model in CAMERA_CONFIGS:
            config = CAMERA_CONFIGS[self.camera_model].copy()
        else:
            print(f"警告: 不支持的型号 '{self.camera_model}'，使用默认配置")
            config = CAMERA_CONFIGS['DEFAULT'].copy()
            self.camera_model = 'DEFAULT'
        
        # 应用自定义配置
        if self.custom_config:
            if 'color' in self.custom_config:
                config['color'].update(self.custom_config['color'])
            if 'depth' in self.custom_config:
                config['depth'].update(self.custom_config['depth'])
        
        return config

    def _initialize(self):
        """初始化相机配置"""
        try:
            # 获取相机配置
            config_dict = self._get_camera_config()
            self.config_info = config_dict
            
            print(f"使用配置: {config_dict['description']}")
            
            # 创建 pipeline 和 config
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            # 如果指定了序列号，则使用特定相机
            if self.serial_number:
                self.config.enable_device(self.serial_number)

            # 配置彩色流
            if self.enable_color:
                color_cfg = config_dict['color']
                self.config.enable_stream(
                    rs.stream.color,
                    color_cfg['width'],
                    color_cfg['height'],
                    color_cfg['format'],
                    color_cfg['fps']
                )
                self.width = color_cfg['width']
                self.height = color_cfg['height']
                self.fps = color_cfg['fps']

            # 配置深度流
            if self.enable_depth:
                depth_cfg = config_dict['depth']
                self.config.enable_stream(
                    rs.stream.depth,
                    depth_cfg['width'],
                    depth_cfg['height'],
                    depth_cfg['format'],
                    depth_cfg['fps']
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
    
    def get_distortion_coeffs(self, stream_type: str = 'color') -> Optional[np.ndarray]:
        """
        获取相机畸变系数

        参数:
            stream_type: 'color' 或 'depth'

        返回:
            畸变系数数组
        """
        if stream_type not in self.intrinsics:
            return None

        intr = self.intrinsics[stream_type]
        return np.array(intr['coeffs'])

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
    print("=" * 80)

    # 列出所有设备
    Camera.print_devices()

    print("\n" + "=" * 80)
    print("测试用例:")
    print("=" * 80)

    # 测试 1: 自动检测型号
    try:
        print("\n[测试 1] 自动检测相机型号")
        print("-" * 80)
        with Camera(camera_model='AUTO') as cam:
            print(f"✓ 检测到型号: {cam.camera_model}")
            print(f"✓ 配置: {cam.width}x{cam.height}@{cam.fps}fps")
            
            # 捕获图像
            color_image, depth_image = cam.get_frames()
            if color_image is not None:
                print(f"✓ 彩色图像尺寸: {color_image.shape}")
                cv2.imshow("Auto - Color", color_image)
            
            if depth_image is not None:
                depth_colormap = cam.get_depth_colormap(depth_image)
                cv2.imshow("Auto - Depth", depth_colormap)
            
            print("按任意键继续...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"✗ 测试 1 失败: {e}")

    # 测试 2: 指定型号
    try:
        print("\n[测试 2] 指定相机型号 (D435)")
        print("-" * 80)
        with Camera(camera_model='D435') as cam:
            print(f"✓ 使用型号: {cam.camera_model}")
            print(f"✓ 配置: {cam.width}x{cam.height}@{cam.fps}fps")
            
            # 捕获并保存
            paths = cam.capture(save_dir="test_images", prefix="d435")
            print(f"✓ 已保存文件:")
            for key, path in paths.items():
                print(f"    {key}: {path}")
                
    except Exception as e:
        print(f"✗ 测试 2 失败: {e}")

    # 测试 3: 自定义配置
    try:
        print("\n[测试 3] 使用自定义配置")
        print("-" * 80)
        custom = {
            'color': {'width': 1920, 'height': 1080, 'fps': 30},
            'depth': {'width': 1280, 'height': 720, 'fps': 30}
        }
        with Camera(camera_model='AUTO', custom_config=custom) as cam:
            print(f"✓ 型号: {cam.camera_model}")
            print(f"✓ 自定义配置: {cam.width}x{cam.height}@{cam.fps}fps")
            
            color_image = cam.get_color_image()
            if color_image is not None:
                print(f"✓ 实际图像尺寸: {color_image.shape}")
                
    except Exception as e:
        print(f"✗ 测试 3 失败: {e}")

    print("\n" + "=" * 80)
    print("所有测试完成")
    print("=" * 80)
