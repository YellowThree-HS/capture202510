from lib.camera import Camera

cam = Camera(camera_model='D405')
cam_intrinsics = cam.get_camera_matrix()
print(cam_intrinsics)