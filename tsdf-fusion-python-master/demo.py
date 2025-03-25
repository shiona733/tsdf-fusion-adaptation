import time
import cv2
import numpy as np
import fusion

if __name__ == "__main__":
    # 数据集路径
    dataset_path = "/root/tsdf-fusion-python-master/scene0106_00"

    # 读取相机内参
    cam_intr_color = np.loadtxt(f"{dataset_path}/intrinsic/intrinsic_color.txt", delimiter=' ')
    cam_intr_depth = np.loadtxt(f"{dataset_path}/intrinsic/intrinsic_depth.txt", delimiter=' ')

    # 估计体素体积边界
    print("Estimating voxel volume bounds...")
    n_imgs = 2323
    vol_bnds = np.zeros((3, 2))
    for i in range(n_imgs):
        # 读取深度图像和相机位姿
        depth_im_path = f"{dataset_path}/depth/{i}.png"
        depth_im = cv2.imread(depth_im_path, -1).astype(float)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to dataset)

        cam_pose_path = f"{dataset_path}/pose/{i}.txt"
        cam_pose = np.loadtxt(cam_pose_path)  # 4x4 rigid transformation matrix

        # 计算相机视锥体和扩展凸包，使用深度相机内参
        view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr_depth, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

    # 初始化体素体积
    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)

    # 循环遍历RGB-D图像并融合
    t0_elapse = time.time()
    for i in range(n_imgs):
        print(f"Fusing frame {i + 1}/{n_imgs}")

        # 读取RGB-D图像和相机位姿
        color_image_path = f"{dataset_path}/color/{str(i).zfill(2)}.jpg"
        color_image = cv2.cvtColor(cv2.imread(color_image_path), cv2.COLOR_BGR2RGB)

        depth_im_path = f"{dataset_path}/depth/{i}.png"
        depth_im = cv2.imread(depth_im_path, -1).astype(float)
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0


        cam_pose_path = f"{dataset_path}/pose/{i}.txt"
        cam_pose = np.loadtxt(cam_pose_path)

        # 假设颜色和深度对齐，将观测融合到体素体积中
        # 注意：这里需要根据实际情况调整，确保颜色和深度信息正确融合
        # 可以在 integrate 函数中分别处理彩色和深度内参
        tsdf_vol.integrate(color_image, depth_im, cam_intr_depth, cam_pose, obs_weight=1.)
    fps = n_imgs / (time.time() - t0_elapse)
    print(f"Average FPS: {fps:.2f}")

    # 从体素体积中获取网格并保存到磁盘（可以用Meshlab查看）
    print("Saving mesh to mesh_new.ply...")
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    fusion.meshwrite("mesh_new.ply", verts, faces, norms, colors)

    # 从体素体积中获取点云并保存到磁盘（可以用Meshlab查看）
    print("Saving point cloud to pc_new.ply...")
    point_cloud = tsdf_vol.get_point_cloud()
    fusion.pcwrite("pc_new.ply", point_cloud)