import os
import pykitti
import numpy as np
import open3d as o3d


# 读取 calib_imu_to_velo.txt 文件
def load_imu_to_velo(calib_path):
    with open(calib_path, "r") as f:
        lines = f.readlines()

    # 提取数值部分
    rotation = list(map(float, lines[1].strip().split()[1:]))  # 去掉 "R:" 开头的字符串
    rotation = np.array(rotation).reshape(3, 3)
    translate = list(map(float, lines[2].strip().split()[1:]))  # 去掉 "R:" 开头的字符串
    translate = np.array(translate).reshape(3, 1)
    values = np.hstack([rotation, translate])
    T_imu_to_velo = np.vstack((values, [0, 0, 0, 1]))  # 变成 4x4 齐次矩阵

    return T_imu_to_velo


# 设定 KITTI 数据集路径
base_dir = "path/to/KITTI/dataset"  # 替换为你的 KITTI 数据集路径
base_dir = "/home/chli/chLi/Dataset/KITTI/raw_data/city/"
date = "2011_09_26"  # 例如 "2011_09_26"
drive = "0001"  # 例如 "0001"

calib_file = base_dir + date + "/calib_imu_to_velo.txt"

save_trans_pcd_folder_path = "/home/chli/chLi/Dataset/KITTI/trans_pcd/"

os.makedirs(save_trans_pcd_folder_path, exist_ok=True)

T_imu_to_velo = load_imu_to_velo(calib_file)

# 读取数据集
dataset = pykitti.raw(base_dir, date, drive)

# 获取第一帧的位姿（作为世界坐标的参考）
T_w_0 = dataset.oxts[0].T_w_imu  # 第一帧的 IMU 到世界坐标变换

all_transformed_points = []

# 遍历所有帧的 LiDAR 数据
for i, (velo_data, oxts) in enumerate(zip(dataset.velo, dataset.oxts)):
    # 只取 XYZ，不要强度值
    points = velo_data[:, :3]

    # 获取该帧的 IMU 到世界坐标系的变换矩阵
    T_w_i = oxts.T_w_imu

    # 计算 LiDAR 到世界坐标的变换
    T_i_l = T_imu_to_velo  # IMU 到 LiDAR 变换
    T_w_l = np.dot(T_w_i, T_i_l)  # 组合变换

    # 齐次坐标变换
    points_h = np.hstack((points, np.ones((points.shape[0], 1)))).T  # (4, N)
    transformed_points = np.dot(T_w_l, points_h)[:3, :].T  # (N, 3)

    all_transformed_points.append(transformed_points)

    trans_pcd = o3d.geometry.PointCloud()
    trans_pcd.points = o3d.utility.Vector3dVector(transformed_points)

    o3d.io.write_point_cloud(
        save_trans_pcd_folder_path + str(i) + "_pcd.ply", trans_pcd
    )

    print(f"Processed frame {i + 1}")

# 合并所有点云
merged_points = np.vstack(all_transformed_points)

print("Point cloud fusion completed!")

merged_pcd = o3d.geometry.PointCloud()
merged_pcd.points = o3d.utility.Vector3dVector(merged_points)

o3d.io.write_point_cloud(
    "./output/merged_kitti_lidar.ply", merged_pcd, write_ascii=True
)

# o3d.visualization.draw_geometries([merged_pcd])
