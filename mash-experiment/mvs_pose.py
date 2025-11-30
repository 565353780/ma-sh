import numpy as np
import os
import json


def parse_colmap(root_folder):
    with open(os.path.join(root_folder, "transforms.json")) as f:
        data = json.load(f)

    fov = np.rad2deg(data["camera_angle_x"])
    aabb = np.array(data["aabb_range"])

    frames = data["frames"]
    Ts = []
    for frame in frames:
        # Camera model matrix
        T = np.array(frame["transform_matrix"])
        Ts.append(T)

    Ts = np.stack(Ts)
    origins = Ts[:, :3, 3]

    pc = []
    pc_n = []
    with open(os.path.join(root_folder, "points3D.txt")) as f:
        for line in f:
            line = line.strip()

            if line.startswith("#"):
                continue

            elems = line.split()
            xyz = np.array(tuple(map(float, elems[1:4])))
            image_id = np.array(tuple(map(int, elems[8::2]))) - 1

            if np.any(xyz < aabb[:, 0]) or np.any(xyz > aabb[:, 1]):
                continue

            view_dir = (origins[image_id] - xyz[None, :]).mean(0)
            view_dir /= np.linalg.norm(view_dir) + 1e-8

            pc.append(xyz)
            pc_n.append(view_dir)

    pc = np.stack(pc)
    pc_n = np.stack(pc_n)

    return fov, Ts, pc, pc_n


if __name__ == "__main__":
    from PIL import Image

    import polyscope as ps
    from icecream import ic

    root_folder = "assets/colmap"
    fov, Ts, pc, pc_n = parse_colmap(root_folder)

    img_folder = "output/ref"
    num_views = len(Ts)

    nz = np.array([0.0, 0.0, -1.0])
    y = np.array([0.0, 1.0, 0.0])

    ps.init()
    intrinsics = ps.CameraIntrinsics(fov_vertical_deg=fov, aspect=1)
    for i in range(num_views):
        tag = str(i).zfill(3)
        img = Image.open(os.path.join(img_folder, f"{tag}.png"))
        img = np.array(img) / 255.0

        R = Ts[i][:3, :3]
        t = Ts[i][:3, 3]

        extrinsics = ps.CameraExtrinsics(
            root=t, look_dir=(R @ nz - t), up_dir=(R @ y - t)
        )
        # Or equivalently
        # extrinsics = ps.CameraExtrinsics(mat=np.linalg.inv(Ts[i]))
        params = ps.CameraParameters(intrinsics, extrinsics)
        ps.register_camera_view(tag, params).add_color_image_quantity(
            "img", img, enabled=True, show_in_camera_billboard=True, transparency=0.5
        )

    ps.register_point_cloud("pc", pc).add_vector_quantity("pc_n", pc_n, enabled=True)
    ps.show()
