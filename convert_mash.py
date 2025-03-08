from time import sleep

from ma_sh.Demo.Convertor.mash import demo as demo_convert_mash


if __name__ == "__main__":
    keep_alive = False

    while True:
        # demo_convert_mash(400, 'Objaverse_82K/manifold_pcd/', 'Objaverse_82K/manifold_mash/')
        # demo_convert_mash(400, 'ShapeNet/manifold_pcd/', 'ShapeNet/manifold_mash/')
        demo_convert_mash(1600, 'Thingi10K/mesh_pcd/', 'Thingi10K/mesh_mash-1600anc/')
        demo_convert_mash(400, 'KITTI/pcd/', 'KITTI/mash-400anc/')
        demo_convert_mash(1600, 'KITTI/pcd/', 'KITTI/mash-1600anc/')
        demo_convert_mash(400, 'ShapeNet/manifold_pcd-4096/', 'ShapeNet/manifold_mash-4096-400anc/')
        demo_convert_mash(200, 'ShapeNet/manifold_pcd-4096/', 'ShapeNet/manifold_mash-4096-200anc/')
        demo_convert_mash(100, 'ShapeNet/manifold_pcd-4096/', 'ShapeNet/manifold_mash-4096-100anc/')

        if not keep_alive:
            break

        sleep(1)
