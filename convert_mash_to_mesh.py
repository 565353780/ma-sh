from time import sleep

from ma_sh.Demo.Convertor.mash_to_mesh import demo as demo_convert_mash_to_mesh

if __name__ == "__main__":
    keep_alive = False

    while True:
        # demo_convert_mash('Objaverse_82K/manifold_mash/', 'Objaverse_82K/manifold_mash_recon/')
        # demo_convert_mash('ShapeNet/manifold_mash/', 'ShapeNet/manifold_mash_recon/')
        # demo_convert_mash_to_mesh('Thingi10K/mesh_mash-1600anc/', 'Thingi10K/mesh_mash_recon-1600anc/')
        # demo_convert_mash_to_mesh('KITTI/mash-400anc/', 'KITTI/mash_recon-400anc/')
        # demo_convert_mash_to_mesh('KITTI/mash-1600anc/', 'KITTI/mash_recon-1600anc/')
        # demo_convert_mash_to_mesh('ShapeNet/manifold_mash-4096-400anc/', 'ShapeNet/manifold_mash_recon-4096-400anc/')
        # demo_convert_mash_to_mesh('ShapeNet/manifold_mash-4096-200anc/', 'ShapeNet/manifold_mash_recon-4096-200anc/')
        # demo_convert_mash_to_mesh('ShapeNet/manifold_mash-4096-100anc/', 'ShapeNet/manifold_mash_recon-4096-100anc/')
        # demo_convert_mash_to_mesh('SimpleShapes/mesh_mash-1600anc/', 'SimpleShapes/mesh_mash_recon-1600anc/')
        # demo_convert_mash_to_mesh('SimpleShapes/mesh_mash-400anc/', 'SimpleShapes/mesh_mash_recon-400anc/')
        # demo_convert_mash_to_mesh('SimpleShapes/mesh_mash-100anc/', 'SimpleShapes/mesh_mash_recon-100anc/')
        # demo_convert_mash_to_mesh('SimpleShapes/mesh_mash-50anc/', 'SimpleShapes/mesh_mash_recon-50anc/')
        # demo_convert_mash_to_mesh('SimpleShapes/mesh_mash-10anc/', 'SimpleShapes/mesh_mash_recon-10anc/')
        # demo_convert_mash_to_mesh('SimpleShapes/mesh_mash-6anc/', 'SimpleShapes/mesh_mash_recon-6anc/')
        # demo_convert_mash_to_mesh('SimpleShapes/mesh_mash-4anc/', 'SimpleShapes/mesh_mash_recon-4anc/')

        demo_convert_mash_to_mesh('Thingi10K/mesh_mash-1600anc/', 'test/Thingi10K/mesh_mash_recon-1600anc/')
        demo_convert_mash_to_mesh('KITTI/mash-400anc/', 'test/KITTI/mash_recon-400anc/')
        demo_convert_mash_to_mesh('KITTI/mash-1600anc/', 'test/KITTI/mash_recon-1600anc/')
        demo_convert_mash_to_mesh('ShapeNet/manifold_mash-4096-400anc/', 'test/ShapeNet/manifold_mash_recon-4096-400anc/')
        demo_convert_mash_to_mesh('ShapeNet/manifold_mash-4096-200anc/', 'test/ShapeNet/manifold_mash_recon-4096-200anc/')
        demo_convert_mash_to_mesh('ShapeNet/manifold_mash-4096-100anc/', 'test/ShapeNet/manifold_mash_recon-4096-100anc/')
        demo_convert_mash_to_mesh('SimpleShapes/mesh_mash-1600anc/', 'test/SimpleShapes/mesh_mash_recon-1600anc/')
        demo_convert_mash_to_mesh('SimpleShapes/mesh_mash-400anc/', 'test/SimpleShapes/mesh_mash_recon-400anc/')
        demo_convert_mash_to_mesh('SimpleShapes/mesh_mash-100anc/', 'test/SimpleShapes/mesh_mash_recon-100anc/')
        demo_convert_mash_to_mesh('SimpleShapes/mesh_mash-50anc/', 'test/SimpleShapes/mesh_mash_recon-50anc/')
        demo_convert_mash_to_mesh('SimpleShapes/mesh_mash-10anc/', 'test/SimpleShapes/mesh_mash_recon-10anc/')
        demo_convert_mash_to_mesh('SimpleShapes/mesh_mash-6anc/', 'test/SimpleShapes/mesh_mash_recon-6anc/')
        demo_convert_mash_to_mesh('SimpleShapes/mesh_mash-4anc/', 'test/SimpleShapes/mesh_mash_recon-4anc/')

        if not keep_alive:
            break

        sleep(1)
