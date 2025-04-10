import sys
sys.path.append('../wn-nc/')

from ma_sh.Method.crop import createCroppedPcdFiles


def demo():
    createCroppedPcdFiles(
        '/home/chli/chLi/Dataset/Washer/sample_pcd/BOSCH_WLG.npy',
        '/home/chli/chLi/Results/ma-sh/output/crop/Washer/anc-1500/',
        crop_num=60,
        angle=0,
        is_crop_right=False,
        render=False,
        overwrite=False,
    )

    createCroppedPcdFiles(
        '/home/chli/chLi/Dataset/XiaomiSU7/sample_pcd/Xiaomi_SU7_2024_low_mesh.npy',
        '/home/chli/chLi/Results/ma-sh/output/crop/XiaomiSU7/anc-1500/',
        crop_num=60,
        angle=0,
        is_crop_right=True,
        render=False,
        overwrite=False,
    )

    return True
