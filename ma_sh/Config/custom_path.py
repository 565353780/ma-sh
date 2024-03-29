import os

mesh_file_path_dict = {
    "mac_plane": "/Users/fufu/Downloads/plane.obj",
    "mac_airplane": "/Users/fufu/Downloads/test.obj",
    "mac_bunny": "/Users/fufu/Downloads/bunny.obj",
    "linux_plane": "/home/chli/Downloads/plane.obj",
    "linux_airplane": "/home/chli/chLi/Dataset/ShapeNet/Core/ShapeNetCore.v2/02691156/2af04ef09d49221b85e5214b0d6a7/models/model_normalized.obj",
    "linux_bunny": "/home/chli/Downloads/bunny.obj",
    "linux_1": "/home/chli/Nutstore Files/paper-materials-ASDF/Dataset/mini/02691156/1b171503b1d0a074bc0909d98a1ff2b4.obj",
    "linux_2": "/home/chli/Nutstore Files/paper-materials-ASDF/Dataset/mini/03001627/2c251c2c49aaffd54b3c42e318f3affc.obj",
}

mac_chair_folder_path = (
    "/Users/fufu/Nutstore Files/paper-materials-ASDF/Dataset/mini/03001627/"
)
if os.path.exists(mac_chair_folder_path):
    obj_filename_list = os.listdir(mac_chair_folder_path)
    chair_idx = 0
    for obj_filename in obj_filename_list:
        if obj_filename[-4:] != ".obj":
            continue
        mesh_file_path_dict["mac_chair_" + str(chair_idx)] = (
            mac_chair_folder_path + obj_filename
        )
        chair_idx += 1

append_dict = {}

for key, item in mesh_file_path_dict.items():
    if key.startswith("mac_"):
        new_key = "mac_linux_" + key[4:]
        new_item = item.replace("/Users/fufu", "/media/psf/Home")

        append_dict[new_key] = new_item

mesh_file_path_dict.update(append_dict)

asdf_folder_path_dict = {
    "mac_bunny_v2-err10": "/Users/fufu/Nutstore Files/paper-materials-ASDF/Data/bunny/v2-err10/params/",
}
