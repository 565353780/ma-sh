import os

mesh_file_path_dict = {
    "mac_plane": "/Users/fufu/Downloads/plane.obj",
    "mac_airplane": "/Users/fufu/Downloads/test.obj",
    "mac_bunny": "/Users/fufu/Downloads/bunny.obj",
    "linux_plane": "/home/chli/Downloads/plane.obj",
    "linux_airplane": "/home/chli/chLi/Dataset/ShapeNet/Core/ShapeNetCore.v2/02691156/2af04ef09d49221b85e5214b0d6a7/models/model_normalized.obj",
    "linux_bunny": "/home/chli/Downloads/bunny.obj",
    "linux_1": "/home/chli/Nutstore Files/paper-materials/Dataset/ShapeNet/mini/02773838/1b9ef45fefefa35ed13f430b2941481.obj",
    "linux_2": "/home/chli/Nutstore Files/paper-materials/Dataset/ShapeNet/mini/02801938/5c67e859bd49074c5a9dc003587438be.obj",
    "linux_3": "/home/chli/Nutstore Files/paper-materials/Dataset/ShapeNet/mini/02808440/a88e254ea9ff7b4dfe52ffd0e748a1ab.obj",
}

mac_chair_folder_path = (
    "/Users/fufu/Nutstore Files/paper-materials/Dataset/ShapeNet/mini/03001627/"
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

asdf_folder_path_dict = {
    "mac_bunny_v2-err10": "/Users/fufu/Nutstore Files/paper-materials-ASDF/Data/bunny/v2-err10/params/",
}
