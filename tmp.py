import os
from shutil import copyfile

folder = '/home/chli/chLi/Dataset/ShapeNet/manifold_test/'

shape_ids = os.listdir(folder)

for shape_id in shape_ids:
    shape_file_path = folder + shape_id + '/models/model_normalized.obj'
    target = folder + shape_id + '.obj'

    copyfile(shape_file_path, target)
