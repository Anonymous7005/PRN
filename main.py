import numpy as np
import os
from glob import glob
import scipy.io as sio
from scipy.misc import imread, imsave, imresize
from time import time

from api import FaceProcess
from utils.write import write_obj


os.environ['CUDA_VISIBLE_DEVICES'] = '7' # GPU number
fp = FaceProcess(is_dlib = False) 

# ------------- load data
image_folder = 'AFLW2000/'
save_folder = os.path.join(image_folder, 'results')
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

types = ('*.jpg', '*.png')
image_path_list= []
for files in types:
    image_path_list.extend(glob(os.path.join(image_folder, files)))
total_num = len(image_path_list)

for i, image_path in enumerate(image_path_list):
    # read image
    image = imread(image_path)

    # the core: regress position map    
    if 'AFLW2000' in image_path:
        mat_path = image_path.replace('jpg', 'mat')
        info = sio.loadmat(mat_path)
        kpt = info['pt3d_68']
        pos = fp.process(image, kpt) # kpt information is only used for detecting face and cropping image
    else:
        pos = fp.process(image) # use dlib to detect face

    # -- Applications
    # get landmarks
    kpt = fp.get_landmarks(pos)
    # 3D vertices
    vertices = fp.get_vertices(pos)
    # corresponding colors
    colors = fp.get_colors(image, vertices)
    # texture map (need opencv) 
    # import cv2
    # texture = cv2.remap(image, pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))

    # -- save
    name = image_path.strip().split('/')[-1][:-4]
    np.savetxt(os.path.join(save_folder, name + '.txt'), kpt) 
    write_obj(os.path.join(save_folder, name + '.obj'), vertices, colors, fp.triangles) #save 3d face(can open with meshlab)

