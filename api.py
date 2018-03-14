import numpy as np
import os
from scipy.misc import imread, imsave, imresize
from skimage.transform import estimate_transform, warp
from time import time

from predictor import PosPrediction


class FaceProcess:
    '''
    Face Reconstruction
    Face Alignment
    '''
    def __init__(self, is_dlib = True, prefix = '.'):
        self.resolution_inp = 256
        self.resolution_op = 256

        #---- load detectors
        if is_dlib:
            import dlib
            path_to_detector = os.path.join(prefix, 'Data/net-data/mmod_human_face_detector.dat')
            self.face_detector = dlib.cnn_face_detection_model_v1(
                    path_to_detector)

        #---- load pos net 
        self.pos_predictor = PosPrediction(self.resolution_inp, self.resolution_op)
        model_path = os.path.join(prefix, 'Data/net-data/256_256_resfcn256_weight')
        self.pos_predictor.restore(model_path)

        # file
        self.uv_kpt_ind = np.loadtxt(prefix + '/Data/uv-data/uv_kpt_ind.txt').astype(np.int32) # 2 x 68 get kpt
        self.face_ind = np.loadtxt(prefix + '/Data/uv-data/face_ind.txt').astype(np.int32) # get valid vertices in the pos map
        self.triangles = np.loadtxt(prefix + '/Data/uv-data/triangles.txt').astype(np.int32) # ntri x 3

    def dlib_detect(self, image):
        return self.face_detector(image, 1)

    def net_forward(self, image):
        return self.pos_predictor.predict(image)

    def process(self, input, image_info = None):
        '''
        Args:
            image: (h,w,3). value range:1~255. 
        Returns:
            pos: the 3D position map. (256, 256, 3).
        '''
        if isinstance(input, str):
            try:
                image = imread(input)
            except IOError:
                print("error opening file: ", input)
                return None
        else:
            image = input

        if image.ndim < 3:
            image = np.tile(image[:,:,np.newaxis], [1,1,3])

        if image_info is not None:
            if np.max(image_info.shape) > 4: # key points
                kpt = image_info
                if kpt.shape[0] > 3:
                    kpt = kpt.T
                left = np.min(kpt[0, :]); right = np.max(kpt[0, :]); 
                top = np.min(kpt[1,:]); bottom = np.max(kpt[1,:])
            else: # bounding box
                bbox = image_info
                left = bbox[0]; right = bbox[1]; top = bbox[2]; bottom = bbox[3]
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
            size = int(old_size*1.6)
        else:
            detected_faces = self.dlib_detect(image)
            if len(detected_faces) == 0:
                print('warning: no detected face')
                return None

            d = detected_faces[0].rect ## only use the first detected face (assume that each input image only contain one face)
            left = d.left(); right = d.right(); top = d.top(); bottom = d.bottom()
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size*0.14])
            size = int(old_size*1.58)

        # crop image
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        image = image/255.
        cropped_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))

        # run our net
        # st = time()
        cropped_pos = self.net_forward(cropped_image)
        # print 'net time:', time() - st

        # restore 
        cropped_pos = np.squeeze(cropped_pos)
        cropped_vertices = np.reshape(cropped_pos, [-1, 3]).T
        z = cropped_vertices[2,:].copy()/tform.params[0,0]
        cropped_vertices[2,:] = 1
        vertices = np.dot(np.linalg.inv(tform.params), cropped_vertices)
        vertices = np.vstack((vertices[:2,:], z))
        pos = np.reshape(vertices.T, [self.resolution_op, self.resolution_op, 3])
        
        return pos
            
    def get_landmarks(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            kpt: 68 3D landmarks. shape = (68, 3).
        '''
        kpt = pos[self.uv_kpt_ind[1,:], self.uv_kpt_ind[0,:], :]
        return kpt


    def get_vertices(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            vertices: the vertices(point cloud). shape = (num of points, 3). n is 45128 here.
        '''
        all_vertices = np.reshape(pos, [self.resolution_op**2, -1]);
        vertices = all_vertices[self.face_ind, :]

        return vertices

    def get_colors(self, image, vertices):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
        '''
        [h, w, _] = image.shape
        vertices[:,0] = np.minimum(np.maximum(vertices[:,0], 0), w - 1)  # x
        vertices[:,1] = np.minimum(np.maximum(vertices[:,1], 0), h - 1)  # y
        ind = np.round(vertices).astype(np.int32)
        colors = image[ind[:,1], ind[:,0], :] # n x 3

        return colors








