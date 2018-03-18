import numpy as np
import cv2
from scipy.misc import imread, imsave, imresize

# list = []
# list.append([18,19,20,21,22] - 1)
# list.append([23,24,25,26,27] - 1)

# list.append([37,38,39,40,41,42] - 1)
# list.append([43,44,45,46,47,48] - 1)

# list.append([28,29,30, 31] - 1)
end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype = np.int32) - 1
def plot_kpt(image, kpt):

    image = image.copy()
    kpt = np.round(kpt).astype(np.int32)
    for i in range(68):
        st = kpt[:2, i]
        image = cv2.circle(image,(st[0], st[1]), 1, (0,0,1), 2)  
        if i in end_list:
            continue
        ed = kpt[:2, i + 1]
        image = cv2.line(image, (st[0], st[1]), (ed[0], ed[1]), (1, 1, 1), 1)
    return image

def plot_vertices(image, vertices):
    # vertices: 2 x n
    image = image.copy()
    vertices = np.round(vertices).astype(np.int32)
    for i in range(0, vertices.shape[1]):
        st = vertices[:2, i]
        image = cv2.circle(image,(st[0], st[1]), 1, (1,0,0), -1)  
    return image

# def plot_vertices(image, vertices):
#     # vertices: 2 x n
#     image = image.copy()
#     mask = np.zeros((image.shape[0], image.shape[1], 0))
#     plot = np.zeros_like(image)
#     vertices = np.round(vertices).astype(np.int32)
#     for i in range(0, vertices.shape[1]):
#         st = vertices[:2, i]
#         mask[st[0], st[1]] = 
#         plot = cv2.circle(plot,(st[0], st[1]), 1, (1,0,0), 1)  
#     return image



def get_point_weight(point, tri_points):
    ''' Get the weights of the position
    Methods: https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
     -m1.compute the area of the triangles formed by embedding the point P inside the triangle
     -m2.Christer Ericson's book "Real-Time Collision Detection". faster, so I used this.
    Args:
        point: [u, v] or [x, y] 
        tri_points: three vertices(2d points) of a triangle. 2 coords x 3 vertices
    Returns:
        w0: weight of v0
        w1: weight of v1
        w2: weight of v3
     '''
    tp = tri_points
    # vectors
    v0 = tp[:,2] - tp[:,0]
    v1 = tp[:,1] - tp[:,0]
    v2 = point - tp[:,0]

    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    w0 = 1 - u - v
    w1 = v
    w2 = u

    return w0, w1, w2