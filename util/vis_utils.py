import math
import torch
import numpy as np
from scipy.spatial.transform import Rotation


def rotate(p, angle, o=[0, 0]):
    '''
    Rotate point p around point o with angle
    p: (x, y)
    o: (x, y)
    angle: degree
    H: height of image
    '''
    x = o[0] + math.cos(angle) * (p[0] - o[0]) - math.sin(angle) * (o[1] - p[1])
    y = o[1] - math.sin(angle) * (p[0] - o[0]) - math.cos(angle) * (o[1] - p[1])
    return x, y


def rotate_view(view, theta, origin=[0, 0]):
    # view: (4, 4)
    view1 = view.clone().numpy()
    R = view1[:3, :3] # rotation matrix

    rot_z = Rotation.from_rotvec(theta * np.array([0, 0, 1])).as_matrix()
    R = rot_z @ R
    view1[:3, :3] = R

    p = view1[:2, 3] # translation x, y
    p = rotate(p, -theta, origin)
    view1[:2, 3] = p
          
    return torch.from_numpy(view1)