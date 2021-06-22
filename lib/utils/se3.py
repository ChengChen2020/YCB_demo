import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat, qmult, qinverse
from transforms3d.euler import quat2euler, mat2euler, euler2quat

def egocentric2allocentric(qt, T):
    dx = np.arctan2(T[0], -T[2])
    dy = np.arctan2(T[1], -T[2])
    quat = euler2quat(-dy, -dx, 0, axes='sxyz')
    quat = qmult(qinverse(quat), qt)
    return quat

    