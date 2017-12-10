#! /usr/bin/python3

from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import IPython


def get_lidar(imagepath):
    xyz = np.array(np.memmap(imagepath.replace('_image.jpg', '_cloud.bin'), dtype=np.float32))
    xyz.resize([3, xyz.size // 3])
    return xyz.transpose()

def get_camera_projection(imagepath):
    proj = np.array(np.memmap(imagepath.replace('_image.jpg', '_proj.bin'), dtype=np.float32))
    proj.resize([3, proj.size // 3])
    return proj

def get_bboxes(imagepath):
    
    try:
        #Convert to array as soon as possible. Otherwise potential data corruption
        bboxes = np.array(np.memmap(imagepath.replace('_image.jpg', '_bbox.bin'),
                                    dtype=np.float32))

    except:
        print('[*] bbox not found.')
        bboxes = np.array([], dtype=np.float32)
    bboxes.resize([bboxes.size // 11, 11])
    return bboxes
    


def rot(n, theta):
    n = n / np.linalg.norm(n, 2)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K



def unpack_bbox(bbox):
    n = bbox[0:3]
    theta = np.linalg.norm(n)
    n /= theta
    R = rot(n, theta)
    t = bbox[3:6]

    # size of the bbox
    sz = bbox[6:9]
    vert_3D, edges = get_bbox(-sz / 2, sz / 2)
    vert_3D = R @ vert_3D + t[:, np.newaxis]
    return vert_3D, edges, t


def get_bbox(p0, p1):
    '''
    Input:
    *   p0, p1
        (3)
        Corners of a bounding box represented in the body frame.

    Output:
    *   v
        (3, 8)
        Vertices of the bounding box represented in the body frame.
    *   e
        (2, 14)
        Edges of the bounding box. The first 2 edges indicate the `front` side
        of the box.
    '''
    v = np.array([[p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
                  [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
                  [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]])
    e = np.array([[2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],
                  [7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]], dtype=np.uint8)

    return v, e

# def in_bbox
