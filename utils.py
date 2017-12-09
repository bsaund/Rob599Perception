#! /usr/bin/python3

from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def getRandomImagePath():
    files = glob('deploy/*/*/*_image.jpg')
    idx = np.random.randint(0, len(files))
    return files[idx]

def getLidar(imagepath):
    xyz = np.memmap(imagepath.replace('_image.jpg', '_cloud.bin'), dtype=np.float32)
    xyz.resize([3, xyz.size // 3])
    return xyz

def getCameraProjection(imagepath):
    proj = np.memmap(snapshot.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
    proj.resize([3, proj.size // 3])
    return proj



