#! /usr/bin/python3

import utils
import numpy as np
import cProfile
from sklearn import linear_model, datasets
from sklearn.cluster import KMeans

import IPython

class PlaneModel:
    def __init__(self, P=None):
        self.P = P
        
    def fit(self, X, y):
        self.P = X

    def predict(self, X):

        # print("predicting")
        P = self.P

        n = np.cross(P[1,:] - P[0,:], P[2,:] - P[1,:])
        n = n/np.linalg.norm(n)
        d = np.matmul((X-P[1,:]),n)
        return np.array(d)

    def score(self, X, y):
        return np.sum(np.abs(y))

    def get_params(self, deep=True):
        return {'P': self.P}

    def set_params(self, P=None, random_state=None):
        if P is not None:
            self.P = P

def mask_large_planes(lidar):
    ransac = linear_model.RANSACRegressor(PlaneModel(),
                                          max_trials=100,
                                          residual_threshold=.2)
    xyz = np.array(lidar).transpose()
    dummy = np.zeros(xyz.shape[0])

    ransac.fit(xyz, dummy)
    # print("fit complete")
    return np.logical_not(ransac.inlier_mask_)


def mask_far_points(lidar):
    xyz = np.array(lidar)
    return np.linalg.norm(xyz, axis=0) < 60


def lidar_mask(lidar):
    not_ground = mask_large_planes(lidar)
    near = mask_far_points(lidar)
    return np.logical_and(not_ground, near)


def cluster(masked_lidar):
    kmeans = KMeans(n_clusters=50)
    kmeans.fit(masked_lidar)
    labels = kmeans.labels_
    return kmeans.cluster_centers_
    # IPython.embed()

def get_points_around_cluster(centers, inliers):
    # IPython.embed()
    clusters = []
    for center in centers:
        # IPython.embed()
        cm = np.linalg.norm(inliers-center,axis=1) < 4.0
        clusters.append(inliers[cm])

    return clusters

# def get_cluster_centers(raw_lidar):


def get_points_of_interest(raw_lidar):
    inlier_mask = lidar_mask(raw_lidar)
    inliers = raw_lidar[:,inlier_mask].transpose()
    centers = cluster(inliers)
    pois = get_points_around_cluster(centers, inliers)
    return pois

def extract_image(image, corners):
    ll, ur = corners
    lower, left = ll
    upper, right = ur
    # IPython.embed()
    return image[left:right, lower:upper, :]
    

def extract_images(image, corners_list):
    images = []
    for corners in corners_list:
        images.append(extract_image(image,corners))
        # IPython.embed()
    return images


def get_potential_car_images(image, raw_lidar, proj):
    pois = get_points_of_interest(raw_lidar)
    corners = []
    for poi in pois:
        poih = np.append(poi,np.ones([poi.shape[0],1]),axis=1).transpose()
        camerah = np.matmul(proj, poih).transpose()
        camerah = camerah/np.repeat(camerah[:,2:3],3,axis=1)

        ll = (np.min(camerah[:,0:2],axis=0)).astype(int)
        ur = (np.max(camerah[:,0:2],axis=0)).astype(int)
        corners.append((ll,ur))
    return extract_images(image, corners)
    
    

if __name__ == "__main__":
    img = utils.getRandomImagePath()
    xyz = utils.getLidar(img)
    mask_large_planes(xyz)
