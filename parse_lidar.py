#! /usr/bin/python3

import utils
import numpy as np
from numpy.linalg import norm
import cProfile
from sklearn import linear_model, datasets
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
import networkx
from networkx.algorithms.components.connected import connected_components

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

def diag_len(cluster):
    return np.linalg.norm(np.max(cluster, axis=0) - np.min(cluster,axis=0))

def mask_out_long_lines(xyz):
    print('starting line filtering')
    line_mask = [False]*xyz.shape[0]
    inds_in_a_line = []
    d_max_acceptable = 0.1
    for point_ind in range(xyz.shape[0]):
        point = xyz[point_ind,:]
        if len(inds_in_a_line) <= 2:
            inds_in_a_line.append(point_ind)
            continue
        d_max = 0
        p1 = xyz[inds_in_a_line[0],:]
        pe = xyz[inds_in_a_line[-1],:]
        print(len(inds_in_a_line))
        for i in range(1, len(inds_in_a_line)-1):
            d_max = max(d_max, norm(np.cross(pe-p1, p1-xyz[inds_in_a_line[i],:]))/norm(pe-p1))
        if d_max > d_max_acceptable:
            if len(inds_in_a_line) > 2 and norm(pe-p1) > 5:
                for ind in inds_in_a_line:
                    line_mask[ind] = True
            inds_in_a_line = []
            continue
        inds_in_a_line.append(point_ind)
    print('ending line filtering')
    return np.logical_not(line_mask)

def mask_largest_plane(xyz):
    """
    Returns a np.array of {True,False} with size of xyz.shape[0]
    This is a mask, where True are points near the largest plane
    """
    ransac = linear_model.RANSACRegressor(PlaneModel(),
                                          max_trials=100,
                                          residual_threshold=.2)

    dummy = np.zeros(xyz.shape[0])

    ransac.fit(xyz, dummy)
    return ransac.inlier_mask_


def mask_out_large_planes_complicated(xyz):
    """
    Returns a np.array of {True,False} with size of lidar.shape[0]
    This is a mask for all sufficiently large planes in the image
    """
    not_plane = np.array([True] * xyz.shape[0])
    
    new_plane_ind = mask_largest_plane(xyz)

    def large_enough_connected_plane(plane_points, req_num, req_diag):
        s = euclid_segmentation(plane_points, 0.1, req_num)
        # IPython.embed()
        if len(s) == 0:
            return False
        max_region = list(max(s, key = len))
        return diag_len(plane_points[max_region,:]) > req_diag

    count = 0
    # while np.sum(new_plane_ind)>1000 and diag_len(xyz[not_plane,:][new_plane_ind,:]) > 10:
    while large_enough_connected_plane(xyz[not_plane,:][new_plane_ind,:], 500, 5):
        count += 1
        not_plane[np.where(not_plane==True)] = np.logical_not(new_plane_ind)
        new_plane_ind = mask_largest_plane(xyz[not_plane,:])

        

    print("Found ", count, "Planes")
    return not_plane
    # return np.logical_not(mask_largest_plane(xyz))

def mask_out_large_planes(xyz):
    """ Masks only the largest plane)"""
    return np.logical_not(mask_largest_plane(xyz))

def mask_far_points(lidar):
    return np.linalg.norm(lidar, axis=1) < 60


def lidar_mask(lidar):
    
    not_ground = mask_out_large_planes(lidar)
    near = mask_far_points(lidar)
    return np.logical_and(not_ground, near)


# def kmeans_cluster(masked_lidar):
#     kmeans = KMeans(n_clusters=50)
#     kmeans.fit(masked_lidar)
#     labels = kmeans.labels_
#     return kmeans.cluster_centers_
#     # IPython.embed()

# def get_points_around_cluster(centers, inliers):
#     # IPython.embed()
#     clusters = []
#     for center in centers:
#         # IPython.embed()
#         cm = np.linalg.norm(inliers-center,axis=1) < 4.0
#         clusters.append(inliers[cm])

#     return clusters

# def get_cluster_centers(raw_lidar):


def to_edges(nodes):
    """
    Return sufficient edges to fully connect nodes
    to_edges(['a','b','c','d']) -> [(a,b), (b,c), (c,d)]
    """
    it = iter(nodes)
    last = next(it)

    for current in it:
        yield last, current
        last = current  
    
    
def euclid_segmentation(point_cloud, dist, min_size):
    """
    Segments point cloud into clusters where each cluster is at least dist apart
    Ignores clusters smaller than min_size
    """
    kdtree = KDTree(point_cloud)
    G = networkx.Graph()
    for point in point_cloud:
        near = kdtree.query_radius([point], dist)[0]
        G.add_nodes_from(near)
        G.add_edges_from(to_edges(near))

    clusters = [cc for cc in connected_components(G) if len(cc)>min_size]
    return clusters

def car_clusters(point_cloud):
    clusters_ind = euclid_segmentation(point_cloud, dist=.3, min_size=20)
    all_clusters = [point_cloud[list(inds),:] for inds in clusters_ind]
    # for cluster in all_clusters:
    #     upper = np.max(cluster, axis=0)
    #     lower = np.min(cluster, axis=0)
        
    #     IPython.embed()
    
    right_sized_clusters = [c for c in all_clusters if
                            diag_len(c) < 6 and diag_len(c) > 0.5]
    return right_sized_clusters
                            

def get_points_of_interest(raw_lidar):
    inlier_mask = lidar_mask(raw_lidar)
    inliers = raw_lidar[inlier_mask,:]
    # centers = kmeans_cluster(inliers)
    # pois = get_points_around_cluster(centers, inliers)
    # clusters_ind = car_clusters(point_cloud

    # for inds in clusters_ind:
    #     IPython.embed()
    pois = car_clusters(inliers)
    # IPython.embed()
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
