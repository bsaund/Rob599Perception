#! /usr/bin/python3
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import IPython
import parse_lidar
import csv
import classify_image
import imageio


CAR_WORDS = ['minivan', 'sports car', 'car,', 'cab', 'taxi', 'convertible', 'limo',
             'jeep', 'landrover', 'R.V.']

def rot(n, theta):
    n = n / np.linalg.norm(n, 2)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


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


classes = ['Unknown', 'Compacts', 'Sedans', 'SUVs', 'Coupes',
           'Muscle', 'SportsClassics', 'Sports', 'Super', 'Motorcycles',
           'OffRoad', 'Industrial', 'Utility', 'Vans', 'Cycles',
           'Boats', 'Helicopters', 'Planes', 'Service', 'Emergency',
           'Military', 'Commercial', 'Trains']

# files = glob('deploy/*/*/*_image.jpg')
# files = glob('../rob599_dataset_deploy/trainval/*/*_image.jpg')
files = glob('../rob599_dataset_deploy/test/0815cc1e-9a0c-4875-a5ca-784ef1a32bba/0008_image.jpg')
# files = glob('../rob599_dataset_deploy/trainval/e95739d4-4eeb-4087-b22f-851964073287/0010_image.jpg')
# files = glob('../rob599_dataset_deploy/trainval/aaa5a89c-d03d-494d-b72e-afd02734e73b/0027_image.jpg')


num_cars = {}
with open('../rob599_dataset_deploy/trainval/num_cars.csv') as true_cars_file:
    reader = csv.reader(true_cars_file, delimiter=',')
    next(reader)
    for row in reader:
        num_cars[row[0]] = int(row[1])

# IPython.embed()

idx = np.random.randint(0, len(files))
snapshot = files[idx]
print(snapshot)

img = plt.imread(snapshot)

xyz = np.memmap(snapshot.replace('_image.jpg', '_cloud.bin'), dtype=np.float32)
xyz.resize([3, xyz.size // 3])
xyz = np.array(xyz).transpose()

proj = np.memmap(snapshot.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
proj.resize([3, proj.size // 3])


try:
    bbox = np.memmap(snapshot.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
except:
    print('[*] bbox not found.')
    bbox = np.array([], dtype=np.float32)
bbox.resize([bbox.size // 11, 11])

uv = proj @ np.vstack([xyz.transpose(), np.ones_like(xyz[:,0])])
uv = uv / uv[2, :]

# classifier = classify.Classifier()

# centers = parse_lidar.get_points_of_interest(xyz)
imgs_of_interest = parse_lidar.get_potential_car_images(img, xyz, np.array(proj))
clusters = parse_lidar.get_points_of_interest(xyz)
inliers = parse_lidar.lidar_mask(xyz)
line_mask = parse_lidar.mask_out_long_smooth_lines(xyz)
# IPython.embed()
# uv = uv[:, inliers]




clr = np.linalg.norm(xyz, axis=1)
fig1 = plt.figure(1, figsize=(16, 9))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.imshow(img)
ax1.scatter(uv[0, inliers], uv[1, inliers], c=clr[inliers], marker='.', s=1)
# ax1.scatter(uv[0, :], uv[1, :], marker='.', s=1)
# ax1.scatter(uv[0, line_mask], uv[1, line_mask], marker='.', s=2)
ax1.axis('scaled')
fig1.tight_layout()



fig2 = plt.figure(2, figsize=(8, 8))
ax2 = Axes3D(fig2)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')

classify_image.create_graph()

car_count = 0


fig3 = plt.figure(3, figsize=(10,10))
num_fig = len(imgs_of_interest)
for i in range(num_fig):
    ax = fig3.add_subplot(np.ceil(np.sqrt(num_fig)),np.ceil(np.sqrt(num_fig)),i+1)
    ax.imshow(imgs_of_interest[i])
    # label = classifier.classify(imgs_of_interest[i])
    imgpath = '/tmp/img.jpg'
    imageio.imwrite(imgpath, imgs_of_interest[i])

    # IPython.embed()
    
    label = classify_image.run_inference_on_image_path(imgpath)
    found = False
    for word in CAR_WORDS:
        if label.find(word) >= 0:
            found = True
            label = word.capitalize()
            car_count += 1
            break
    # IPython.embed()
    ax.set_title(label)
    ax.set_axis_off()


# IPython.embed()
# ax2.scatter(centers[:,0], centers[:,1], centers[:,2], marker='x', s=100)

step = 5
# ax2.scatter(xyz[0, ::step], xyz[1, ::step], xyz[2, ::step], \
#     c=clr[::step], marker='.', s=1)
# ax2.scatter(xyz[0, inliers], xyz[1, inliers], xyz[2, inliers], marker='.', s=1)
print(len(clusters), "clusters found")
for cluster in clusters:
    # IPython.embed()
    ax2.scatter(cluster[:,0], cluster[:,1], cluster[:,2], marker='.', s=1)

colors = ['C{:d}'.format(i) for i in range(10)]
for k, b in enumerate(bbox):
    n = b[0:3]
    theta = np.linalg.norm(n)
    n /= theta
    R = rot(n, theta)
    t = b[3:6]

    # size of the bbox
    sz = b[6:9]
    vert_3D, edges = get_bbox(-sz / 2, sz / 2)
    vert_3D = R @ vert_3D + t[:, np.newaxis]

    vert_2D = proj @ np.vstack([vert_3D, np.ones(8)])
    vert_2D = vert_2D / vert_2D[2, :]

    clr = colors[k % len(colors)]
    for e in edges.T:
        ax1.plot(vert_2D[0, e], vert_2D[1, e], color=clr)
        ax2.plot(vert_3D[0, e], vert_3D[1, e], vert_3D[2, e], color=clr)

    c = classes[int(b[9])]
    ignore_in_eval = bool(b[10])
    if ignore_in_eval:
        ax2.text(t[0], t[1], t[2], c, color='r')
    else:
        ax2.text(t[0], t[1], t[2], c)

ax2.auto_scale_xyz([-40, 40], [-40, 40], [0, 80])
ax2.view_init(elev=-30, azim=-90)

for e in np.identity(3):
    ax2.plot([0, e[0]], [0, e[1]], [0, e[2]], color=e)

id = snapshot[0:-10]
id = id[34:]

try:
    print('ground truth: ', num_cars[id])
except:
    pass

    
print('we count:     ', car_count)
plt.show()


