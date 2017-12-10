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
import random
import time


CAR_WORDS = ['minivan', 'sports car', 'car,', 'cab', 'taxi', 'convertible', 'limo',
             'jeep', 'landrover', 'R.V.', 'go-kart', 'dustcart', 'pickup',
             'snowplow', 'cassette player', 'Model T', 'ricksha', 'rickshaw']
             # 'moving van', #This one is iffy. sometimes random stuff is moving vans


# classes = ['Unknown', 'Compacts', 'Sedans', 'SUVs', 'Coupes',
#            'Muscle', 'SportsClassics', 'Sports', 'Super', 'Motorcycles',
#            'OffRoad', 'Industrial', 'Utility', 'Vans', 'Cycles',
#            'Boats', 'Helicopters', 'Planes', 'Service', 'Emergency',
#            'Military', 'Commercial', 'Trains']


files = glob('../rob599_dataset_deploy/test/*/*_image.jpg')
files.sort()
classify_image.create_graph()
fig1 = plt.figure(1, figsize=(16, 9))
fig3 = plt.figure(3, figsize=(10,10))

with open('./outfile.txt','w') as f:
    f.write('guid/image,N\n')

plt.ion()

for i in range(len(files)):
    if i%1 == 0:
        print("Trial ", i, 'out of ', len(files))
        
    imgpath = files[i]
    img = plt.imread(imgpath)

    fig1.clear()
    fig3.clear()

    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.imshow(img)
    


    xyz = np.memmap(imgpath.replace('_image.jpg', '_cloud.bin'), dtype=np.float32)
    xyz.resize([3, xyz.size // 3])
    xyz = np.array(xyz).transpose()

    proj = np.memmap(imgpath.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
    proj.resize([3, proj.size // 3])



    imgs_of_interest = parse_lidar.get_potential_car_images(img, xyz, np.array(proj))
    # clusters = parse_lidar.get_points_of_interest(xyz)
    # inliers = parse_lidar.lidar_mask(xyz)
    # line_mask = parse_lidar.mask_out_long_smooth_lines(xyz)

    car_count = 0
    num_fig = len(imgs_of_interest)

    for i in range(num_fig):
        tmp_imgpath = '/tmp/img.jpg'
        imageio.imwrite(tmp_imgpath, imgs_of_interest[i])
        label = classify_image.run_inference_on_image_path(tmp_imgpath)

        for word in CAR_WORDS:
            if label.find(word) >= 0:
                car_count += 1
                break


        ax = fig3.add_subplot(np.ceil(np.sqrt(num_fig)),np.ceil(np.sqrt(num_fig)),i+1)
        ax.imshow(imgs_of_interest[i])
        # label = classifier.classify(imgs_of_interest[i])
        
        for word in CAR_WORDS:
            if label.find(word) >= 0:
                found = True
                label = word.capitalize()
                break
        
        ax.set_title(label)
        ax.set_axis_off()
    plt.show()
    plt.pause(0.1)


    with open('./outfile.txt','a') as f:
        f.write(imgpath[30:-10] + ',' + str(car_count) + '\n')



