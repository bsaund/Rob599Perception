#! /usr/bin/python3


## Generates labeled images using the lidar segementation and ground truth bounding boxes


from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import IPython
import parse_lidar
import utils
import csv
import classify_image
import imageio
import random
import time
import os


CAR_WORDS = ['minivan', 'sports car', 'car,', 'cab', 'taxi', 'convertible', 'limo',
             'jeep', 'landrover', 'R.V.', 'go-kart', 'dustcart', 'pickup',
             'snowplow', 'cassette player', 'Model T', 'ricksha', 'rickshaw']
             # 'moving van', #This one is iffy. sometimes random stuff is moving vans


# classes = ['Unknown', 'Compacts', 'Sedans', 'SUVs', 'Coupes',
#            'Muscle', 'SportsClassics', 'Sports', 'Super', 'Motorcycles',
#            'OffRoad', 'Industrial', 'Utility', 'Vans', 'Cycles',
#            'Boats', 'Helicopters', 'Planes', 'Service', 'Emergency',
#            'Military', 'Commercial', 'Trains']

SAVE_DIRECTORY = os.path.join('..','labeled_training_data')

def get_labeled_outfile(image_path, i):
    """
    Returns the path where the image should be written
    """
    segments = os.path.normpath(imgpath).split(os.sep)
    img_number = segments[-1][0:4]
    outdir = os.path.join(SAVE_DIRECTORY, segments[-2], img_number)
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, str(i) + '.jpg')
    return outfile

def classify_using_bounding_box(lidar, bbox):
    """
    Returns the class of the img based on the bounding box
    """
    IPython.embed()



files = glob('../rob599_dataset_deploy/trainval/*/*_image.jpg')
files.sort()
classify_image.create_graph()
fig1 = plt.figure(1, figsize=(16, 9))
fig3 = plt.figure(3, figsize=(10,10))


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

    try:
        bbox = np.memmap(imgpath.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
    except:
        print('[*] bbox not found.')
        bbox = np.array([], dtype=np.float32)
    bbox.resize([bbox.size // 11, 11])




    imgs_of_interest, pois = parse_lidar.get_imgs_and_clusters(img, xyz, np.array(proj))


    car_count = 0
    num_fig = len(imgs_of_interest)

    for i in range(num_fig):
        saved_imgpath = get_labeled_outfile(imgpath, i)
        imageio.imwrite(saved_imgpath, imgs_of_interest[i])
        # label = classify_image.run_inference_on_image_path(saved_imgpath)
        label = classify_using_bounding_box(pois[i], bbox)

        ax = fig3.add_subplot(np.ceil(np.sqrt(num_fig)),np.ceil(np.sqrt(num_fig)),i+1)
        ax.imshow(imgs_of_interest[i])
        # label = classifier.classify(imgs_of_interest[i])
        label = classify_using_bounding_box(pois[i], bbox)
        
        # for word in CAR_WORDS:
        #     if label.find(word) >= 0:
        #         found = True
        #         label = word.capitalize()
        #         break
        
        # ax.set_title(label)
        ax.set_axis_off()
    plt.show()
    plt.pause(0.1)
    IPython.embed()


    # with open('./outfile.txt','a') as f:
    #     f.write(imgpath[30:-10] + ',' + str(car_count) + '\n')



