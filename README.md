# Perception

## Setup on linux
    cd ~
    virtualenv -p python3 tf3
    source ~/tf3/bin/activate
    sudo apt install python3-tk
    pip install tensorflow matplotlib pillow scikit-learn imageio


Download the training data and place in a folder at the same level as the Perception folder (not in this git repo - that would be too large)
Run `python ./classify_image.py` to download the relevant tensorflow files

Run `python ./count_test` to start computing labels for the test data. This will write to outfile.txt.
To view this file in real time: `tail -f outfile.txt`

To play around with the training data, run `python ./tmp`. Using this file for testing random changes.


## Overview
The basic algorithms is:
1. Filter lidar data to remove large planes and uninteresting data
2. Segment lidar data into clusters of interest
3. Get corresponding images of interest
4. use tensorflow to classify these images of interest, counting the number of cars
    
    