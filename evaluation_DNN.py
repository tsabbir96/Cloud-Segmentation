# This provides the results of Table I in the paper.
import os, os.path
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import imageio
import cv2

from keras.models import load_model

import sys
sys.path.insert(0, './scripts/')

from score_card import *
from roc_items import *

def show(show_img):
    cv2.imshow("Image", show_img) 
    cv2.waitKey(0)

# This discusses the results of combined dataset
#RESULT_FOLDER = './results/combined/'

# This discusses the results of balanced ramdom sample
RESULT_FOLDER = './results/withAUG_dataset/'
RESULT_FOLDER2 = './results/balanced_random_sample/'
print("loading model...")
ae = load_model(RESULT_FOLDER + 'cloudsegnet.hdf5')



X_testing = np.load(RESULT_FOLDER + 'xtesting.npy')
print ('from the saved data')
print (X_testing.shape)


Y_testing = np.load(RESULT_FOLDER + 'ytesting.npy')
print ('from the saved data')
print (Y_testing.shape)

## Evaluate - all time images
(no_of_testing_images, _, _, _) = X_testing.shape

precision_array = []
recall_array = []
fscore_array = []
error_array = []

threshold_value = 0.5

(no_of_testing_images, _, _, _) = X_testing.shape

for sample_iter in range(no_of_testing_images):
    
    image_array = X_testing[sample_iter]
    gt_array = Y_testing[sample_iter]
    gt_array = np.squeeze(gt_array)
    save_img_name = RESULT_FOLDER2 + 'testing_images/day/img/' + str(sample_iter) + '.png'


    plt.imsave(save_img_name, cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    print(" Orig: ", save_img_name)
    save_gt_name = RESULT_FOLDER2 + 'testing_images/day/GT/img' + str(sample_iter) + '_GT.png'
    plt.imsave(save_gt_name, gt_array, cmap=cm.gray)
    print(" GT: ", save_gt_name)

    gt_image = Y_testing[sample_iter]
    gt_image = np.squeeze(gt_image)
    input_image = X_testing[sample_iter]
    image_map = calculate_map(input_image, ae)
    segmented = RESULT_FOLDER2 + 'testing_images/day/segmented/' + str(sample_iter) + '.png'
    imageio.imwrite(segmented, image_map)
    print("image shape: ",image_map.shape)
    print("image type: ", type(image_map))
    print("image map: \n\n",image_map)
    print("\n\n")

    #thr, mask = cv2.threshold(image_map, 0.1, 255.0, cv2.THRESH_BINARY);
    #print("mask: ", mask.shape)
    #mask = mask.astype(int)

    #mask = cv2.resize(mask, (300,300))
    #print(image_array.shape)
    #result = cv2.bitwise_or(image_array, image_array, mask=mask)
    #show(result)
DATAPATH_1 = RESULT_FOLDER2 + 'testing_images/day/segmented/' # DataSet path for method1 sick images. "1" is for sick
DATAPATH_2 = RESULT_FOLDER2 + 'testing_images/day/img/'
DATAPATH_3 = RESULT_FOLDER2 + 'testing_images/day/final/'
filenames = os.listdir(DATAPATH_1)

for filename in filenames:
    mask  = cv2.imread(DATAPATH_1+str(filename)) #read image
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    thr, mask = cv2.threshold(gray,100.0, 255.0, cv2.THRESH_BINARY)
    #kernel of 5x5 matrix of ones
    #kernelOpen=np.ones((2,2))
    #kernel of 20x20 matrix of ones
    #kernelClose=np.ones((20,20))
    #maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    #maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
    #maskFinal=maskOpen
    #kernel = np.ones((2,2),np.uint8)
    #erosion = cv2.erode(maskFinal,kernel,iterations = 3)
    img  = cv2.imread(DATAPATH_2+str(filename))
    #print(DATAPATH_2+str(filename))	#read image
    result = cv2.bitwise_or(img, img, mask = mask)
    plt.imsave(DATAPATH_3+str(filename), cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
