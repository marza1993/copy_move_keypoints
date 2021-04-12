import cv2
import numpy as np
from skimage import io
from ctypes import *
from sys import platform
import matplotlib.pyplot as plt
import os


os.environ['PATH'] = os.environ['CUDA_MATRICES'] +\
                    os.pathsep + os.environ['OPENCV_CONTRIB_BIN'] + os.pathsep + os.environ['PATH']


DATA_SET_PATH = "D:\\dottorato\\copy_move\\MICC-F220\\"
nomeFile = "DSC_0812tamp1.jpg"

shared_lib_path = "..\\x64\\Release\\copy_move_detection_lib.dll"


img = cv2.imread(DATA_SET_PATH + nomeFile, cv2.IMREAD_GRAYSCALE)


class Point2dStruct(Structure):
    _fields_ = [
        ("x", c_float),
        ("y", c_float)
        ]


class KeyPointsMatchStruct(Structure):
    _fields_ = [
        ("p1", POINTER(Point2dStruct)),
        ("p2", POINTER(Point2dStruct)),
        ("descriptorsDistance", c_float),
        ("isValid", c_int)
        
        ]


# input img

# input params
input_img_data = img.astype(np.uint8).ctypes.data_as(POINTER(c_uint8))
img_W = c_uint(img.shape[1])
img_H = c_uint(img.shape[0])
soglia_sift = c_uint(0)
minPuntiIntorno = c_uint(3)
soglia_lowe = c_float(0.43)
eps = c_float(50)
soglia_desc_cluster = c_float(0.19)
getOutputImg = c_int(1)

# output params
resultForgedOrNot = c_int(-1)
foundMatches = pointer(KeyPointsMatchStruct())
output_img_data = pointer(c_uint8())
N_found_matches = c_uint(-1)
outImg_W = c_uint(-1)
outImg_H = c_uint(-1)



# carico la dll
try:
    copy_move_lib = CDLL(shared_lib_path)
    print("Successfully loaded ", copy_move_lib)
except Exception as e:
    print(e)

copy_move_lib.SIFT_copy_move_detection(input_img_data, img_W, img_H, soglia_sift, minPuntiIntorno, 
                                       soglia_lowe, eps, soglia_desc_cluster, byref(resultForgedOrNot), 
                                       byref(foundMatches), byref(N_found_matches), getOutputImg, 
                                       byref(output_img_data), byref(outImg_W), byref(outImg_H))



outputImg = np.ctypeslib.as_array(output_img_data, shape = (outImg_H.value, outImg_W.value, 3)).copy()



outputImg = cv2.cvtColor(outputImg, cv2.COLOR_BGR2RGB)
plt.imshow(outputImg)
plt.show()


print("l'immagine e' {}".format("forged" if resultForgedOrNot.value else "originale"))

#for i in range(N_found_matches.value):
#    print("match {}-> p1 = ({},{}), p2 = ({},{}), is valid: {}".format(i, foundMatches[i].p1.contents.x, 
#                                                                       foundMatches[i].p1.contents.y,
#                                                                       foundMatches[i].p2.contents.x, 
#                                                                       foundMatches[i].p2.contents.y,
#                                                                       ("yes" if foundMatches[i].isValid else "no")
#                                                                       ))


copy_move_lib.SIFT_copy_move_free_mem(byref(foundMatches), N_found_matches, byref(output_img_data));



















