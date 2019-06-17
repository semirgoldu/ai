import os
import sys
import json
from androidhelper.sl4a import Android
import glob
import time
import numpy as np
import urllib
import cv as cv2

import numpy as np
import cv as cv2
#from PIL import Image as img
#Initialize Android
droid = Android()
faceCascade = cv2.CascadeClassifier("/sdcard/cas.xml")
import subprocess
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
def detect_face(path,layout):
	image = cv2.imread(path)
	height, width = image.shape[:2]
	resized = cv2.resize(image, (width/5, height/5))
	faces = faceCascade.detectMultiScale(
    			resized,
    			scaleFactor=1.2,
    			minNeighbors=5,
    			minSize=(5,5),
    			flags = cv2.cv.CV_HAAR_SCALE_IMAGE
			)
	(x, y, w, h) = faces[0]
	rect=faces[0]
	draw_rectangle(resized, rect)
	draw_text(resized, "Face", rect[0], rect[1]-5)
	cv2.imwrite("/sdcard/face.jpg",resized)
	layout.views.preview.src="file:///sdcard/face.jpg"
	return True