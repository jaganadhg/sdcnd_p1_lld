#!/usr/bin/env python
"""
Author : Jaganadh Gopinadhan
e-mail : jaganadhg@gmail.com
Licence : MIT

Third Eye V1.0 Udacity Self Driving Car Nano Degree Project 1
Finding Lane Lines on the Road
"""

import glob

import matplotlib.image as mpimg
import numpy as np

import cv2

from lineutil import plot_img,LineDrawBase
from pputil import FrameTransformer

from pipeline import Pipeline

def save_img(img,fn):
    cv2.imwrite(img,fn)


def test_pipe_img(img):
    pipe = Pipeline() 
    frm = pipe.fit_frame(img)
    plot_img(frm)
    return frm

def test_pipe_vid(vid,outf):
    pipe = Pipeline() 
    ann_vid = pipe.fit_vid(vid)
    ann_vid.write_videofile(outf, audio=False)



if __name__ == "__main__":
    images = glob.glob("/Users/jagan/Documents/workspace/ThirdEye/thirdeye_v1.0/test_images/*.jpg")
    videos = glob.glob("/Users/jagan/Documents/workspace/ThirdEye/thirdeye_v1.0/test_videos/*.mp4")
    imout = "/Users/jagan/Documents/workspace/ThirdEye/thirdeye_v1.0/test_images_output/"
    vidout = "/Users/jagan/Documents/workspace/ThirdEye/thirdeye_v1.0/test_videos_output/"
    
    for img in images:
        print "Processing :", img
        imd = mpimg.imread(img)
        pimg = test_pipe_img(imd)
        fn = img.split("/")[-1]
        save_img(imout + fn , pimg)
        print "Saved :", img
    
    for vd in videos:
        print "Processing :", vd
        fn = vd.split("/")[-1]
        ofile = vidout + fn
        test_pipe_vid(vd,ofile)
        print "Processed :", vd
    
