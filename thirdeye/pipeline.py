#!/usr/bin/env python
"""
Author : Jaganadh Gopinadhan
e-mail : jaganadhg@gmail.com
Licence : MIT

Third Eye V1.0 Udacity Self Driving Car Nano Degree Project 1
Finding Lane Lines on the Road
"""


import matplotlib.image as mpimg
import numpy as np

from moviepy.editor import VideoFileClip

from pputil import FrameTransformer
from lineutil import LineDrawBase, plot_img

class Pipeline(object):
    """
    Basic pipeline for Lane Line detection
    TODO : Improve with more features
    """
    
    def __init__(self):
        self.rho = 1
        self.theta = np.pi/180
        self.threshold = 15
        self.min_line_len = 25
        self.max_line_gap = 250
        self.preprocessor = FrameTransformer()
        self.annotator = LineDrawBase()
    
    def fit_frame(self,image):
        """
        Preprocess and draw image
        """
        roi = self.preprocessor.transform(image)
        annotated = self.annotator.draw(image,roi,self.rho,self.theta,\
        self.threshold, self.min_line_len,self.max_line_gap)
        return annotated
    
    def fit_vid(self,vidfile):
        """
        Process video file
        """
        vf = VideoFileClip(vidfile)
        white_clip = vf.fl_image(self.fit_frame)
        
        return white_clip
    


if __name__ == "__main__":
    print 