#!/usr/bin/env python
"""
Author : Jaganadh Gopinadhan
e-mail : jaganadhg@gmail.com
Licence : MIT

Third Eye V1.0 Udacity Self Driving Car Nano Degree Project 1
Finding Lane Lines on the Road
"""


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np

import cv2

from linedrawer import draw_lines_h

class LineDrawBase(object):
    """
    A Basic Lane Line drawing utility
    """
    
    def __init__(self):
        self.randn = 10
    
    def hough_transform(self,img,rho,theta,threshold,min_line_len,max_line_gap):
        """
        Apply Hough Transform in an image
        """
        
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        #line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        #self.draw_line(line_img, lines)
        return lines
    
    def draw_lines(self,img, lines, color=[0, 255, 0], thickness=2):
        """
        TODO : Improve here from the template
	THis function was adopted from Udacitys tenplate 
        """
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        return img
    
    def _get_line(self,line):
        [x1,y1,x2,y2] = line[0]
        
        m = (y2 - y1) / (x2 - x1)
        
        liner = lambda m,X: m * (X - x1) + y1
        
        lf = np.vectorize(liner)
        
        x = np.arange(x1,x2 + 1, dtype=np.uint16)
        y = lf(m,x).astype(np.uint16)
        
        return np.stack((x,y), axis = -1)
    
    
    def draw(self,img,roi,rho,theta,threshold,min_line_len,max_line_gap):
        """
        Draw the line over the image 
        
        Parameters
        ----------
        img : np ndarry
            Original image as numpy array
        rho : int
        theta : float
        threshold : int
        min_line_len : int
        max_line_len : int
        
        Returns
        -------
        line_imge : np ndarry
            Image with line annotated
        """
        hough_lines = self.hough_transform(roi,rho,theta,threshold,min_line_len,max_line_gap)
        image_with_line = draw_lines_h(img,hough_lines)
        image_with_line_weighted = self.weighted_image(image_with_line,img)
        return image_with_line_weighted

    def weighted_image(self,lanes,img, alpha = .9 , beta = 0.95 , lbda = 0.):
        """
	This function was adopted from the UDacity template code
        """
        temp_img = np.copy(img)*0
        return cv2.addWeighted(temp_img, alpha, lanes, beta, lbda)


def plot_img(image):
    """
    Utility to plot image
    """
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    base = LineDrawBase()
