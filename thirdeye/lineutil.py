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
    
    def line_fitting(self,base_img,line,grey):
        
        region_tofit = 325
        
        tf_img = np.array([],dtype=np.uint16).reshape(0,2) #cnt
        img_skleton = np.zeros_like(grey) #grey till canny
        
        for ll in line:
            [x1,y1,x2,y2] = ll[0]
            cv2.line(img_skleton,(x1,y1),(x2,y2),255,2)
            line_points = self._get_line(ll)
            
            tf_img = np.vstack((tf_img,line_points))
        
        [vx,vy,x,y] = cv2.fitLine(tf_img,cv2.DIST_L2,0,0.01,0.01)
        
        max_x = int(((region_tofit - y) * vx/vy ) + x)
        min_x = int(((grey.shape[0] - y) * vx / vy) + x) 
        
        my_imge = np.zeros_like(grey)
        
        #cv2.line(my_imge,(min_x,grey.shape[0] - 1),(max_x,region_tofit),255,2)
        cv2.line(my_imge,(vy,vx),(x,y),255,2)
        return my_imge
        
    
    
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
        #draw_lines_h
        #image_with_line = self.draw_lines(img,hough_lines)
        image_with_line_weighted = self.weighted_image(image_with_line,img)
        return image_with_line_weighted

    def weighted_image(self,lanes,img, alpha = .9 , beta = 0.95 , lbda = 0.):
        """
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