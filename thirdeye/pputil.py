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


class FrameTransformer(object):
    """
    Image pre-processing module for the Lane Line Detector.
    Pre-processing techniques used:
    1) Gray Scaling 
    2) Canny Transform with automatic threshold detection
    3) Guassian Blur
    
    Parameters
    ----------
    threshold : float
        Blah
    """
    
    def __init__(self):
        self.kernel_size = 17
        self.sigma = 0.3
    
    
    def _togrey(self,img):
        """
        Convert image to grey scale.
        
        Parameters
        ----------
        img : numpy ndarry
            Image as numpy ndiamensional array.
            
        Returns
        -------
        gimg : numpy ndarray
            Gray scale image as numpy ndiamensional array
        """
        gimg = cv2.cvtColor(img,cv2.COLOR_RGBA2GRAY)
        
        return gimg
    
    def select_colors(self,img):
        """
        Select Yellow and white color from the image. The Lane lines are 
        either yellow or white. So converting only to greay scale will not
        yield desired result. To overcome this filter white and yellow pixels
        from the image.
        
        Parameters
        ---------
        img : np ndaaary
            Numpy ndiamensional array, which reasulted from reading the image
            using apy image reading api in Python.
        
        Returns
        -------
        ywimg : np ndarray
            Numy Ndiamensional Array. Image with white and yellow pixes.
        
        """
        hls_filtered = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
        white_pix = cv2.inRange(hls_filtered, np.uint8([20,200,0]), np.uint8([255,255,255]))
        yellow_pix = cv2.inRange(hls_filtered, np.uint8([10,50,100]), np.uint8([100,255,255]))
        combined_images = cv2.bitwise_or(white_pix, yellow_pix)
        ywimg = cv2.bitwise_and(img, img, mask = combined_images)
        
        return ywimg
    
    def _gaussian_blur(self,img):
        """
        Apply Gaussian Smoothing in the image
        
        Parameters
        ----------
        img : np ndarray
            Numpy ndiamensional array, which reasulted from reading the image
            using apy image reading api in Python.
        
        Returns
        -------
        gsimg : np ndarray
            Gaussain smoother image
        
        Notes
        -----
        TODO : Find the kernel size aut0matically
        """
        gsimg = cv2.GaussianBlur(img,(self.kernel_size, self.kernel_size),0)
        
        return gsimg
    
    def __get_canny_thresh_ott(self,img):
        """
        Find the otsu threshold from gray scale image. The threshold value can then used 
        for determing the low and high threshold in canny edge.
        Parameters
        ----------
        img : np ndarray
            Numpy ndiamensional array, which reasulted from reading the image
            using apy image reading api in Python.
        
        Returns
        -------
        (high_thresh,low_thresh) : tuple
            Tuple containing high threshold and low threshold for Canny
            (high_thresh,low_thresh)
        """
        
        grey_img = self._togrey(img)
        
        high_thresh, image_tresh = cv2.threshold(grey_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low_thresh = 0.5 * high_thresh
        
        return (high_thresh,low_thresh)
        
    
    def __get_canny_median_thresh(self,img):
        """
        Get Canny low and high threshold with image median pixels.
        Parameters
        ----------
        img : np ndarray
            Numpy ndiamensional array, which reasulted from reading the image
            using apy image reading api in Python.
        
        Returns
        -------
        (high_thresh,low_thresh) : tuple
            Tuple containing high threshold and low threshold for Canny
            (high_thresh,low_thresh)
        """
        grey_img = self._togrey(img)
        v = np.median(grey_img)
        low_thresh = int(max(0, (1.0 - self.sigma) * v))
        high_thresh = int(min(255, (1.0 + self.sigma) * v))
        
        return (high_thresh,low_thresh)
    
    def canny_transform(self,img):
        """
        Apply Canny edge detector in the image
        
        Parameters
        ----------
        img : np ndarray
            Numpy ndiamensional array, which reasulted from reading the image
            using apy image reading api in Python.
        
        Returns
        -------
        cimg : np ndarray
            Canny Transfomed image
        """
        thresholds_ott = self.__get_canny_thresh_ott(img)
        gb_img = self._gaussian_blur(img)
        cimg = cv2.Canny(gb_img, thresholds_ott[1], thresholds_ott[0])
        
        return cimg
    
    def get_roi(self,img,vertices):
        """
        Apply region of intrerest mask in an image based on supplied vertices.
        Parameters
        ----------
        img : np ndarray
            Numpy ndiamensional array, which reasulted from reading the image
            using apy image reading api in Python.
        
        vertices : np array
            Vertices of an image to mask
        
        Returns
        -------
        roi_img : np ndarray
            region_of_intetest masked image
        """
        
        image_mask = np.zeros_like(img)
        
        if len(img.shape) > 2:
            channels = img.shape[2]
            mask_ignore = (255,) * channels
        else:
            mask_ignore = 255
        
        cv2.fillPoly(image_mask, vertices, mask_ignore)

        roi_img = cv2.bitwise_and(img,image_mask)
        
        return roi_img
    
    def transform(self,img):
        """
        Perfro pre-processing in an image
        """
        yellow_white = self.select_colors(img)
        canny_edges = self.canny_transform(yellow_white)
        
        shape = img.shape
        
        vertices = np.array([[(100,shape[0]),(shape[1]*.45, shape[0]*0.6),\
        (shape[1]*.55, shape[0]*0.6), (shape[1],shape[0])]], dtype=np.int32)
        
        roi = self.get_roi(canny_edges,vertices)
        
        return roi

if __name__ == "__main__":
    print


