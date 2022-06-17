'''
author: Pruthviraj Patil
version: 1.0

'''

import numpy as np
import cv2
from Hog import hog_feature_generator

class image_utility:
	def __init__(self):
		self.Hog_feats=hog_feature_generator()

	def convert_rgb_gray(self, images):
		#status: complete
		grays=[]
		for img in images:
			frame = 0.114 * img[:,:,0] + 0.587 * img[:,:,1] + 0.299 * img[:,:,2]
			grays.append(frame)
		return grays

	
	def generate_hog(self, image):
		#status: complete 
		features=self.Hog_feats.generate_feats(image, (8, 8), 2)
		return features
