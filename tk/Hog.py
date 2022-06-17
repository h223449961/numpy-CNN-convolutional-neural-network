## 
'''
author: Pruthviraj Patil
version: 1.0

'''

import numpy as np
import cv2
from canny import Canny_edge_detector

class hog_feature_generator:
	def __init__(self):
		self.gx=None
		self.gy=None
		self.magnitudes=None
		self.directions=None
		self.image_into_pos=None
		self.hog_descriptor=None
		self.canny_util=Canny_edge_detector()

	def generate_grads(self, image):
		#status: complete
		self.gx, self.gy=self.canny_util.grad_from_image(image)

	def generate_directions(self):
		#status: complete
		self.directions=self.canny_util.dir_from_grads(self.gx, self.gy)

	def generate_magnitudes(self):
		#status: complete
		self.magnitudes=self.canny_util.mag_from_grads(self.gx, self.gy)

	def histogram_for_position(self, m_matrix, d_matrix):
		#status: complete 
		##############start here
		histogram=np.zeros(9)
		x_len=m_matrix.shape[0]
		y_len=m_matrix.shape[1]

		for i in range(x_len):
			for j in range(y_len):
				current_mag=m_matrix[i][j]
				current_dir=d_matrix[i][j]
				percentage=(current_dir%20)/20

				hist_position=int(current_mag/20)-1
				if(hist_position>=8):
					hist_position_next=0
				else:
					hist_position_next=hist_position+1

				histogram[hist_position]+=((1-percentage)*current_mag)
				histogram[hist_position_next]+=(percentage*current_mag)

		return(histogram)

	def generate_feats(self, image, hist_window_dims, norm_window_size):
		#status: complete
	#steps as said in README

	#step 2 : generate grads, directions, magnitudes
		self.generate_grads(image)
		self.generate_directions()
		self.generate_magnitudes()

	#step 3 : defining positions
		x_dim=hist_window_dims[0]
		y_dim=hist_window_dims[1]
		x, y=self.magnitudes.shape
		
		self.image_into_pos=np.zeros((int(x/x_dim), int(y/y_dim), 9))#9 histogram valued array in each position.

		for i in range(int(x/x_dim)):
			for j in range(int(y/y_dim)):
				histogram_array=self.histogram_for_position(self.magnitudes[i*x_dim: i*x_dim+x_dim, j*y_dim: j*y_dim+y_dim], self.directions[i*x_dim: i*x_dim+x_dim, j*y_dim: j*y_dim+y_dim])
				self.image_into_pos[i][j]=histogram_array

	#step 4 : 16*16 block normalization and concatenation
		x_len, y_len, n_hists_per_window=self.image_into_pos.shape
		
		norm_x_len= int(x_len-norm_window_size/2)
		norm_y_len= int(y_len-norm_window_size/2)

		self.hog_descriptor=np.zeros((norm_x_len, norm_y_len, 36))#cuz size gets reduced according to the window size (as window has to use 9*4 histograms)


		for i in range(norm_x_len):
			for j in range(norm_y_len):
				hist_concat_pos=np.concatenate((self.image_into_pos[i][j], self.image_into_pos[i+1][j], self.image_into_pos[i][j+1], self.image_into_pos[i+1][j+1]), axis=None)
				L2_normalized_val=np.sqrt(np.sum(hist_concat_pos**2))

				self.hog_descriptor[i][j]=hist_concat_pos/L2_normalized_val
				self.hog_descriptor[np.isnan(self.hog_descriptor)] = 0.0

		return self.hog_descriptor

# hog=hog_feature_generator()
# img=cv2.imread('sample_fruits.jpg')
# img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# feats=hog.generate_feats(img, (8, 8), 2)
# print(feats.shape)
# size = feats.shape[0]
# imgs = np.reshape(feats, [size, -1])  
# imgs2=feats.flatten()
# print(imgs.shape)
# print(imgs2.shape)