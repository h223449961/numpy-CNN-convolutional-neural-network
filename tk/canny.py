'''
author: Pruthviraj Patil
version: 1.0

'''

import cv2
import numpy as np
import math

class Canny_edge_detector:
	def __init__(self):
		self.Xprewitt_filter=np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
		self.Yprewitt_filter=np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

	def save_image(self, image, name):
		#status:complete
		cv2.imwrite("./", name, image)

	def apply_filter_per_pixel(self, n_factor, mask, image_patch):
		#status: complete
		convolved_val=0
		x_size=mask.shape[0]
		y_size=mask.shape[1]

		for i in range(x_size):
			for j in range(y_size):
				convolved_val=mask[i][j]*image_patch[i][j]

		return(convolved_val/n_factor)

	def find_convolution(self, image, xMask, yMask):
		#status:complete
		x_grad_output_image=np.zeros(image.shape)
		y_grad_output_image=np.zeros(image.shape)

		padding=int(math.floor(xMask.shape[0]))#masks are same sized for both x, y

		x_val=image.shape[0]
		y_val=image.shape[1]
		norm=3
		for x_pivot in range(padding, x_val-1):
			for y_pivot in range(padding, y_val-1):
				x_grad_output_image[x_pivot][y_pivot]=self.apply_filter_per_pixel(norm, xMask, image[x_pivot-padding: x_pivot+padding+1, y_pivot-padding : y_pivot+padding+1])
				y_grad_output_image[x_pivot][y_pivot]=self.apply_filter_per_pixel(norm, yMask, image[x_pivot-padding: x_pivot+padding+1, y_pivot-padding : y_pivot+padding+1])

		return (x_grad_output_image, y_grad_output_image)

	def grad_from_image(self, image):
		# status: complete
		#apply filters over the image
		gx, gy=self.find_convolution(image, self.Xprewitt_filter, self.Yprewitt_filter,)

		# save_image(gx)
		# save_image(gy)

		return(gx, gy)

	def dir_from_grads(self, gx, gy):
		# status: complete
		# calculate the gradients
		x_len=gx.shape[0]
		y_len=gx.shape[1]
		out=np.zeros(gx.shape, np.float32)

		for i in range(x_len):
			for j in range(y_len):
				#keeping in mind that tan(0)=inf, and dir=tan(gy/gx)
				if(gx[i][j]!=0):
					out[i][j]=math.degrees(math.atan2(gy[i][j], gx[i][j]))
				else:
					if(gy[i][j]>0):
						out[i][j]=90
					elif(gy[i][j]<0):
						out[i][j]=-90
					else:
						out[i][j]=0
				# clipping to 180
				if(out[i][j]<-180):
					out[i][j]=out[i][j]+360
				elif(out[i][j]<=180):
					out[i][j]=out[i][j]+180

				if(out[i][j]>=180):
					out[i][j]=out[i][j]-180

		return out

	def mag_from_grads(self, gx, gy):
		#status: complete
		mags=np.zeros(gx.shape, np.float32)
		x_len=gx.shape[0]
		y_len=gx.shape[1]

		for i in range(x_len):
			for j in range(y_len):
				mags[i][j]=np.sqrt(gx[i][i]**2 + gy[i][j]**2) #div/root2?????

		# save_image(mags, "mag")
		return(mags)