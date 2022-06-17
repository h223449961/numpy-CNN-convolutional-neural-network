
import numpy as np
import math
import random

class net:

	def __init__(self, n_hidden_layers, hidden_dims, input_dim):
		self.total_layers=n_hidden_layers+1
		self.parameters={}
		self.input_dimension=input_dim
		self.f_cache={}
		self.relu_cache={}

		inp_dim=self.input_dimension
		for i in range(self.total_layers-1):
			self.parameters['W_'+str(i+1)]=np.random.randn(inp_dim, hidden_dims[i])*0.01
			self.parameters['B_'+str(i+1)]=np.zeros([hidden_dims[i]])
			inp_dim=hidden_dims[i]

		self.parameters['W_'+str(self.total_layers)]=np.random.randn(inp_dim, 2)*0.01
		self.parameters['B_'+str(self.total_layers)]=np.zeros([2])

	def forward_activate(self, z):
		#status: complete
		activation=z.copy()
		activation[activation<0]=0
		return activation, z

	def single_layer_forward(self, A, W, b):
		#status: complete
		reshaped_A=np.reshape(A, [A.shape[0], -1])
		Z=np.dot(reshaped_A, W)+b
		cache=(A, W, b)
		return Z, cache

	def forward_prop_full(self, X):
		#status: complete
		size=X.shape[0]
		image_feats=np.reshape(X, [size, -1]) #from (b_size, 7, 7, 34) to (bs, 1764)

		for i in range(self.total_layers-1):
			z, self.f_cache[str(i+1)]=self.single_layer_forward(image_feats, self.parameters['W_'+str(i+1)], self.parameters["B_"+str(i+1)])
			activation, self.relu_cache[str(i+1)]=self.forward_activate(z)
			image_feats=activation.copy()

		z, last_layer_mem=self.single_layer_forward(image_feats, self.parameters['W_'+str(self.total_layers)], self.parameters["B_"+str(self.total_layers)])
		return z, last_layer_mem

	def back_prop_single(self, prev_da, final_cache):
		#status: complete
		a, w, b=final_cache
		num_images=a.shape[0]
		##doubt if this works without reshape 
		a=np.reshape(a, [num_images, -1])
		dx=np.dot(prev_da, w.T)
		dx=np.reshape(dx, a.shape)
		dw=np.dot(a.T, prev_da)
		db=np.sum(prev_da, axis=0)
		return dx, dw, db

	def backward_activation(self, prev_da, cache):
		#status: complete
		relu_mask=(cache>=0)
		dx=prev_da*relu_mask
		return dx

	def softmax_loss(self, y_hat, y):
		#status: complete
		#ref:http://bigstuffgoingon.com/blog/posts/softmax-loss-gradient/
		shifted=y_hat-np.max(y_hat, axis=1, keepdims=True)
		sums=np.sum(np.exp(shifted), axis=1, keepdims=True)
		log_probs=shifted-np.log(sums)
		probs=np.exp(log_probs)
		num_images=y_hat.shape[0]
		loss=-np.sum(log_probs[np.arange(num_images), y])/num_images
		
		dy=probs.copy()
		dy[np.arange(num_images), y]-=1
		dy=dy/num_images
		return loss, dy


	def back_prop_full(self, y_hat, y, final_layer_cache):
		#status: complete
		grads={}
		loss=0.0
		#for last layer outside the loop
		loss, prev_da=self.softmax_loss(y_hat, y)
		###note: may add regluarization here..
		#calculating derivatives to subtract from prev layer using last layer cache and dact
		prev_dx, prev_dw, prev_db=self.back_prop_single(prev_da, final_layer_cache)
		
		#initialize gradients
		grads['W_'+str(self.total_layers)]=prev_dw#add reg factor here
		grads['B_'+str(self.total_layers)]=prev_db

		for i in range(self.total_layers-1, 0, -1):#from reverse
			prev_da=self.backward_activation(prev_dx, self.relu_cache[str(i)]) #relu cache has one value. i.e. z
			prev_dx, prev_dw, prev_db=self.back_prop_single(prev_da, self.f_cache[str(i)])#use f_cache for backward relu
			grads["W_"+str(i)]=prev_dw
			grads["B_"+str(i)]=prev_db
			#may  add regularization here

		return loss, grads

	def update_params(self, grads_values, learning_rate):
		#status: complete
		for p, w in self.parameters.items():
			dw = grads_values[p]
			prev_dw = dw
			self.parameters[p] = self.parameters[p]-dw*learning_rate
	
	def accuracy(self, y_hat, y):
		return(np.mean(np.hstack(y_hat)==y))

	def train(self, X, Y, epochs, b_size, learning_rate):
		#status: complete
		np.random.seed(1029)
		n_images=X.shape[0]
		n_epochs=epochs
		nth_img=0
		for epoch in range(n_epochs):
			mask=np.random.choice(n_images, b_size)
			x=X[mask]
			y=Y[mask]
			scores, cache=self.forward_prop_full(x)
			# print(scores)
			# print(scores.shape)
			loss, gradients=self.back_prop_full(scores, y, cache)			
			self.update_params(gradients, learning_rate)
			
			if(epoch%20==0):
				print("epoch :", epoch)
				print("loss :", loss) 
				#print("accuracy :", self.accuracy(np.argmax(scores, axis=1), y))

	def test(self, X):
		# status: complete
		preds=[]
		y_pred, cache=self.forward_prop_full(X)
		preds.append(np.argmax(y_pred, axis=1))
		return y_pred