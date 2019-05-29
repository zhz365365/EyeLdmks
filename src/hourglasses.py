from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

slim = tf.contrib.slim

class Net(object):
	def __init__(self, net_input, net_name, stack=3, features=256,
                 use_batch_norm=False, is_training=True, dropout_keep_prob=0.5, reuse=None):
		self.net_input = net_input
		self.net_name = net_name
		self.stack = stack
		self.features = features
		self.use_batch_norm= use_batch_norm
		self.is_training = is_training
		self.dropout_keep_prob = dropout_keep_prob
		self.reuse = reuse
		self.net_arg_scope = self.arg_scope()
		self.scope = net_name
		self.predict_layers = []
		self.logits = {}
		with slim.arg_scope(self.net_arg_scope):
			self.hourglasses_net()

	def arg_scope(self):
		weight_decay=0.0000005
		batch_norm_decay=0.997
		batch_norm_epsilon=1e-5
		batch_norm_scale=True
		activation_fn=tf.nn.relu
		use_batch_norm=self.use_batch_norm
		batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS
		batch_norm_params = {
            'decay':batch_norm_decay,
            'epsilon':batch_norm_epsilon,
            'scale':batch_norm_scale,
            'updates_collections':batch_norm_updates_collections,
            'fused':None,
		}

		with slim.arg_scope(
            [slim.conv2d, slim.conv2d_transpose],
            padding='SAME',
            weights_regularizer=slim.l2_regularizer(weight_decay),
            biases_regularizer=slim.l2_regularizer(weight_decay),
            activation_fn=activation_fn,
            normalizer_fn=slim.batch_norm if use_batch_norm else None,
            normalizer_params=batch_norm_params if use_batch_norm else None):
			with slim.arg_scope([slim.batch_norm], **batch_norm_params):
				with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
					return arg_sc
    
	@slim.add_arg_scope
	def _residual(self, inputs, stride=1, outputs_collections=None, scope=None):
		"""
		(N)(H)(W)(C)
		[1x1xd]
		[3x3xd]     +   [1x1x4d]
		[1x1x4d]
		(N)(H//stride)(W//stride)(4d)
		"""
		with tf.variable_scope(scope, 'residual', [inputs]) as sc:
			depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
			if self.use_batch_norm:
				preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
			else:
				preact = tf.nn.relu(inputs)
			if depth_in != self.features:
				shortcut = slim.conv2d(preact, self.features, [1, 1], stride=stride,
    	                               normalizer_fn=None, activation_fn=None,
        	                           scope='shortcut')
			else:
				if stride == 1:
					shortcut = preact
				else:
					shortcut = slim.max_pool2d(preact, [1, 1], stride=stride, scope='shortcut')
			residual = slim.conv2d(preact, self.features//4, [1, 1], stride=1, scope='conv1')
			residual = slim.conv2d(residual, self.features//4, 3, stride=stride, scope='conv2')
			residual = slim.conv2d(residual, self.features, [1, 1], stride=1,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='conv3')
			output = shortcut + residual
			return slim.utils.collect_named_outputs(outputs_collections,
                                                    sc.name,
                                                    output)
    
	@slim.add_arg_scope
	def _up_add(self, a, b, outputs_collections=None, scope=None):
		with tf.variable_scope(scope, 'up', [a, b]) as sc:
			shape = b.get_shape().as_list()[1:3]
			output = tf.image.resize_nearest_neighbor(a, shape) + b
			return slim.utils.collect_named_outputs(outputs_collections,
                                                    sc.name,
                                                    output)
    
	@slim.add_arg_scope
	def _hourglasses(self, inputs, outputs_collections=None, scope=None):
		"""
		inputs:40x64x512
		outputs:40x64x512
		logits:40x64x55		
		C1:inputs:          40x64x512                                                                           C1_fix:residual(C1):    40x64x512
		C2:residual_2(C1):  20x32x512                                                                           C2_fix:residual(C2):    20x32x512
		C3:residual_2(C2):  10x16x512                                                                           C3_fix:residual(C3):    10x16x512
		C4:residual_2(C3):  5x8x512                                     
		C4a:residual(C4):   5x8x512     C4b:residual(C4a):      5x8x512         C4c:residual(C4b):  5x8x512
										C3b:up(C4c)+C1_fix:     10x16x512       C3c:residual(C3b):  10x16x512
										C2b:up(C3c)+C2_fix:     20x32x512       C2c:residual(C2b):  20x32x512
										C1b:up(C2c)+C3_fix:     40x64x512       C1c:residual(C1b):  40x64x512	C1d:residual(C1c):		40x64x512
		logits=conv2d(C1c):	40x64x55																			C1e:residual(logits):	40x64x512
		outputs:C1+C1d+C1e:	40x64x512
		"""
		with tf.variable_scope(scope, 'stack', [inputs]) as sc:
			C1 = slim.utils.collect_named_outputs(outputs_collections, sc.name+'/C1', inputs)
			C1_fix = self._residual(C1, scope='C1_fix')
			C2 = self._residual(C1, stride=2, scope='C2')
			C2_fix = self._residual(C2, scope='C2_fix')
			C3 = self._residual(C2, stride=2, scope='C3')
			C3a = self._residual(C3, scope='C3a')
			C3b = self._residual(C3a, scope='C3b')
			C3c = self._residual(C3b, scope='C3c')
			C2b = self._up_add(C3c, C2_fix)
			C2c = self._residual(C2b, scope='C2c')
			C1b = self._up_add(C2c, C1_fix)
			C1c = self._residual(C1b, scope='C1c')
			if self.use_batch_norm:
				postact = slim.batch_norm(C1c, activation_fn=tf.nn.relu, scope='postact')
				logits = slim.conv2d(postact, 55, [1, 1], stride=1, activation_fn=None, normalizer_fn=None, scope='heatmap')
			else:
				postact = tf.nn.relu(C1c)
				drop = slim.dropout(postact, self.dropout_keep_prob, is_training=self.is_training, scope='dropout')
				logits = slim.conv2d(drop, 55, [1, 1], stride=1, activation_fn=None, normalizer_fn=None, scope='heatmap')
			C1d = self._residual(C1c, scope='C1d')
			C1e = self._residual(logits, scope='C1e')
			output = C1+C1d+C1e
			return logits, slim.utils.collect_named_outputs(outputs_collections, sc.name, output)

	def hourglasses_net(self):
		with tf.variable_scope(self.scope, 'hourglasses', [self.net_input], reuse=self.reuse) as sc:
			end_points_collection = sc.original_name_scope + '_end_points'
			with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.max_pool2d, 
								 self._hourglasses, self._residual, self._up_add],
                                 outputs_collections=end_points_collection):
				with slim.arg_scope([slim.batch_norm], is_training=True):
					net = self.net_input
					with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
						# 160x256x3
						net = slim.conv2d(net, self.features//2, 3, stride=2, scope='conv1')
						# 80x128x256
						net = slim.conv2d(net, self.features, 1, stride=1, scope='conv2')
						net = slim.conv2d(net, self.features, 1, stride=1, scope='conv3')
					net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool1')
					# 40x64x512
					for i in range(5-self.stack, 5, 1):
						logits, net = self._hourglasses(net, scope='stack%d'%i)
						if not self.use_batch_norm:
							net = slim.dropout(net, 1.6*self.dropout_keep_prob, scope='dropout%d'%i)
						self.logits['P%d'%i] = logits
						self.predict_layers.append('P%d'%i)

					self.end_points = slim.utils.convert_collection_to_dict(end_points_collection)
					self.end_points[self.scope + '/logits'] = self.logits
