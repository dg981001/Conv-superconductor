import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from function.batch_class import Dataset
from function.Evalution import Evaluating_DL
# from function.oversampling import Oversampling
import argparse
import math
import os
import time as ts



def load_data(file_path):
	data = np.loadtxt(file_path, delimiter = ",", skiprows = 1, dtype = str)
	feature = data[:,1:-1].astype(np.float32)
	target = data[:,-1].astype(np.float32)[:,np.newaxis]
	data = data[:,1:].astype(np.float32)
	return data, feature, target

def mkdir(path):
	isExists = os.path.exists(path)
	if not isExists:
		os.makedirs(path)
		return True
	else:
		return False

def linear(input, output_dim, scope = None, name = None, stddev = 1.0):
	with tf.variable_scope(scope or "linear"):
		w = tf.get_variable(
				name = "w",
				shape = [input.get_shape()[1], output_dim], 
				initializer = tf.random_normal_initializer(stddev = stddev)
			)
		b = tf.get_variable(
				name = "b",
				shape = [output_dim],
				initializer = tf.constant_initializer(0.0)
			)
	return tf.add(tf.matmul(input, w), b, name = name)

def cnn2d(input, filter_shape, strid_shape, padding, scope = None, name = None, stddev = 1.0):
	with tf.variable_scope(scope or "cnn"):
		w = tf.get_variable(
				name = "w",
				shape = filter_shape,
				initializer = tf.random_normal_initializer(stddev = stddev)
			)
		b = tf.get_variable(
				name = "b",
				shape = filter_shape[-1],
				initializer = tf.constant_initializer(0.0)
			)
	return tf.add(
				tf.nn.conv2d(
					input = input, 
					filter = w,
					strides = strid_shape,
					padding = padding
				),
				b,
				name = name
			)

def bn_layer(x, is_training, scope = None, name = None, moving_decay = 0.9, eps = 1e-5):
	param_shape = x.get_shape()[-1]
	with tf.variable_scope(scope or "BatchNorm"):
		gamma = tf.get_variable(
					name = "gamma",
					shape = param_shape,
					initializer = tf.constant_initializer(1)
				)
		beta = tf.get_variable(
					name = "beat",
					shape = param_shape,
					initializer = tf.constant_initializer(0)
				)
		axis = list(range(len(x.get_shape()) - 1))
		batch_mean, batch_var = tf.nn.moments(
				x, 
				axis, 
				name = "moments"
			)		
		ema = tf.train.ExponentialMovingAverage(
					decay = moving_decay,
					name = "ExponentialMovingAverage"
				)
		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		mean, var = tf.cond(
						pred = tf.equal(is_training, True), 
						true_fn = mean_var_with_update, 
						false_fn = lambda : (ema.average(batch_mean), ema.average(batch_var))
					)
	return tf.nn.batch_normalization(
				x = x, 
				mean = mean, 
				variance = var, 
				offset = beta, 
				scale = gamma, 
				variance_epsilon = eps, 
				name = name
			)

def min_max(data):
	"""
	Matrix normalization by row
	data:A three-dimensional matrix
	"""
	data = data.reshape((data.shape[0], 22, 6))
	max, min = np.max(data, axis = 2)[:,:,np.newaxis], np.min(data, axis = 2)[:,:,np.newaxis]
	result = np.nan_to_num(
				np.subtract(data, min) / np.subtract(max, min)
			)[:,:,:,np.newaxis]

	return result

def optimizer(loss, var_list, learning_rate = 0.01):
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
		loss,
		var_list = var_list
	)
	return optimizer

def data_divide(feature, target, percentage = 0.1):
	"""
	Random partition of training set and test set according to percentage
	"""
	all_index = np.arange(feature.shape[0])
	np.random.shuffle(all_index)
	val_numb = int(feature.shape[0] * percentage)
	val_index = all_index[:val_numb]
	val_feature, val_target = feature[val_index], target[val_index]
	train_feature = np.delete(feature, val_index, axis = 0)
	train_target = np.delete(
						target,
						val_index,
						axis = 0
					)
	return (val_feature, val_target), (train_feature, train_target)

def netwotk(input, is_training):

	h0 = cnn2d(
			input = input, 
			filter_shape = [1, 6, 1, 32], 
			strid_shape = [1, 1, 1, 1],
			padding = "VALID",
			scope = "cnn0",
			name = "c0"
		)
	bn0 = bn_layer(
			x = h0, 
			is_training = is_training,
			scope = "batch_norm_0", 
			name = "bn0"
		)
	h1 = cnn2d(
			input = tf.nn.relu(bn0),
			filter_shape = [1, 1, 32, 32],
			strid_shape = [1, 1, 1, 1],
			padding = "VALID",
			scope = "cnn1",
			name = "c0"
		)
	bn1 = bn_layer(
			x = h1, 
			is_training = is_training,
			scope = "batch_norm_1", 
			name = "bn1"
		)
	h2 = cnn2d(
			input = tf.nn.relu(bn1),
			filter_shape = [22, 1, 32, 32],
			strid_shape = [1, 1, 1, 1],
			padding = "VALID",
			scope = "cnn2",
			name = "c2"
		)
	bn2 = bn_layer(
				x = h2, 
				is_training = is_training,
				scope = "batch_norm_2", 
				name = "bn2"
			)

	l0 = linear(
			input = tf.reshape(tf.nn.relu(bn2), [-1, 32]),
			output_dim = 16,
			scope = "linear_0",
			name = "l0"
		)
	l1 = linear(
			input = tf.nn.relu(l0),
			output_dim = 1,
			scope = "linear_1",
			name = "l1"
		)
	return l1

class model(object):

	def __init__(self, lr, **args):
		self.x = tf.placeholder(
					tf.float32, 
					shape = args["input_shape"],
					name = "x"
				)
		self.y = tf.placeholder(
					tf.float32,
					shape = (None, 1),
					name = "y"
				)
		self.is_training = True
		with tf.variable_scope("CNN"):
			self.pre = netwotk(self.x, self.is_training)

		self.loss = tf.reduce_mean(tf.square(self.pre - self.y))

		vars = tf.trainable_variables()

		self.params = [v for v in vars if v.name.startswith("CNN/")]

		self.opt_net = optimizer(self.loss, self.params, lr)

def get_net_predict(data, **params):

	with tf.Session() as sess:
		saver = tf.train.import_meta_graph(params["model_name"])
		saver.restore(sess, tf.train.latest_checkpoint(params["model_path"]))
		graph = tf.get_default_graph()
		net_output = graph.get_tensor_by_name("CNN/l1:0")
		x = graph.get_tensor_by_name("x:0")	
		predict = sess.run(
						net_output,
						feed_dict = {
							x : data
						}
					)
		return np.squeeze(predict)

def train(sess, model, feature, target, batch_size):
	index = Dataset(
		np.arange(0, feature.shape[0])
		)
	train_ev = Evaluating_DL()
	model.is_training = True
	while index.iteration:
		ind = index.next_batch(batch_size)
		_, pre, _ = sess.run(
				[model.opt_net, model.pre, model.loss], 
				feed_dict = {
					model.x : feature[ind],
					model.y : target[ind]
				}
			)
		train_ev.update(target[ind], pre)
	return (train_ev.mae(), train_ev.rmse(), train_ev.R())

def test(sess, model, feature, target, batch_size):
	index = Dataset(
		np.arange(0, feature.shape[0])
		)
	test_ev = Evaluating_DL()
	model.is_training = False
	while index.iteration:
		ind = index.next_batch(batch_size)
		pre, _ = sess.run(
				[model.pre, model.loss], 
				feed_dict = {
					model.x : feature[ind],
					model.y : target[ind]
				}
			)
		test_ev.update(target[ind], pre)
	return (test_ev.mae(), test_ev.rmse(), test_ev.R())

def val(feature, target, batch_size, **params):
	index = Dataset(
		np.arange(0, feature.shape[0])
		)
	val_ev = Evaluating_DL()
	while index.iteration:
		ind = index.next_batch(batch_size)
		pre = get_net_predict(feature[ind], **params)
		val_ev.update(target[ind], pre)
	tf.reset_default_graph()
	return [val_ev.mae(), val_ev.rmse(), val_ev.R()]

def main(args):
	_, feature, target = load_data(args.data)
	feature = min_max(
				feature.reshape((feature.shape[0], 22, 6))
			)
	best_test_rt = np.zeros([args.times, 6])
	all_val_rt = np.zeros([args.times, 3])

	for time in range(args.times):
		(val_fe, val_tg), (train_fe, train_tg) = \
										data_divide(feature, target, args.val_per)
		(test_fe, test_tg), (train_fe, train_tg) = \
										data_divide(train_fe, train_tg, args.test_per)
		# _, (feature, target) = \
		# 				data_divide(feature, target, 0.49)
		# (val_fe, val_tg), (train_fe, train_tg) = \
		# 								data_divide(feature, target, args.val_per)
		# (test_fe, test_tg), (train_fe, train_tg) = \
		# 								data_divide(train_fe, train_tg, args.test_per)
		# print(train_fe.shape)

		timestart = ts.time()
		best_R = 0
		net = model(
				args.lr, 
				input_shape = [None, feature.shape[1], feature.shape[2], 1], 
			)

		model_path = args.model_path + str(time)
		mkdir(model_path)
		model_name = model_path + "/best_model.ckpt"

		saver = tf.train.Saver(max_to_keep = 1)

		with tf.Session() as sess:
			saver = tf.train.Saver(max_to_keep = 1)
			tf.local_variables_initializer().run()
			tf.global_variables_initializer().run()
			for step in range(args.num_steps):
				stepstart = ts.time()
				train_rt = train(
								sess = sess, 
								model = net, 
								feature = train_fe, 
								target = train_tg, 
								batch_size = args.batch_size
							)
				test_rt = test(
								sess = sess, 
								model = net, 
								feature = test_fe, 
								target = test_tg, 
								batch_size = args.batch_size
							)
				print("step:{} train_mae:{:.4f} train_rmse:{:.4f} train_R:{:.4f} test_mae:{:.4f} test_rmse:{:.4f} test_R:{:.4f} time:{:.2f}".format(
						step,
						train_rt[0], train_rt[1], train_rt[2],
						test_rt[0], test_rt[1], test_rt[2],
						ts.time() - stepstart
					)
				)

				if test_rt[2] > best_R:
					best_R = test_rt[2]
					best_test_rt[time] = np.concatenate(
										[train_rt, test_rt], 
										axis = 0
									)
					if args.save_model:
						saver.save(sess, model_name)
		print("Time taken : {:.2f}".format(ts.time()-timestart))
		print("*"*10 + "Start calculating the results of the validation set" + "*"*10)
		val_rt = val(
					feature = val_fe, 
					target = val_tg, 
					batch_size = args.batch_size, 
					model_path = model_path, 
					model_name = model_name + ".meta"
				)
		all_val_rt[time] = val_rt
		print("*"*10 + str(time) + " time best test result is:" +"*"*10)
		print(best_test_rt[time])
		print("*"*10 + str(time) + " time val result is: "+"*"*10)
		print(all_val_rt[time])

	print("*"*10 + str(args.times) + " times best result are:" + "*"*10)
	print(best_test_rt)
	print("*"*10 + str(args.times) + " times test set mean result is:" + "*"*10)
	print(np.mean(best_test_rt, axis = 0))
	print("*"*10 + str(args.times) + " times val result are:" + "*"*10)
	print(all_val_rt)
	print("*"*10 + str(args.times) + " times val set mean result is:" + "*"*10)
	print(np.mean(all_val_rt, axis = 0))

def parse_args():
	path = os.path.abspath("..")
	parser = argparse.ArgumentParser()

	parser.add_argument("--data", type = str, default = "Dataset/DataK.csv",
						help = "The path of data")
	parser.add_argument("--val_per", type = float, default = 0.1,
						help = "percentage of validation set")
	parser.add_argument("--test_per", type = float, default = 0.1,
						help = "percentage of test set")
	parser.add_argument("--num_steps", type = int, default = 700,
						help = "the number of training steps to take")
	parser.add_argument("--batch_size", type = int, default = 32,
						help = "the batch size")
	parser.add_argument("--lr", type = float, default = 1e-2,
						help = "model learning rate")
	parser.add_argument("--times", type = int, default = 10,
						help = "cv times")
	parser.add_argument("--save_model", type = bool, default = True,
						help = "whether to save model")
	parser.add_argument("--model_path", type = str, default = "check_point_k/model_",
						help = "model save path")

	return parser.parse_args()

if __name__ == "__main__":
	main(parse_args())
