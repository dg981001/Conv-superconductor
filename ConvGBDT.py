import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import tensorflow as tf
import numpy as np
from function.Evalution import Evaluating_ML
import matplotlib.pyplot as plt
import argparse


def load_data(file_path):
	data = np.loadtxt(file_path, delimiter = ",", skiprows = 1, dtype = str)
	feature = data[:,1:-1].astype(np.float32)
	target = data[:,-1].astype(np.float32)
	data = data[:,1:].astype(np.float32)
	return data, feature, target

def min_max_matrix(data):
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

def min_max_vector(data, ways):
	"""
	Matrix normalization
	data:A two-dimensional matrix
	"""
	if ways == "row":
		max, min = np.max(data, axis = 1)[:,np.newaxis], np.min(data, axis = 1)[:,np.newaxis]
		result = np.nan_to_num(
					np.subtract(data, min) / np.subtract(max, min)
				)
	if ways == "column":
		max, min = np.max(data, axis = 0)[np.newaxis,:], np.min(data, axis = 0)[np.newaxis,:]
		result = np.nan_to_num(
					np.subtract(data, min) / np.subtract(max, min)
				)
	return result

def data_divide(feature, target, percentage = 0.1):
	"""
	Random partition of training set and test set according to percentage
	"""
	all_index = np.arange(feature.shape[0])
	np.random.shuffle(all_index)
	val_numb = int(feature.shape[0] * percentage)
	val_index = all_index[:val_numb]
	val_feature, val_target = feature[val_index], target[val_index][:,np.newaxis]
	train_feature = np.delete(feature, val_index, axis = 0)
	train_target = np.delete(
						target, 
						val_index, 
						axis = 0
					)[:,np.newaxis]
	return (val_feature, val_target), (train_feature, train_target)

def get_cnn_feature(data, args):
	data = min_max_matrix(data)
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph(args.model_name)
		saver.restore(sess, tf.train.latest_checkpoint(args.model_path))
		graph = tf.get_default_graph()
		cnn_input = graph.get_tensor_by_name("CNN/c2:0")
		x = graph.get_tensor_by_name("x:0")	
		cnn_feature = sess.run(
						cnn_input,
						feed_dict = {
							x : data
						}
					)
		return np.squeeze(cnn_feature)

class ML(object):
	def __init__(self, args):
		gbdt = GradientBoostingRegressor(
					loss = "ls", 
					n_estimators = 400, 
					learning_rate = 0.06, 
					subsample = 0.4, 
					max_depth = 20,
				)
		self.way = gbdt

def train(model, feature, target):
	train_ev = Evaluating_ML()
	model.way.fit(feature, target)
	pre = model.way.predict(feature)
	train_ev.update(target, pre)
	return (train_ev.mae(), train_ev.rmse(), train_ev.R())

def val(model, feature, target):
	test_ev = Evaluating_ML()
	pre = model.way.predict(feature)
	test_ev.update(target, pre)
	return (test_ev.mae(), test_ev.rmse(), test_ev.R())

def main(args):
	_, fe, target = load_data(args.data)
	print(fe.shape)
	cnn_fe = min_max_vector(
				data = get_cnn_feature(fe, args),
				ways = "column"
			)
	print(cnn_fe.shape)
	best_rt = np.zeros([args.times, 6])	
	for time in range(args.times):
		(val_fe, val_tg), (train_fe, train_tg) = \
								data_divide(cnn_fe, target, args.percentage)
		method = ML(args)
		train_rt = train(method, train_fe, train_tg)
		val_rt = val(method, val_fe, val_tg)
		best_rt[time] = np.concatenate(
							[train_rt, val_rt], 
							axis = 0
						)
		print("*"*10 + str(time) + " time result is :" + "*"*10)
		print(best_rt[time])
	print("*"*10 + str(args.times) + " times best result are:" + "*"*10)
	print(best_rt)
	print("*"*10 + str(args.times) + " times mean result are:" + "*"*10)
	print(np.mean(best_rt, axis = 0))

def parse_args():

	parser = argparse.ArgumentParser() 
	parser.add_argument("--data", type = str, default = "Dataset/DataV.csv",
						help = "the data to enconder")
	parser.add_argument("--model_name", type = str, default = "check_point/DataV/best_model.ckpt.meta",
						help = "model name")
	parser.add_argument("--model_path", type = str, default = 'check_point/DataV',
						help = "model path")
	parser.add_argument("--percentage", type = float, default = 0.1,
						help = "percentage")
	parser.add_argument("--times", type = int, default = 10,
						help = "cv times")
	parser.add_argument("--rf_estiminators", type = int, default = 500,
						help = "Number of trees in random forests")
	parser.add_argument("--rf_cri", type = str, default = "mse",
						help = "Random forest optimization standard")
	return parser.parse_args()

if __name__ == "__main__":
	main(parse_args())