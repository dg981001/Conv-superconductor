import numpy as np
from tsne import bh_sne
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_data(file_path):
	data = np.loadtxt(file_path, delimiter = ",", skiprows = 1, dtype = str)
	formula = data[:,0]
	feature = data[:,1:-1].astype(np.float64)
	target = data[:,-1].astype(np.float64)
	return formula, feature, target

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

def sampling(data, label, percentage = 0.001):
	data_ind = np.arange(data.shape[0])
	np.random.shuffle(data_ind)
	sample_numb = int(data.shape[0] * percentage)
	sample_index = data_ind[:sample_numb]
	return data[sample_index], label[sample_index]

def scatter_plot(data, label, name):
	a, b, c, d = 0, 0, 0, 0
	for i, xy in enumerate(data):
		if ("Cu" in label[i]) and ("Fe" not in label[i]):
			if a == 1:
				plt.scatter(xy[0], xy[1], color = "#3A5FCD", marker = "s", s = 5, label = "Cuprates")
				a +=1 
			else:
				plt.scatter(xy[0], xy[1], color = "#3A5FCD", marker = "s", s = 5)
				a += 1
		if ("Fe" in label[i]) and ("Cu" not in label[i]):
			if b == 1:
				plt.scatter(xy[0], xy[1], color = "#EE0000", marker = "v", s = 2, label = "Iron-based")
				b += 1
			else:
				plt.scatter(xy[0], xy[1], color = "#EE0000", marker = "v", s = 2)
				b += 1
		if ("Cu" not in label[i]) and ("Fe" not in label[i]):
			if c == 1:
				plt.scatter(xy[0], xy[1], color = "black", marker = "x", s = 2, label = "Cuprates and Iron-based")
				c += 1
			else:
				plt.scatter(xy[0], xy[1], color = "black", marker = "x", s = 2)
				c += 1
		if ("Fe" in label[i]) and ("Cu" in label[i]):
			if d == 1:
				plt.scatter(xy[0], xy[1], color = "#00FF00", marker = "o", s = 2, label = "Neither Cup nor Iron-based")
				d += 1
			else:
				plt.scatter(xy[0], xy[1], color = "#00FF00", marker = "o", s = 2)
				d += 1
	plt.legend(shadow = True, numpoints = 1, framealpha = 1, frameon = True, edgecolor = "black")
	plt.savefig(name + ".pdf", bbox_inches = "tight")
	plt.show()

def sample(data, perc):
	index = np.arange(len(data))
	np.random.shuffle(index)
	return index[:int(len(index)*perc)]

class Dim_reduce(object):
	def __init__(self, way = "tsne"):
		self.way = way
		if self.way == "tsne":
			self.red_dim = bh_sne
		if self.way == "pca":
			self.red_dim = PCA(n_components = 2).fit_transform

def main(args):
	formula, fe, target = load_data(args.data)
	print(fe.shape)
	cnn_fe = min_max_vector(
				data = get_cnn_feature(fe, args),
				ways = "column"
			).astype(np.float64)
	model = Dim_reduce(way = args.way)
	X_2d = model.red_dim(cnn_fe)
	# X_2d = model.red_dim(fe)
	# [sample(X_2d, 0.3)]
	X_2d, formula = sampling(X_2d, formula, percentage = args.per)

	scatter_plot(X_2d, formula, args.way)
 
def parse_args():

	parser = argparse.ArgumentParser()
	parser.add_argument("--data", type = str, default = "data/DataK.csv",
						help = "the data to enconder")
	parser.add_argument("--model_name", type = str, default = "check_point/DataK/model_0/best_model.ckpt.meta",
						help = "model name")
	parser.add_argument("--model_path", type = str, default = 'check_point/DataK/model_0',
						help = "model path")
	parser.add_argument("--per", type = float, default = 1,
						help = "percentage of data to plot")
	parser.add_argument("--way", type = str, default = "tsne",
					help = "the way to dimensionality reduction")

	return parser.parse_args()

if __name__ == "__main__":
	main(parse_args())