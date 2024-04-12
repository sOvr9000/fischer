
from namegen import generate_name


import numpy as np
import tensorflow.keras as keras


class NameClassifier:
	def __init__(self):
		self.model = None
	def load_model(self, fpath):
		try:
			self.model = keras.models.load_model(fpath)
		except Exception:
			raise Exception(f'Model could not be loaded at path \'{fpath}\'.')
	def classify(self, name):
		if self.model is None:
			raise Exception('model is not defined.  Set it with NameClassifier.load_model(fpath)')
		p = self.model.predict(name).reshape((-1,))
		return np.argmax(p)



def train_name_classifier(
	model_fpath = None, # None means create a new, randomly initialized model
	model = None, # THIS MUST BE A KERAS MODEL if model_fpath is None
	model_save_fpath = None, # None means save in current directory
	data_set_fname = None, # if not None, add to a previously created data and keep training on the whole data set
	data_set_save_fname = None, # if not None, save generated data for later use
):
	inp = ' '
	d = []
	if model_fpath is not None:
		model = keras.models.load_model(model_fpath)
	if model is None:
		raise Exception('model_fpath was left as None, but model is not defined.  If not loading a model, make sure to set model to be a Keras.models.Sequential object, as a keyword argument in train_name_classifier().')
	while inp != '':
		n = generate_name()
		print('\n\n' + '-'*16)
		print(n)
		print('\nClassify this name.')
		inp = input().strip()
		if inp != '':
			d.append([n, inp])
	if data_set_save_fname is not None:
		with open(data_set_save_fname, 'wb') as f:
			f.write('name,class\n' + '\n'.join([
				','.join([
					a
					for a in b
				])
				for b in d
			]))
	if model_save_fpath is None:
		print('WARNING -- model is not being saved! It can be saved by specifying the keyword argument model_save_fpath in train_name_classifier().')
		print('Set it now? (y/n)')
		inp = input()
		if inp == 'y':
			print('Please specify the file path now so that the model can be saved.')
			while True:
				inp = input()
				try:
					model.save(inp)
					break
				except Exception:
					print('There was an error saving there! Try a different file path, and include the .h5 file extension.')
	else:
		model.save(model_save_fpath)


if __name__ == '__main__':
	train_name_classifier()

