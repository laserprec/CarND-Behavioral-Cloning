import csv
import cv2
import numpy as np

ACTIVATION_FUNC = 'tanh'
LOSS_FUNC       = 'mse'
LEARNING_RATE   = 1e-4

lines = []
with open('./data/driving_log.csv') as f:
	reader = csv.reader(f)
	for line in reader:
		lines.append(line)

images = []
measurements = []

for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	curr_path = './data/IMG/' + filename
	image = cv2.imread(curr_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Activation, Convolution2D, Cropping2D, Dense, Dropout
from keras.layers import Flatten, Lambda, MaxPooling2D

def build_model():
	""" 
	Building the model for the behavioral cloning net.
	Model is based on Nvidia's
	"End to End Learning for Self-Driving Cars" paper
	"""
	model = Sequential()
	# Normalize the data
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
	# Crop the data 
	model.add(Cropping2D(cropping=((70, 25), (0,0))))
	# Convolution layers
	model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation=ACTIVATION_FUNC))
	model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation=ACTIVATION_FUNC))
	model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation=ACTIVATION_FUNC))
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), activation=ACTIVATION_FUNC))
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), activation=ACTIVATION_FUNC))
	# Fully-connected layers
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))

	model.compile(optimizer=Adam(lr=LEARNING_RATE), loss=LOSS_FUNC)
	return model

model = build_model()
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=6)

model.save('model.h5')
