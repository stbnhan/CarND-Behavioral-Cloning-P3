import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

def getImages(csv_dir):
	
	lines = []
	
	with open(csv_dir) as csvfile:
		reader = csv.reader(csvfile)
		next(reader, None)
		for line in reader:
			lines.append(line)
	lines.pop(0)

	# Split Data
	from sklearn.model_selection import train_test_split
	return train_test_split(lines, test_size=0.2)
# end getImages()

def generator(samples, img_dir, batch_size=128):
	num_samples = len(samples)
	while 1:
		shuffle(samples)
		for offset in range (0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			measurements = []
			images_rev = []
			measurements_rev = []
			for batch_sample in batch_samples:
				name = img_dir+batch_sample[0].split('/')[-1]
				center_image = cv2.imread(name)
				center_angle = float(batch_sample[3])
				images.append(center_image)
				measurements.append(center_angle)
				images_rev.append(cv2.flip(center_image,1))
				measurements_rev.append(-center_angle)
			X_train = np.concatenate((np.array(images),np.array(images_rev)),axis=0)
			y_train = np.concatenate((np.array(measurements),np.array(measurements_rev)),axis=0)
			yield sklearn.utils.shuffle(X_train, y_train)
		# end for loop
	# end while loop
# end generator()

from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Flatten, Dense, Cropping2D, Activation, Dropout
from keras.layers.normalization import BatchNormalization

def myPreprocess():
	model = Sequential()
	model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
	model.add(Lambda(lambda x: (x / 255.0) - 0.5))

	return model
# end myProcess()

def myModel():
	model = myPreprocess()

	model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'))
	model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'))
	model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'))
	model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
	model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dense(50))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dense(10))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dense(1))

	return model
# end myModel()


# Piepline #

# Parameters:
batch_size = 64
epoch = 10

# Directory:
img_dir = './data/IMG/'

train_samples1, validation_samples1 = getImages('./data/driving_log.csv')
print(len(train_samples1))
print(len(validation_samples1))
train_samples2, validation_samples2 = getImages('./data/driving_log_swerve.csv')
print(len(train_samples2))
print(len(validation_samples2))
train_samples3, validation_samples3 = getImages('./data/driving_log_debug1.csv')
print(len(train_samples3))
print(len(validation_samples3))
train_samples4, validation_samples4 = getImages('./data/driving_log_debug2.csv')
print(len(train_samples4))
print(len(validation_samples4))
train_samples5, validation_samples5 = getImages('./data/driving_log_map2.csv')
print(len(train_samples5))
print(len(validation_samples5))

train_samples = []
validation_samples = []
train_samples = train_samples1 + train_samples2 + train_samples3 + train_samples4 + train_samples5
validation_samples = validation_samples1 + validation_samples2 + validation_samples3 + validation_samples4 + validation_samples5

print("train_samples size: ",len(train_samples))
print("validation_samples size: ",len(validation_samples))

train_generator = generator(train_samples, img_dir, batch_size=batch_size)
validation_generator = generator(validation_samples, img_dir, batch_size=batch_size)

model = myModel()
model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, steps_per_epoch= \
            len(train_samples)/batch_size, validation_data=validation_generator, \
            validation_steps=len(validation_samples)//batch_size, epochs=epoch)

model.summary()

# Save model data
model.save('./model.h5')

# Save model weights & model separately
# model.save_weights('./model.h5')
# json_string = model.to_json()
# with open('./model.json', 'w') as f:
# 	f.write(json_string)
