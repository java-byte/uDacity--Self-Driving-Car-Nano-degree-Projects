import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout

## Reading driving log CSV file using csv reader

lines=[]

with open('data/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)

## Adding Center, left, right images and their corresponding steering angle in images array and measurement array respectively.        

images=[]
measurements=[]
for line in lines:
    source_path_center=line[0]
    source_path_left=line[1]
    source_path_right=line[2]
    filename_center=source_path_center.split('/')[-1]
    filename_left=source_path_left.split('/')[-1]
    filename_right=source_path_right.split('/')[-1]
    current_path_center='data/IMG/'+filename_center
    current_path_left='data/IMG/'+filename_left
    current_path_right='data/IMG/'+filename_right
    image_center=cv2.imread(current_path_center)
    image_left=cv2.imread(current_path_left)
    image_right=cv2.imread(current_path_right)
    images.extend([cv2.cvtColor(image_center, cv2.COLOR_BGR2RGB),cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB),cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)])
    #images.extend([cv2.cvtColor(image_center, cv2.COLOR_BGR2YUV),cv2.cvtColor(image_left, cv2.COLOR_BGR2YUV),cv2.cvtColor(image_right, cv2.COLOR_BGR2YUV)])
    measurement_center=float(line[3])
    measurement_left=float(line[3])+0.2
    measurement_right=float(line[3])-0.2
    measurements.extend([measurement_center,measurement_left,measurement_right])



## Image Augmentation.
## Adding Extra images for training using flipping of images.
augmented_images, augmented_measurements=[],[]

for images,measurements in zip(images,measurements):
    augmented_images.append(images)
    augmented_measurements.append(measurements)
    augmented_images.append(cv2.flip(images,1))
    augmented_measurements.append(measurements*-1.0)


## Spliting of images in training and validation.
samples=list(zip(augmented_images,augmented_measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Total Sample size: {}'.format(len(samples)))
print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))

#Total Sample size: 53604
#Train samples: 42883
#Validation samples: 10721

def generator(samples, batch_size=32):
    """
    Generate the required images and measurments for training/
    `samples` is a list of pairs (`imagePath`, `measurement`).
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for image, measurement in batch_samples:
                images.append(image)
                angles.append(measurement)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

## Model Architecture

model = Sequential()

# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: (x/255.0)-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))

## Image cropping2D layer
## 50 rows pixels from the top of the image
## 20 rows pixels from the bottom of the image
## 0 columns of pixels from the left of the image
## 0 columns of pixels from the right of the image

## Adding Connvolutional Layers
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu')) ## subsample in Keras is the same as strides in tensorflow.
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=0)
model.save('model.h5')

"""
If the above code throw exceptions, try 
model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)
"""


