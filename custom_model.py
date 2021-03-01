from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D

from keras import models
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

my_model= models.Sequential()

# Add first convolutional block
my_model.add(Conv2D(16, (3, 3), activation='relu', padding='same', 
                    input_shape=(178,218,3)))
my_model.add(MaxPooling2D((2, 2), padding='same'))

# second block
my_model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
my_model.add(MaxPooling2D((2, 2), padding='same'))
# third block
my_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
my_model.add(MaxPooling2D((2, 2), padding='same'))
# fourth block
my_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
my_model.add(MaxPooling2D((2, 2), padding='same'))


# global average pooling
my_model.add(GlobalAveragePooling2D())
# fully connected layer
my_model.add(Dense(64, activation='relu'))
my_model.add(BatchNormalization())
# make predictions
my_model.add(Dense(2, activation='sigmoid'))


# Show a summary of the model. Check the number of trainable parameters
my_model.summary()


# use early stopping to optimally terminate training through callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint
es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)


mc= ModelCheckpoint('/content/drive/My Drive/dataset_model.h5', monitor='val_loss', 
                    mode='min', verbose=1, save_best_only=True)
cb_list=[es,mc]

# compile model 
my_model.compile(optimizer='adam', loss='binary_crossentropy', 
                 metrics=['accuracy'])

from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# set up data generator
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

# get batches of training images from the directory
train_generator = data_generator.flow_from_directory(
        '/content/drive/My Drive/dataset/train',
        target_size=(178, 218),
        batch_size=12,
        class_mode='categorical')
# get batches of validation images from the directory
validation_generator = data_generator.flow_from_directory(
        '/content/drive/My Drive/dataset/val',
        target_size=(178, 218),
        batch_size=12,
        class_mode='categorical')

history = my_model.fit_generator(
        train_generator,
        epochs=15,
        steps_per_epoch=400,
        validation_data=validation_generator,
        validation_steps=50, callbacks=cb_list)

# plot training and validation accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylim([.5,1.1])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("Custom_Keras_ODSC.png", dpi=300)

# load a saved model
from keras.models import load_model
import os
os.chdir('/content/drive/My Drive')
saved_model = load_model('Custom_Keras_CNN.h5')

test_generator = data_generator.flow_from_directory(
        '/content/drive/My Drive/dataset/test',
        target_size=(178, 218),
        batch_size=1,
        class_mode='categorical',
        shuffle=False)
import numpy as np
test_generator.reset()
pred=saved_model.predict_generator(test_generator, verbose=1, steps=1000)


# determine the maximum activation value for each sample
predicted_class_indices=np.argmax(pred,axis=1)

labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames
filenz=[0]
for i in range(0,len(filenames)):
    filenz.append(filenames[i].split('\\')[0])
filenz=filenz[1:]

match=[]
for i in range(0,len(filenames)):
    match.append(filenz[i]==predictions[i])
match.count(True)/1000