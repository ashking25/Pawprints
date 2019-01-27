import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Input, Concatenate
from keras import backend as K
import keras
from sklearn.utils import class_weight
from keras.callbacks import TensorBoard
from skimage import exposure
import sys

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
cnn_model = sys.argv[1]
nblocks = float(sys.argv[2])
if cnn_model == 'Inception':
    from keras.applications.inception_v3 import preprocess_input
elif cnn_model == 'VGG':
    from keras.applications.vgg16 import preprocess_input
elif cnn_model == 'ResNet':
    from keras.applications.resnet50 import preprocess_input

def imageLoader(files, batch_size, y_train, df_name):

    L = len(files)

    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            X = load_images_from_list(files[batch_start:limit])
            X = np.reshape(np.array(X), (np.shape(X)[0], np.shape(X)[1],
                                         np.shape(X)[2], np.shape(X)[3]))
            X_metadata = load_metadata(files[batch_start:limit].astype('int'), df_name)
            Y = y_train[batch_start:limit]
            yield [np.array(X),np.array(X_metadata)],Y #a tuple with two numpy arrays with batch_size samples

            batch_start += batch_size
            batch_end += batch_size

def load_images_npy(files):
    #already preprocessed images
    # just flip and rotate at random
    images = []
    for f in files:
        x = np.load('../images_footprints/'+str(int(f))+'.npy')
        x = np.rot90(x,k=np.random.choice(range(4)))
        if np.random.choice(range(2)) == 1:
            x = np.transpose(x, axes=[1,0,2])
        x = np.expand_dims(x, axis=0)
        images.append(x[0])
    return images

def load_images_from_list(files, xpixels=299, ypixels=299):

    if cnn_model == 'ResNet':
        xpixels = 224
        ypixels = 224
    images = []
    for f in files:
        #load images
        img_path = '../images_footprints/'+str(int(f))+'.jpg'
        img = image.load_img(img_path)
        x = image.img_to_array(img)
        x[np.isnan(x)]=0
        x = exposure.equalize_adapthist(x/np.max(x), clip_limit=0.03)*256
        x = np.rot90(x,k=np.random.choice(range(4)))
        if np.random.choice(range(2)) == 1:
            x = np.transpose(x, axes=[1,0,2])

        #Reshape
        if x.shape[0] < xpixels or x.shape[1] < ypixels:
            # pad image with zeros
            dummy = np.zeros((np.max((x.shape[0], xpixels)), np.max((x.shape[1], ypixels)), 3))
            xstart = np.max((int((xpixels-x.shape[0])/2),0))
            ystart = np.max((int((ypixels-x.shape[1])/2),0))
            dummy[xstart:xstart+x.shape[0], ystart:ystart+x.shape[1],:] = x
            x = dummy
        if x.shape[0] > xpixels or x.shape[1] > ypixels:
            #crop image in the center
            xlen = x.shape[0]
            ylen = x.shape[1]
            x = x[int((xlen - xpixels)/2):int((xlen - xpixels)/2)+xpixels, \
                  int((ylen - xpixels)/2):int((ylen - xpixels)/2)+ypixels, :]

        x = np.expand_dims(x, axis=0)

        x = preprocess_input(x)
        images.append(x[0])
    return images

def load_metadata(files, df_name):
    """ Load information about size, location, and day of the year """
    df= pd.read_csv(df_name,index_col=0, )

    size = np.random.uniform(low=df['min_track'].loc[files], high=df['max_track'].loc[files])#/12. # normalize to 1

    day_sin = np.sin(df['DayofYear'].loc[files]/365.*np.pi*2.).values+1
    day_cos = np.cos(df['DayofYear'].loc[files]/365.*np.pi*2.).values+1

    lat = df['latitude'].loc[files].values / 360. # normalize to 1
    long = (360+df['longitude'].loc[files].values)/360. #normalize to 1

    if np.isnan(np.sum(lat)) or np.isnan(np.sum(long)):
        lat[np.where(np.isnan(np.array(lat)))] = 0
        long[np.where(np.isnan(np.array(long)))] = 0

    return np.array([size, day_sin, day_cos, lat, long]).T

def my_model(cnn_model, class_weights):
    # create the base pre-trained model
    if cnn_model == 'Inception':
        base_model = InceptionV3(weights='imagenet', include_top=False)
    elif cnn_model == 'VGG':
        base_model = VGG16(weights='imagenet', include_top=False)
    elif cnn_model == 'ResNet':
        base_model = ResNet50(weights='imagenet', include_top=True)

    #medtadata
    metadata_input = Input(batch_shape=(None,5))

    # add a global spatial average pooling layer
    x = base_model.output
    if cnn_model != 'ResNet':
        x = GlobalAveragePooling2D()(x)
        #  add some fully-connected layers
        x = Dense(1024, activation='relu')(x)
    a = Concatenate()([x, metadata_input])

    x = Dense(1024, activation='relu')(a)
    x = Dense(512, activation='relu')(x)

    # and a logistic layer -- let's say we have 25 classes
    predictions = Dense(len(class_weights), activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=[base_model.input, metadata_input], outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                                 decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])#,
    return model

def load_data(filename):
    """ load data and class weights"""
    df= pd.read_csv(filename, index_col=0)
    X, y = df.index.values, df['classification'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
        random_state=42)
    X_validate, X_test, y_validate, y_test = train_test_split(X_test, y_test,
        test_size=0.50, random_state=42)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y_train)

    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    # Validation Dataset
    integer_encoded_validate = label_encoder.transform(y_validate)
    integer_encoded_validate = integer_encoded_validate.reshape(len(integer_encoded_validate), 1)
    onehot_encoded_validate = onehot_encoder.transform(integer_encoded_validate)

    # Class Weights
    class_weights = class_weight.compute_class_weight('balanced', \
        np.unique(np.array(integer_encoded)), integer_encoded[:,0])
    return X_train, X_validate, onehot_encoded, onehot_encoded_validate, class_weights


if __name__ == '__main__':
    # Load Data

    df_name = 'df_measurements_50.csv'
    X_train, X_validate, onehot_encoded, onehot_encoded_validate, class_weights = \
        load_data(df_name)

    validate_generator = imageLoader(X_validate, 100, onehot_encoded_validate, df_name)
    X_validate_generated, onehot_encoded_validate_generated = next(validate_generator)
    train_generator = imageLoader(X_train, 32, onehot_encoded, df_name)

    # Load Model
    if os.path.isfile('model_'+cnn_model+'.h5'):
        model = load_model('model_'+cnn_model+'.h5')
        # compile the model (should be done *after* setting layers to non-trainable)
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                                     decay=0.0, amsgrad=False)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])#,
    else:
        model = my_model(cnn_model, class_weights)

        # train the model on the new data for a few epoch
        model.fit_generator(train_generator, steps_per_epoch=100, epochs=10, \
            verbose=1, class_weight = class_weights,\
            validation_data=(X_validate_generated, onehot_encoded_validate_generated))

        #Save weights
        model.save('model_'+cnn_model+'.h5')

    # Unfreeze a number of CNN layers
    ntotal_layers = len(model.layers)
    nadded_layers = 7 #(7 plus 1)

    if cnn_model == 'Inception':
        nlayers = int(nblocks*31) # find out how many layers in a block
    elif cnn_model == 'VGG':
        nlayers = int(nblocks*4)
    elif cnn_model == 'ResNet':
        nlayers = int(nblocks*1)


    for layer in model.layers[:ntotal_layers - nadded_layers - nlayers]:
       layer.trainable = False
    for layer in model.layers[ntotal_layers - nadded_layers - nlayers:]:
       layer.trainable = True

    # compile the model (should be done *after* setting layers to non-trainable)
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None,
                                 decay=0.01, amsgrad=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model on the new data for a few epoch
    model.fit_generator(train_generator, steps_per_epoch=100, epochs=15, \
        verbose=1, class_weight = class_weights,\
        validation_data=(X_validate_generated, onehot_encoded_validate_generated))

    #Save weights
    model.save('model_'+cnn_model+'.h5')
