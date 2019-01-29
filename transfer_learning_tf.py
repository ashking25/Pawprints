import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import load_model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Input, Concatenate, Lambda
from keras import backend as K
import keras
from sklearn.utils import class_weight
from keras.callbacks import TensorBoard
from skimage import exposure
import sys
import tensorflow_hub as hub
import tensorflow as tf
hub_to_use="https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1"


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
cnn_model = sys.argv[1]
nblocks = float(sys.argv[2])


def imageLoader(files, batch_size, y_train, df_name):

    L = len(files)

    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            X = load_images_from_list(files[batch_start:limit], xpixels=224, ypixels=224)
            X = np.reshape(np.array(X), (np.shape(X)[0], np.shape(X)[1],
                                         np.shape(X)[2], np.shape(X)[3]))
            X_metadata = load_metadata(files[batch_start:limit])
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

    images = []
    for f in files:
        #load images
        img_path = '../'+f
        img = image.load_img(img_path, target_size=(xpixels, ypixels))
        x = image.img_to_array(img)
        x[np.isnan(x)]=0
        if 1 == 10:
            # don't need to smooth or flip right now
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

def load_metadata(files):
    """ Load information about size, location, and day of the year """
    results = []
    df= pd.read_csv('df_measurements_50.csv',index_col=0 )
    for f in files:
        if 'NonPawprints' in f:
            size = np.random.uniform(low=0.1, high=12)
            day_sin = np.random.uniform(low=0, high=2*np.pi)+1
            day_cos = np.random.uniform(low=0, high=2*np.pi)+1
            lat = 0
            long = 0
        else:
            f = f.split('/')[-1]
            f = np.int(f[:-4])
            size = np.random.uniform(low=df['min_track'].loc[f], high=df['max_track'].loc[f])#/12. # normalize to 1

            day_sin = np.sin(df['DayofYear'].loc[f]/365.*np.pi*2.)+1
            day_cos = np.cos(df['DayofYear'].loc[f]/365.*np.pi*2.)+1

            lat = df['latitude'].loc[f] / 360. # normalize to 1
            long = (360+df['longitude'].loc[f])/360. #normalize to 1

            if np.isnan(np.sum(lat)) or np.isnan(np.sum(long)):
                lat = 0
                long = 0
        results.append([size, day_sin, day_cos, lat, long])
    return results

def my_model(cnn_model, class_weights, target_width=224):
    # create the base pre-trained model

    #medtadata
    metadata_input = Input(batch_shape=(None,5))

    # add a global spatial average pooling layer
    graph = tf.get_default_graph()
    mobilenet_features_module = hub.Module(hub_to_use)

    input = Input(batch_shape=(None, target_width, target_width, 3))
    base_model = Lambda(Lambda(mobilenet_features_module))(input)
    a = Concatenate()([base_model, metadata_input])

    x = Dense(1024, activation='relu')(a)
    x = Dense(512, activation='relu')(x)

    # and a logistic layer -- let's say we have 25 classes
    predictions = Dense(len(class_weights), activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=[input, metadata_input], outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers[:-5]:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                                 decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])#,
    #model_total=Sequential()
    #model_total.add(Lambda(mobilenet_features_module, input_shape=(target_width, target_width,3)))

    #metadata_input = Sequential()
    #model_total.add([model_total, metadata_input])
    #model_total.add(Dense(len(class_weights), activation='softmax'))
    #sess = K.get_session()
    #init = tf.global_variables_initializer()
    #sess.run(init)
    #model_total.layers[1].set_weights(head_model.layers[0].get_weights())


    # compile the model (should be done *after* setting layers to non-trainable)
    #adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
#                                 decay=0.0, amsgrad=False)
    #model_total.compile(optimizer=adam, loss='categorical_crossentropy',
#                metrics=['accuracy'])
    return model#_total

def load_data(filename):
    """ load data and class weights"""
    names = np.loadtxt('../images_categories_names.txt', dtype='str', delimiter=' ')
    y = []
    x = []
    for n in names:
        y.append(n.split('/')[1])
        x.append(n.split('/')[-1])
    X_train, X_test, y_train, y_test = train_test_split(names,y, test_size=0.20, random_state=42)
    X_validate, X_test, y_validate, y_test = train_test_split(X_test, y_test, test_size=0.50, random_state=42)

    df= pd.read_csv(filename, index_col=0)

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
        print(model.summary())
        # train the model on the new data for a few epoch
        model.fit_generator(train_generator, steps_per_epoch=100, epochs=10, \
            verbose=1, class_weight = class_weights,\
            validation_data=(X_validate_generated, onehot_encoded_validate_generated))

        #Save weights
        model.save('model_'+cnn_model+'.h5')

    # Unfreeze a number of CNN layers
    for layer in model.layers:
       layer.trainable = True

    # compile the model (should be done *after* setting layers to non-trainable)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                                 decay=0.01, amsgrad=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model on the new data for a few epoch
    model.fit_generator(train_generator, steps_per_epoch=1000, epochs=15, \
        verbose=1, class_weight = class_weights,\
        validation_data=(X_validate_generated, onehot_encoded_validate_generated))

    #Save weights
    model.save('model_'+cnn_model+'.h5')
