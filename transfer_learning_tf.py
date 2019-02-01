import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
from keras.preprocessing import image
from keras.models import Model
from keras.models import load_model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Input, Concatenate, Lambda, BatchNormalization
from keras import backend as K
import keras
from sklearn.utils import class_weight
from keras.callbacks import TensorBoard
from skimage import exposure
import sys
#import tensorflow_hub as hub
#import tensorflow as tf
#hub_to_use="https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1"
from imblearn.over_sampling import RandomOverSampler 

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
cnn_model = sys.argv[1]
cnn_name = sys.argv[2]

if cnn_model == 'Inception':
    from keras.applications.inception_v3 import preprocess_input
elif cnn_model == 'VGG':
    from keras.applications.vgg16 import preprocess_input
elif cnn_model == 'ResNet':
    from keras.applications.resnet50 import preprocess_input

def imageLoader(files, base_model, batch_size, y_train, df_name):

    L = len(files)

    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            X = load_images_from_list(files[batch_start:limit], base_model, xpixels=500, ypixels=500)
            #X = load_images_npy(files[batch_start:limit])
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
        x = np.load('../'+f[:-4]+'_incept.npy')#/images_footprints/'+str(int(f))+'.npy')
        #x = np.rot90(x,k=np.random.choice(range(4)))
        #if np.random.choice(range(2)) == 1:
        #    x = np.transpose(x, axes=[1,0,2])
        #x = np.expand_dims(x, axis=0)
        images.append(x)
    return images

def load_images_from_list(files, base_model, xpixels=299, ypixels=299):

    images = []
    if cnn_model == 'ResNet':
        xpixels = 224
        ypixels = 224
    for f in files:
        #load images
        img_path = '../'+f
        img = image.load_img(img_path)#, target_size=(xpixels, ypixels))
        x = image.img_to_array(img)
        x[np.isnan(x)]=0
        #if 1 == 1:
            # don't need to smooth or flip right now
        x = x/np.max(x)
        x = exposure.adjust_gamma(x, gamma=1./2.2, gain=1)*256
        
            #x = exposure.equalize_adapthist(x/np.max(x), clip_limit=0.03)*256
            #x = np.rot90(x,k=np.random.choice(range(4)))
            #if np.random.choice(range(2)) == 1:
            #    x = np.transpose(x, axes=[1,0,2])

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
        #predict = base_model.predict(x)
        images.append(x[0])
    images_predict = base_model.predict(np.array(images))
    
    return images_predict

def load_metadata(files):
    """ Load information about size, location, and day of the year """
    results = []
    df= pd.read_csv('../df_measurements_50.csv',index_col=0 )
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

            size = np.random.normal(loc=(df['min_track'].loc[f]+df['max_track'].loc[f])/2., 
                                    scale=(df['max_track'].loc[f]-df['min_track'].loc[f])/2./np.sqrt(2.))#/12. # normalize to 1
            day_sin = np.sin(df['DayofYear'].loc[f]/365.*np.pi*2.)+1
            day_cos = np.cos(df['DayofYear'].loc[f]/365.*np.pi*2.)+1

            lat = df['latitude'].loc[f] / 360. # normalize to 1
            long = (360+df['longitude'].loc[f])/360. #normalize to 1

            if np.isnan(np.sum(lat)) or np.isnan(np.sum(long)):
                lat = 0
                long = 0
            if np.random.uniform(0, 1) < .10:
                lat = 0
                long = 0
        results.append([size, day_sin, day_cos, lat, long])
    return results

def my_model(cnn_model, class_weights, target_width=224):
    # create the base pre-trained model
    if cnn_model == 'Inception':
        x_input = Input(batch_shape=(None, None, None, 2048))
    elif cnn_model == 'VGG':
        x_input = Input(batch_shape=(None, None, None, 512))
    elif cnn_model == 'ResNet':
        x_input = Input(batch_shape=(None, None, None, 2048))
    #medtadata
    metadata_input = Input(batch_shape=(None,5))

    # add a global spatial average pooling layer
    #x_input = Input(batch_shape=(None, None, None, 2048))
    #x = base_model.output
    print('shape',np.shape(X_validate_generated[0]))
    
    #if cnn_model != 'ResNet':
    x = GlobalAveragePooling2D()(x_input)
    #  add some fully-connected layers
    if cnn_name == 'First':
        predictions = Dense(len(class_weights), activation='softmax')(x)
    else:
        a = Concatenate()([x, metadata_input])
        predictions = Dense(len(class_weights), activation='softmax')(a)

    # this is the model we will train
    model = Model(inputs=[x_input, metadata_input], outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    #for layer in base_model.layers:
    #    layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    adam = keras.optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=None,
                                 decay=0.0, amsgrad=False)
#    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])#,

    model.compile(optimizer=adam, loss='kullback_leibler_divergence', metrics=['accuracy'])
    return model

def load_data(filename):
    """ load data and class weights"""
    names = np.loadtxt('../images_categories_names.txt', dtype='str', delimiter=' ')
    names_top5 = []
    top5 = ['Raccoon', 'Coyote', 'Deer', 'Bobcat', 'Bear']
    for n in names:
        if set(set(n.split('/')) & set(top5)):
            names_top5.append(n)
    y = []
    x = []
    for n in names_top5:
        y.append(n.split('/')[1])
        x.append(n.split('/')[-1])
    X_train, X_test, y_train, y_test = train_test_split(names_top5, y, test_size=0.20, random_state=42)
    X_validate, X_test, y_validate, y_test = train_test_split(X_test, y_test, test_size=0.50, random_state=42)

    print('here')
#    ros = RandomOverSampler(random_state=42, sampling_strategy='not majority')
#    X_train, y_train= ros.fit_sample(np.reshape(X_train, (len(X_train), 1)), y_train)
#    X_train = X_train[:,0]
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

    df_name = '../df_measurements_50.csv'

    X_train, X_validate, onehot_encoded, onehot_encoded_validate, class_weights = \
        load_data(df_name)
    # Load pretrained models
    # create the base pre-trained model
    if cnn_model == 'Inception':
        base_model = InceptionV3(weights='imagenet', include_top=False)
    elif cnn_model == 'VGG':
        base_model = VGG16(weights='imagenet', include_top=False)
    elif cnn_model == 'ResNet':
        base_model = ResNet50(weights='imagenet', include_top=False)    
    print('here')
    validate_generator = imageLoader(X_validate, base_model, 5, onehot_encoded_validate, df_name)
    X_validate_generated, onehot_encoded_validate_generated = next(validate_generator)
    for i in range(60):
        print('validate', i)
        # add more validation data
        X_validate_generated_extra, onehot_encoded_validate_generated_extra = next(validate_generator)
        X_validate_generated[0]= np.append(X_validate_generated[0], X_validate_generated_extra[0], axis=0)
        X_validate_generated[1]= np.append(X_validate_generated[1], X_validate_generated_extra[1], axis=0)
        onehot_encoded_validate_generated= np.append(onehot_encoded_validate_generated, 
                                                 onehot_encoded_validate_generated_extra, axis=0)
    train_generator = imageLoader(X_train, base_model, 8, onehot_encoded, df_name)
    print('built generators')
    # Load Model
    if os.path.isfile('../model_'+cnn_model+'_'+cnn_name+'.h5'):
        model = load_model('../model_'+cnn_model+'_'+cnn_name+'.h5')
        # compile the model (should be done *after* setting layers to non-trainable)
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                                     decay=0.0, amsgrad=False)
#        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])#,
        model.compile(optimizer=adam, loss='kullback_leibler_divergence', metrics=['accuracy'])
    else:
        model = my_model(cnn_model, class_weights)
        #print(final_model.summary())
        # train the model on the new data for a few epoch
        model.fit_generator(train_generator, steps_per_epoch=150, epochs=5, \
            verbose=1, class_weight = class_weights,\
            validation_data=(X_validate_generated, onehot_encoded_validate_generated))

        #Save weights
        model.save('../model_'+cnn_model+'_'+cnn_name+'.h5')

    # Unfreeze a number of CNN layers
    for layer in model.layers:
       layer.trainable = True

    # compile the model (should be done *after* setting layers to non-trainable)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                                 decay=0.01, amsgrad=False)
    #model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=adam, loss='kullback_leibler_divergence', metrics=['accuracy'])

    #callbacks
    callbacks = keras.callbacks.ModelCheckpoint('../model_'+cnn_model+'_'+cnn_name+'.h5', monitor='val_loss',
        verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=5)


    # train the model on the new data for a few epoch
    model.fit_generator(train_generator, steps_per_epoch=150, epochs=15, \
        verbose=2, class_weight = class_weights,\
        validation_data=(X_validate_generated, onehot_encoded_validate_generated),\
        callbacks=[callbacks])

    np.save('model_history_'+cnn_model+'_'+cnn_name, model.history.history)
    #Save weights
    #model.save('model_'+cnn_model+'.h5')
