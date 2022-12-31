import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
from keras.models import load_model
from PIL import Image
import pathlib
import tensorflow as tf

def get_embedding(model, face_pixels) :
    #scale pixel values
    face_pixels = face_pixels.astype('float32')
    #standardize pixel value across channels (globals)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    #transform face into one sample
    samples = expand_dims(face_pixels, axis = 0)
    #make prediction to make embedding
    yhat = model.predict(samples)
    return yhat[0]

#Load the model
keras_model = load_model('models/facenet_keras.h5')
keras_model.load_weights('models/facenet_keras_weights.h5')

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# load faces
data = load(pathlib.Path('models/compressed_arrays/dataset.npz'))
testX_faces = data['arr_2']

# load face embeddings
data = load(pathlib.Path('models/compressed_arrays/face-embeddings.npz'))

trainX, trainY, testX, testY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainY)
trainY = out_encoder.transform(trainY)
testY = out_encoder.transform(testY)

# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainY)
selection = choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
random_face_emb = testX[selection]
random_face_class = testY[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])

samples = expand_dims(random_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)                
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)
# get name
class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))

pyplot.imshow(random_face_pixels)
title = '%s (%.3f)' % (predict_names[0], class_probability)
pyplot.title(title)
pyplot.show()
