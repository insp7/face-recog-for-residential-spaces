from numpy import *
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


data = load('models/compressed_arrays/dataset.npz')
trainX, trainY, testX, testY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded : ', trainX.shape, trainY.shape, testX.shape, testY.shape)

#Load the model
model = load_model('models/facenet_keras.h5')
print('Facenet Model Loaded')

model.load_weights('models/facenet_keras_weights.h5')
print('weights loaded')

#Convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX :
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)

newTrainX = asarray(newTrainX)
print(newTrainX.shape)

#Convert each face in the test set to an embedding
newTestX = list()
for face_pixels in testX :
    embedding = get_embedding(model, face_pixels)
    newTestX.append(embedding)

newTestX = asarray(newTestX)
print(newTestX.shape)

#save arrays to one file in compressed format
savez_compressed('models/compressed_arrays/face-embeddings.npz', newTrainX, trainY, newTestX, testY)





