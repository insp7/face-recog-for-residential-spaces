import tensorflow as tf
from keras.models import load_model
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from os import listdir
from os.path import isdir
from PIL import Image
from numpy import asarray
from numpy import savez_compressed
import os
import pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def extract_face(filename, required_size = (160, 160)):
	#load images
	image =  Image.open(filename)
	#Convert to RGB
	image = image.convert('RGB')
	#Convert to array
	pixels = asarray(image)
	#Create the detector, using default weights
	detector = MTCNN()
	#Detect faces in the image
	results = detector.detect_faces(pixels)
	#Extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]

	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

def load_faces(directory) :
    faces = list()
    # enumerate files
    for filename in listdir(directory) :
        #path
        path = directory + filename
        #get face
        face = extract_face(path)
        #store
        faces.append(face)
    
    return faces

def load_dataset(directory) :
    X, Y = list(), list()
    for subdir in listdir(directory) :
        #path
        path = directory + subdir + '/'
        
        #skip anyfiles that might be in the directory
        if not isdir(path) :
            continue
        
        #load all faces in the subdirectory
        faces = load_faces(path)

        #create labels
        labels = [subdir for _ in range(len(faces))]
        
        #summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        #store
        X.extend(faces)
        Y.extend(labels)
    
    return asarray(X),asarray(Y)

trainX, trainY = load_dataset('dataset/train/')
print(trainX.shape, trainY.shape)

testX, testY = load_dataset('dataset/test/')
print(testX.shape, testY.shape)

savez_compressed('models/compressed_arrays/dataset.npz', trainX, trainY, testX, testY)
# tf.logging.set_verbosity(tf.logging.ERROR)


# model = load_model('facenet_keras.h5')

# print(model.inputs)
# print(model.outputs)