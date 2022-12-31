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
from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# load faces
data = load('models/compressed_arrays/dataset.npz')
testX_faces = data['arr_2']

# load face embeddings
data = load('models/compressed_arrays/face-embeddings.npz')
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

# adding the labels
for i in out_encoder.classes_ :
    if not os.path.exists('frames/'+i) :
        os.makedirs('frames/'+i)

# adding unknown
if not os.path.exists('frames/unknowns') :
    os.makedirs('frames/unknowns')

# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainY)

#creating video capture
video = cv2.VideoCapture(0)
while True:
    ret_val, frame = video.read()
    frame = cv2.flip(frame, 1)
    #converting frame to numpy array of pixels
    pixels = np.asarray(frame)

    #creating  detector using MTCNN
    detector = MTCNN()

    #getting results from face
    results = detector.detect_faces(pixels)

    face_embeddings = list()
    let_go = True
    for i in results :
        x1, y1, width, height = i['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, x1 + width
        face = pixels[y1:y2, x1:x2]
        # extract the face
        try:
            image = Image.fromarray(face)
        except:
            let_go = False
        
        image = image.resize((160, 160))
        face_array = np.asarray(image)
        # create embeddings
        face_embeddings.append(get_embedding(keras_model, face_array))

    if let_go :
        _ids = list()
        for i in face_embeddings :
            samples = expand_dims(i, axis=0)
            yhat_class = model.predict(samples)
            yhat_prob = model.predict_proba(samples)                
            yhat_class = model.predict(samples)
            yhat_prob = model.predict_proba(samples)

            # get name and probabiltiy
            class_index = yhat_class[0]
            class_probability = yhat_prob[0,class_index] * 100
            predict_names = out_encoder.inverse_transform(yhat_class)
            print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
            if class_probability > 50 :
                _ids.append(predict_names[0])
            else :
                _ids.append("unknowns")

        current_timestamp = datetime.now()
        for i in _ids : 
            cv2.imwrite('frames/'+ i + '/' + str(current_timestamp) +'.jpeg', frame)

        # cv2.imshow('window', frame)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit

cv2.destroyAllWindows()

# GET : /inhome/id