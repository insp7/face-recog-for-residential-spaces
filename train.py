#develop a classifier for the dataset
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# load dataset
data = load('models/compressed_arrays/face-embeddings.npz')
trainX, trainY, testX, testY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))

# normalize input vector
in_encoder = Normalizer(norm = 'l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainY)
# print(out_encoder.classes_)
trainY = out_encoder.transform(trainY)
testY = out_encoder.transform(testY)

# fit model
model = SVC(kernel = 'linear', probability = True)
model.fit(trainX, trainY)

# predict
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)

# score
score_train = accuracy_score(trainY, yhat_train)
score_test = accuracy_score(testY, yhat_test)

# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

# # saving the model
# file_name = 'trained_model.sav'
# f = open(file_name, 'wb')
# pickle.dump(model, f)
# pickle.dump(in_encoder, f)
# pickle.dump(out_encoder, f)