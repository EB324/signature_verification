import sigver.signet as signet
import os
import re
from sigver.cnn_model import CNNModel
from scipy.misc import imread
from sigver.preprocess.normalize import preprocess_signature
from sklearn import svm, datasets
from sklearn.utils import shuffle

# Maximum signature size
canvas_size = (952, 1360)

# Load the trained user-independent model
model_weight_path = 'sigver/models/signet.pkl'
model = CNNModel(signet, model_weight_path)

# Define a function to get features from loaded signature pictures
def get_feature(sig_img):
    processed = preprocess_signature(sig_img, canvas_size)
    #Use the CNN to extract features
    feature_vector = model.get_feature_vector(processed)
    feature_vector = feature_vector[0]
    return feature_vector

# Define a function to train user-dependent model
def SVM_Train(X_train, y_train):
    X, y = shuffle(X_train,y_train, random_state=2)
    clf = svm.SVC(C=1, class_weight='balanced', kernel='linear', probability=True)
    clf.fit(X, y)  
    return clf

# Define a function to train a set of signature pictures
def train(dir):
    genuine_sig_list = os.listdir(dir+'genuine/')
    forged_sig_list = os.listdir(dir+'forged/')
    sig_feature_list = []
    sig_label_list = []

    for sig in genuine_sig_list:
        if sig == '.DS_Store':
            genuine_sig_list.remove('.DS_Store')
        else:
            #Load the signature
            original = imread(dir+'genuine/'+sig, flatten = 1)

            #Retrieve features of the signature
            sig_feature_list.append(get_feature(original))

            #Label the signature
            sig_label_list.append('Genuine')
    
    for sig in forged_sig_list:
        if sig == '.DS_Store':
            forged_sig_list.remove('.DS_Store')
        else:
            #Load the signature
            original = imread(dir+'forged/'+sig, flatten = 1)

            #Retrieve features of the signature
            sig_feature_list.append(get_feature(original))

            #Label the signature
            sig_label_list.append('Forged')

    return SVM_Train(sig_feature_list, sig_label_list)