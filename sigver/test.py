import signet
import os
import re
from cnn_model import CNNModel
from scipy.misc import imread
from preprocess.normalize import preprocess_signature
from sklearn import svm, datasets
from sklearn.utils import shuffle

canvas_size = (952, 1360)  # Maximum signature size

# Load the trained model
model_weight_path = 'models/signet.pkl'
model = CNNModel(signet, model_weight_path)

# Define a function to get features from processed signature pictures
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

# Load the signature picture
sig_list = os.listdir('trainsig/')
sig_list.remove('.DS_Store')
sig_feature_list = []
sig_label_list = []

for sig in sig_list:
    #Pre-process the signature
    original = imread('trainsig/'+sig, flatten = 1)

    #Retrieve the features of the signature
    sig_feature_list.append(get_feature(original))

    #Label the signature
    if re.match('.*t.*', sig):
        sig_label_list.append('T')
    else:
        sig_label_list.append('F')

#Test
test_sig_list = os.listdir('testsig/')
test_sig_list.remove('.DS_Store')
test_sig_feature_list = [get_feature(imread('testsig/'+testsig, flatten=1)) for testsig in test_sig_list]

print(test_sig_list)
print(SVM_Train(sig_feature_list, sig_label_list).predict(test_sig_feature_list))