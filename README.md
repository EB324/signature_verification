# Signature Verification 

This repository contains the code and instructions to use the trained CNN models described in [1] to extract features for Offline Handwritten Signatures and use SVM models to train writer-dependent classifiers.

It also contains a web application which can predict the genuineness of an uploaded signature picture compared to the one used for SVM model training.   

The web application frame is modified from https://github.com/sampathweb/apparel-styles/tree/master/app.

## Setup Environment

### Pre-requisites

Create a new environment

```
conda create -n sigver -y python=3
source activate sigver
```

The following libraries are required

* Scipy version 0.18
* Pillow version 3.0.0
* OpenCV
* Pandas
* Theano
* Lasagne
* Tornado >= 4.4
* Scikit-learn >= 0.17
* requests >= 2.10
* jupyter >= 1.0

They can be installed by the following commands

```
conda install -y "scipy=0.18.0" "pillow=3.0.0"
pip install opencv-python
pip install pandas
pip install "Theano==0.9"
pip install https://github.com/Lasagne/Lasagne/archive/master.zip
pip install tornado
pip install scikit-learn
pip install requests
pip install jupyter
````

### Download the pre-trained CNN models

```
git clone https://github.com/EB324/signature_verification
cd signature_verification/sigver/models
wget "https://storage.googleapis.com/luizgh-datasets/models/signet_models.zip"
unzip signet_models.zip
```

## Import dataset for SVM model trainning

* Put genuine signatures under ```sigver/trainsig/genuine/```  
* Put forged signatures under ```sigver/trainsig/forged/```

The model is writer-dependent and can be trained for one writer each time only.     
I have put some signatures (both genuine and forged) in them as a demo.

## Test the App

Create a temp-img folder 

```
cd signature_verification/app/static/
mkdir temp-img
```

Run the App

```
python run_server.py
```

Open Browser:  [http://localhost:9000](http://localhost:9000)  
Upload a signature and check its genuineness. You can use signatures in  ```sigver/testsig/``` as an example.

## References
[1] Hafemann, Luiz G., Robert Sabourin, and Luiz S. Oliveira. "Learning Features for Offline Handwritten Signature Verification using Deep Convolutional Neural Networks" http://dx.doi.org/10.1016/j.patcog.2017.05.012 ([preprint](https://arxiv.org/abs/1705.05787))

[2] Hafemann, Luiz G., Robert Sabourin, and Luiz S. Oliveira. "Fixed-sized representation learning from Offline Handwritten Signatures of different sizes" https://doi.org/10.1007/s10032-018-0301-6 ([preprint](https://arxiv.org/abs/1804.00448))

[3] https://github.com/luizgh/sigver_wiwd

[4] https://github.com/sampathweb/apparel-styles/tree/master/app

## Dataset
GPDS: Vargas, J.F., M.A. Ferrer, C.M. Travieso, and J.B. Alonso. 2007. “Off-Line Handwritten Signature GPDS-960 Corpus.” In Document Analysis and Recognition, 9th International Conference on, 2:764–68. doi:10.1109/ICDAR.2007.4377018.

http://www.cedar.buffalo.edu/NIJ/data/signatures.rar
