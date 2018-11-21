# Signature Verification App


## Setup Environment on Local Machine

### Pre-requisites

```
conda create -n sigver -y python=3
source activate sigver
```
The following libraries are required

* Scipy version 0.18
* Pillow version 3.0.0
* OpenCV
* Theano<sup>2</sup>
* Lasagne<sup>2</sup>

### Downloading the models

```
git clone https://github.com/felice024/signature_verification
cd signature_verification/models
wget "https://storage.googleapis.com/luizgh-datasets/models/signet_models.zip"
unzip signet_models.zip

````

### Test App

Run the app
```
python run_server.py
```
Open Browser:  [http://localhost:9000](http://localhost:9000)


## Citation
[1] Hafemann, Luiz G., Robert Sabourin, and Luiz S. Oliveira. "Learning Features for Offline Handwritten Signature Verification using Deep Convolutional Neural Networks" http://dx.doi.org/10.1016/j.patcog.2017.05.012 ([preprint](https://arxiv.org/abs/1705.05787))

[2] Hafemann, Luiz G., Robert Sabourin, and Luiz S. Oliveira. "Fixed-sized representation learning from Offline Handwritten Signatures of different sizes" https://doi.org/10.1007/s10032-018-0301-6 ([preprint](https://arxiv.org/abs/1804.00448))

[3] https://github.com/luizgh/sigver_wiwd

[4] https://github.com/sampathweb/apparel-styles/tree/master/app

## Dataset:

GPDS: Vargas, J.F., M.A. Ferrer, C.M. Travieso, and J.B. Alonso. 2007. “Off-Line Handwritten Signature GPDS-960 Corpus.” In Document Analysis and Recognition, 9th I    nternational Conference on, 2:764–68. doi:10.1109/ICDAR.2007.4377018.


### The End.
=======
