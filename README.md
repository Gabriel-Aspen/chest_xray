# chest_xray

Deep Learning approach to pneumonia diagnosis

### Preprocessing
For computational efficiency the images are resized to 200 x 200 px, for XCeption model they are sized to 224 x 224. Image reshaping and augmentation performed with OpenCV and ImgAug libraries

## Training

### Simple convolutional neural network
A convolutional neural network consisting of 3 convolutional layers and one dense layer with Tensorflow 2.6 default parameters. Trained for 50 epochs achieving a validation AUC score of 0.8987


### XCeption model by Google
Transfer learning was performed on an XCeption model pretrained on the imagenet dataset. The top layer was replaced with a 2 node dense layer representing our positive and negative classes. Learning rate was increased and bottom layers were frozen for 5 epoch. All layers were unfrozen and the learning rate was decreased and the model was trained for 40 epochs
