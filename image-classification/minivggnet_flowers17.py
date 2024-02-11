from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.nn.conv import MiniVGGNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# constructing the argument parser
ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required="True",help="path to image dataset")
args=vars(ap.parse_args())
#next the class labels need to be extracted
# -----LOADING-----
print("[INFO] loading images...")
# the dataset folder has the structure
# flowers17
# |-species
# |--{images}
# hence using [-2] we can easily obtain the classnames
imagePaths=list(paths.list_images(args["dataset"]))
classNames=[pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames=[str(x) for x in np.unique(classNames)]
# now we initialize the preprocessors
# aspect aware is intellegent resizing along whichever axis is lacking
# image to array preprocessor is for converting the image into an array
# using img_to_array() function from keras.preprocessing.image
aap=AspectAwarePreprocessor(64,64)
iap=ImageToArrayPreprocessor()
# now we load the raw dataset from the disk
# and instantiate them using these two preprocessors
sdl=SimpleDatasetLoader(preprocessors=[aap,iap])
# the data and labels are loaded next
(data,labels)=sdl.load(imagePaths,verbose=500)
# next the data is normalized within the range [0,1]
data=data.astype("float")/255.0
# the data is then split into training set and validation set
# with 25% of the data kept aside for testing
(trainX, testX, trainY, testY)=train_test_split(data,labels,test_size=0.25,random_state=42)
# converting the labels from integers to vectors
# if we do not mention the random_state=<some_integer> then each time we run the code
# we get a different output
trainY=LabelBinarizer().fit_transform(trainY)
testY=LabelBinarizer().fit_transform(testY)
# -----TRAINING-----
# to train our algorithm we will be using the MiniVGGNet and SGD optimizer
# so we compile the model
print("[INFO] compiling the model...")
opt=SGD(lr=0.05)
model=MiniVGGNet.build(width=64,height=64,depth=3,classes=len(classNames))
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
# training the model
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	batch_size=32, epochs=100, verbose=1)
# now we wait
# sorry..
# now we evaluate the network
print("[INFO] evaluating the network...")
predictions=model.predict(testX,batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=classNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()