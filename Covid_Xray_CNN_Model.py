#Developing the CNN model
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

#Importing the Necessary Libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16,VGG19,InceptionV3,InceptionResNetV2,Xception,DenseNet201, ResNet50
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import shutil
import cv2
import os

dataset_path = './dataset'

%%bash
rm -rf dataset
mkdir -p dataset/covid
mkdir -p dataset/normal

covid_dataset_path = '../input/covid-chest-xray'

csvPath = os.path.sep.join([covid_dataset_path, "metadata.csv"])
df = pd.read_csv(csvPath)

# loop over the rows of the COVID-19 data frame
for (i, row) in df.iterrows():
    # if (1) the current case is not COVID-19 or (2) this is not
    # a 'PA' view, then ignore the row
    if row["finding"] != "COVID-19" or row["view"] != "PA":
        continue

    # build the path to the input image file
    imagePath = os.path.sep.join([covid_dataset_path, "images", row["filename"]])

    # if the input image file does not exist (there are some errors in
    # the COVID-19 metadeta file), ignore the row
    if not os.path.exists(imagePath):
        continue

    # extract the filename from the image path and then construct the
    # path to the copied image file
    filename = row["filename"].split(os.path.sep)[-1]
    outputPath = os.path.sep.join([f"{dataset_path}/covid", filename])

    # copy the image
    shutil.copy2(imagePath, outputPath)


basePath = os.path.sep.join([pneumonia_dataset_path, "train", "NORMAL"])
imagePaths = list(paths.list_images(basePath))

# randomly sample the image paths
random.seed(42)
random.shuffle(imagePaths)
imagePaths = imagePaths[:]

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the filename from the image path and then construct the
    # path to the copied image file
    filename = imagePath.split(os.path.sep)[-1]
    outputPath = os.path.sep.join([f"{dataset_path}/normal", filename])

    # copy the image
    shutil.copy2(imagePath, outputPath)


def ceildiv(a, b):
    return -(-a // b)

def plots_from_files(imspaths, figsize=(10,5), rows=1, titles=None, maintitle=None):
    """Plot the images in a grid"""
    f = plt.figure(figsize=figsize)
    if maintitle is not None: plt.suptitle(maintitle, fontsize=10)
    for i in range(len(imspaths)):
        sp = f.add_subplot(rows, ceildiv(len(imspaths), rows), i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        img = plt.imread(imspaths[i])
        plt.imshow(img)

normal_images = list(paths.list_images(f"{dataset_path}/normal"))
covid_images = list(paths.list_images(f"{dataset_path}/covid"))

print(f"Length of Covid19 Samples: {len(covid_images)}")
print(f"Length of Normal Samples: {len(normal_images)}")

plots_from_files(normal_images[:5], rows=1, maintitle="Normal X-ray images")

plots_from_files(covid_images[:10], rows=2, maintitle="Covid-19 X-ray images")

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-3
EPOCHS = 100
BS = 8

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset_path))
data = []
labels = []
# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]
    # load the image, swap color channels, and resize it to be a fixed
    # 224x224 pixels while ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 1]
data = np.array(data) / 255.0
labels = np.array(labels)

##############################################Functional Approach#########################

def cnn_model(batch_size,epoch):
    # load the VGG16 network, ensuring the head FC layer sets are left
    # off
    baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(64, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)
    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)
    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False
    
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    return model


lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

def fit_and_evaluate(t_x, val_x, t_y, val_y, EPOCHS=100, BATCH_SIZE=8):
    model = None
    model = cnn_model(BATCH_SIZE, EPOCHS)
    # perform one-hot encoding on the labels
    # perform one-hot encoding on the labels
    
    trainAug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")
    results = model.fit_generator(
                    trainAug.flow(t_x, t_y, batch_size=BS),
                    steps_per_epoch=len(val_x) // BS,
                    validation_data=trainAug.flow(val_x, val_y, batch_size=BS),
                    validation_steps=len(val_x) // BS,
                    epochs=EPOCHS)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax = ax.ravel()
    for i, met in enumerate(['accuracy', 'loss']):
        ax[i].plot(results.history[met])
        ax[i].plot(results.history['val_' + met])
        ax[i].set_title('Model {}'.format(met))
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(met)
        ax[i].legend(['train', 'val'])
        ax[i].savefig('pic'+i+'.svg',format='svg',dpi=1200)
    
    prediction = model.predict(val_x, batch_size=BS)
    test_result = np.argmax(val_y, axis=1)
    prediction_result = np.argmax(prediction, axis=1)
    confusion__matrix=confusion_matrix(test_result, prediction_result)
    print(classification_report(test_result, prediction_result, target_names=lb.classes_))
    print(confusion__matrix)
    
    print("Val Score: ", model.evaluate(val_x, val_y))
    return results

n_folds=5
epochs=100
batch_size=8

model_history = [] 

for i in range(n_folds):
    print("Training on Fold: ",i+1)
    t_x, val_x, t_y, val_y = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state = np.random.randint(1,1000, 1)[0])
    model_history.append(fit_and_evaluate(t_x, val_x, t_y, val_y, epochs, batch_size))
    print("======="*12, end="\n\n\n")

plt.figure(figsize=(20,10))
plt.title('Accuracies vs Epochs')
plt.plot(model_history[0].history['accuracy'], label='Training Fold 1')
plt.plot(model_history[1].history['accuracy'], label='Training Fold 2')
plt.plot(model_history[2].history['accuracy'], label='Training Fold 3')
plt.plot(model_history[3].history['accuracy'], label='Training Fold 4')
plt.plot(model_history[4].history['accuracy'], label='Training Fold 5')
plt.legend()
plt.savefig('fold_accuracy.svg',format='svg',dpi=1200)
plt.show()

plt.figure(figsize=(20,10))
plt.title('Train Accuracy vs Val Accuracy')
plt.plot(model_history[0].history['accuracy'], label='Train Accuracy Fold 1', color='black')
plt.plot(model_history[0].history['val_accuracy'], label='Val Accuracy Fold 1', color='black', linestyle = "dashdot")
plt.plot(model_history[1].history['accuracy'], label='Train Accuracy Fold 2', color='red', )
plt.plot(model_history[1].history['val_accuracy'], label='Val Accuracy Fold 2', color='red', linestyle = "dashdot")
plt.plot(model_history[2].history['accuracy'], label='Train Accuracy Fold 3', color='green', )
plt.plot(model_history[2].history['val_accuracy'], label='Val Accuracy Fold 3', linestyle = "dashdot")
plt.plot(model_history[2].history['accuracy'], label='Train Accuracy Fold 4', )
plt.plot(model_history[2].history['val_accuracy'], label='Val Accuracy Fold 4', color='green', linestyle = "dashdot")
plt.plot(model_history[2].history['accuracy'], label='Train Accuracy Fold 5', color='blue', )
plt.plot(model_history[2].history['val_accuracy'], label='Val Accuracy Fold 5', color='blue', linestyle = "dashdot")
plt.legend()
plt.savefig('fold_accuracy_val.svg',format='svg',dpi=1200)
plt.show()


################################VGG16 VS Fine Tuned VGG16 + Transfer Learning #####################
# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)
# initialize the training data augmentation object
trainAug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")

#Normal VGG16 Architecture
# load the VGG16 network, ensuring the head FC layer sets are left
# off
baseModel = VGG16(weights="imagenet",input_tensor=Input(shape=(224, 224, 3)))
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
#headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
#headModel = Flatten(name="flatten")(headModel)
#headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model1 = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

model1.summary()


# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model1.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model1.fit_generator(
    trainAug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)


#Fine Tuned VGG16 + transfer learning
# load the VGG16 network, ensuring the head FC layer sets are left
# off
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
tuned_model = model.fit_generator(
    trainAug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

model.save_weights('covidvgg16_64.h5')

results_tuned = tuned_model #Tuned VGG16
results = H #VGG16

import matplotlib.pyplot as plt
def plot_acc_loss(results, epochs):
    #Normal VGG16 train accuracy and loss
    
    vgg16_acc = results.history['accuracy']
    vgg16_loss = results.history['loss']
    #Tuned VGG16 train accuracy and loss
    tuned_vgg16_acc = results_tuned.history['accuracy']
    tuned_vgg16_loss = results_tuned.history['loss']
    #Normal VGG16 validation accuracy and loss
    vgg16_val_acc = results.history['val_accuracy']
    vgg16_val_loss = results.history['val_loss']
    #Tuned VGG16 validation accuracy and results
    tuned_vgg16_val_acc = results_tuned.history['val_accuracy']
    tuned_vgg16_val_loss = results_tuned.history['val_loss']
    
    
    fig = plt.figure(figsize=(15, 5))
    fig.patch.set_facecolor('xkcd:mint green')
    plt.subplot(121)
    #Normal VGG16 results
    plt.plot(range(1,epochs), vgg16_acc[1:], label='Train_acc - VGG16')
    plt.plot(range(1,epochs), vgg16_val_acc[1:], label='Test_acc - VGG16')
    #Tuned VGG16 results
    plt.plot(range(1,epochs), tuned_vgg16_acc[1:], label='Train_acc - Tuned VGG16 + Transfer learning')
    plt.plot(range(1,epochs), tuned_vgg16_val_acc[1:], label='Test_acc - Tuned VGG16 + Transfer learning')
    plt.title('Accuracy over ' + str(epochs) + ' Epochs', size=15)
    
    plt.legend()
    plt.savefig('loss_compare1.svg',format='svg',dpi=1200)
    plt.grid(True)
    plt.subplot(122)
    
    plt.plot(range(1,epochs), vgg16_loss[1:], label='Train_loss -VGG16')
    plt.plot(range(1,epochs), vgg16_val_loss[1:], label='Test_loss -VGG16')
    
    plt.plot(range(1,epochs), tuned_vgg16_loss[1:], label='Train_loss - Tuned VGG16 + Transfer learning')
    plt.plot(range(1,epochs), tuned_vgg16_val_loss[1:], label='Test_loss - Tuned VGG16 + Transfer learning')
    
    plt.title('Loss over ' + str(epochs) +  ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_compare.svg',format='svg',dpi=1200)
    plt.show()
    
 

plot_acc_loss(results, 100)
    