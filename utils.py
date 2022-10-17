import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

IMG_SIZE = 224

def label_img(img):
    word_label = img.split('_')[0]
    if word_label == 'covid': return 1
    elif word_label == 'noncovid': return 0
    
    
def createDataSplitSet(datapath):
    X=[]
    y=[]

    for img in os.listdir(datapath):
        label = label_img(img) #labeling image and returning 0 if noncovid and 1 if covid
        path = os.path.join(datapath, img) 
        image = cv2.resize(cv2.imread(path), (IMG_SIZE, IMG_SIZE)) #resizing image
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) #normalizing image
        X.append(np.array(image)) #appending image to list
        y.append(label) #appending labels to list

    return np.array(X), np.array(y) #returning images, and labels


def plot_train_data(img):
    #displaying a random image
    plt.imshow(img[2849], interpolation='nearest')
    plt.show()
    

def plot_results(history):
    #plotting accuracy and loss per epoch
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
    
    return acc, val_acc, loss, val_loss
    
    

def get_classification_report(model, test_X, test_y, acc, val_acc):
    ## Test Accuracy
    predictions = model.predict(test_X)
    ypred = predictions > 0.5
    test_acc = accuracy_score(test_y, ypred)

    

    precision, recall, f1score, _ = precision_recall_fscore_support(test_y, ypred, average='binary')

    auc = roc_auc_score(test_y, ypred)

    print("Train Accuracy:\t", acc[-1])
    print("Val Accuracy:\t", val_acc[-1])
    print("Test Accuracy:\t", test_acc)
    print("Precision:\t", precision)
    print("Recall:\t\t", recall)
    print("F1 Score:\t", f1score)
    print("AUC:\t\t", auc)
    
    
    
def compile_train_get_results(model, train_X, train_y, val_X, val_y, test_X, test_y):
    #compiling model
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['acc'])


    #training model with batch size 10 and 100 epochs
    history = model.fit(train_X, train_y, batch_size=10, epochs=100, validation_data=(val_X, val_y))
    acc, val_acc, loss, val_loss = plot_results(history)
    get_classification_report(model, test_X, test_y, acc, val_acc)