import sys
import pandas as pd 
import numpy as np
import sklearn
import matplotlib
import keras
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import Dense 
from keras.optimizers import Adam


def heartDiseasePredict():
    data = pd.read_csv('heart.csv')
    data = data.apply(pd.to_numeric)
    X = np.array(data.drop(['target'], 1))
    y = np.array(data['target'])

    X_train, X_test, y_train,y_test = model_selection.train_test_split(X,y,test_size=0.2)

    # Convert traget to categorical
    Y_train = to_categorical(y_train,num_classes=None)
    Y_test = to_categorical(y_test,num_classes=None)
    print(Y_train.shape)
    print(Y_train[:10])

    # define a function to build the keras model
    def create_model():
        # create model
        model = Sequential()
        model.add(Dense(16,input_dim=13,kernel_initializer='normal',activation='relu'))
        model.add(Dense(8, kernel_initializer='normal',activation='relu'))
        model.add(Dense(2,activation='softmax'))
        
        # compile model
        adam = Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy',optimizer=adam, metrics=['accuracy'])

        return model

    model = create_model()
    print(model.summary())


    history = model.fit(X_train,Y_train, validation_data=(X_test,Y_test), epochs=200, batch_size=10, verbose=10)


    # Model accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'])
    plt.show()


    # Model loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'])
    plt.show()

    # generate classification report using predictions for categorical model
    categorical_pred = np.argmax(model.predict(X_test),axis=1)
    print('Result for Categorical Model')
    print(accuracy_score(y_test, categorical_pred))
    print(classification_report(y_test, categorical_pred))


    # predict 
    print("Predict New Patient : ",np.argmax(model.predict(np.array([[25,1,3,200,233,1,0,120,0,2.3,0,0,2]])), axis=1 ))
