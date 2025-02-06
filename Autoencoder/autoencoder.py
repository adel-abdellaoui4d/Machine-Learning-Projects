from sklearn.datasets import make_regression

# generate the data 
X,Y = make_regression(n_samples=100,n_features=150,n_informative=10,noise=0.1,random_state=1)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error


#splitting to train & test data
X_train,X_test,Y_train,Y_test= train_test_split(Y,Y,test_size=.34,random_state=1)

process = MinMaxScaler()
process.fit(X_train)
X_train=process.transform(X_train)
X_test=process.transform(X_test)

#ENCODER
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, ReLU,BatchNormalization

inputs = X_test.shape[1]

# define the encoder
encoder_input = Input(shape=(inputs,))
encoder_layer = Dense(inputs*2)(encoder_input)
encoder_layer = BatchNormalization()(encoder_layer)
encoder_layer = ReLU()(encoder_layer)

# define the end of the encoder : bottleNeck
n_bottleNeck = inputs
bottleNeck = Dense(n_bottleNeck)(encoder_layer)

#define the decoder
decoder = Dense(inputs*2)(bottleNeck)
decoder = BatchNormalization()(decoder)
decoder = ReLU()(decoder)
decoder_output = Dense(inputs,activation='linear')(decoder)

# define the autoencoder
autoencoder = Model(inputs=encoder_input,outputs=decoder_output)
# Compiling & configiration
autoencoder.compile(optimizer='adam',loss='mse')

# Plot the Autoencoder
from tensorflow.keras.utils import plot_model
plot_model(autoencoder,'auto.jpg')


# train the autoencoder
history = autoencoder.fit(X_train,X_train,epoch=500,batch_size=20,valdation_data=(X_test,X_test))

from matplotlib import pyplot as plt

plt.title('The losses')
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
plt.legend()
plt.show()

#Split the encoder from the autoencoder
e2 = Model(inputs=encoder_input,outputs=bottleNeck)

#Plot
plot_model(e2,'e2.jpg',show_shapes=True)

# Save the encoder
e2.save('encoder.h5')


# Generate another data, and build another model (in order to use the encoder )
X,Y = make_regression(n_samples=1000,n_features=150,n_informative=10,noise=0.1,random_state=1)

X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=1)


# preprocessing
Y_train1 = Y_train.reshape((len(Y_train),1))
Y_test1 = Y_test.reshape((len(Y_test),1))

# normalization
process2 = MinMaxScaler()
process2.fit(X_train)

X_train2 = process2.transform(X_train)
X_test2 = process2.transform(X_test)

process3 = MinMaxScaler()
process3.fit(Y_train1)

Y_train1 = process3.transform(Y_train1)
Y_test1 = process3.transform(Y_test1)

# Define the model svr (Support vector Regression) Supervised Learning 

model = SVR()

model.fit(X_train1,Y_train1.ravel())

Y_predict = model.predict(X_test1)
Y_predict1 = Y_predict.reshape((len(Y_predict),1))

#inverse transform 
Y_predict1 = process3.inverse_transform(Y_predict1)
Y_test1 = process3.inverse_transform(Y_test1)

score = mean_absolute_error(Y_test1,Y_predict1)

print('The error is %d' % score)


# Load & Use the encoder
from tensorflow.keras.models import load_model
encoder = load_model('encoder.h5')

# preprocessing
Y_train2 = Y_train.reshape((len(Y_train),1))
Y_test2 = Y_test.reshape((len(Y_test),1))

# normalization
process2 = MinMaxScaler()
process2.fit(X_train)

X_train2 = process2.transform(X_train)
X_test2 = process2.transform(X_test)

process3 = MinMaxScaler()
process3.fit(Y_train2)

Y_train2 = process3.transform(Y_train2)
Y_test2 = process3.transform(Y_test2)

# Define the model svr (Support vector Regression) Supervised Learning 

X_train_e = e2.predict(X_train2)
X_test_e = e2.predict(X_test2)

model = SVR()

model.fit(X_train_e,Y_train2.ravel())

Y_predict = model.predict(X_test_e)
Y_predict2 = Y_predict.reshape((len(Y_predict),1))

#inverse transform 
Y_predict2 = process3.inverse_transform(Y_predict2)
Y_test2 = process3.inverse_transform(Y_test2)

score = mean_absolute_error(Y_test2,Y_predict2)

print('The error is %d' % score)