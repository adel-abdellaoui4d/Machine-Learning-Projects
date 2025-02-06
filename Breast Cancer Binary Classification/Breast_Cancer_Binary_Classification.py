
# Cancer Survival Project

#Import data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
#load data 
data = pd.read_csv('haberman.csv',header=None)
print('***************************')
print(data)
print(data.shape)
print(data.describe())
columns=['age','year','nodes','class']

# Load the data with names for the columns
data_names = pd.read_csv('haberman.csv',header=None,names=columns)

# calculate the targets 
targets = data_names['class'].values

counter = Counter(targets)
print(counter)
for key,value in counter.items():
    percentage = value / len(targets)*100
    print('Class= %d , Count = %d , Percentage=%.3f %%' % (key,value,percentage))

# preparing the input(X) and output(Y)
X,Y = data_names.value[:,:-1],data_names.value[:,-1]

# preprocessing
X = X.astype('float32')
Y = LabelEncoder().fit_transform(Y)


# splitting
X_train, X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.5,stratify=Y,random_state=3)


# Train the Ml model
n_features = X.shape[1]

# Build the model
model = Sequential()
model.add(Dense(10,activation='relu',input_shape=(n_features,)))
model.add(Dense(1,activation='sigmoid'))

# set the Configuration
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# Train the Ml model
history = model.fit(X_train,Y_train,epochs=200,batch_size=20,validation_data=(X_test,Y_test))

# predict 
Y_predict = (model.predict(X_test) > 0.5).astype(int)

score = accuracy_score(Y_test,Y_predict)

print('Accuracy : %.3f %%'% score)

from matplotlib import pyplot as plt

plt.title('Learning Curve :: Lose')
plt.xlabel("Epoch")
plt.ylabel('Performance')
plt.plot(history.history['accuracy'],label='train')
plt.plot(history.history['val_accuracy'],label='validation')
plt.legend()
plt.show()


