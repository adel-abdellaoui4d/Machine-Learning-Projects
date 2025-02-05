import numpy as np
from keras.models import Sequential
from keras.layers import Dense

data = np.loadtxt('pima-indians-diabetes.csv',delimiter=",")

#input 
input_data = data[:,0:8]
print(input_data[0])
#output
output_data = data[:,8]
print(output_data[0])


# build Machine Learning Model
model = Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# Set the Configiration
model.compile(loss='binary_crossentropy',optimizer='adam',matrics=['accuracy'])

# Train The Ml Model
model.fit(input_data,output_data,epochs=150, batch_size=10,verbose=0)

accuracy = model.evaluate(input_data,output_data,verbose=0)

print("Accuracy : %.2f %%"  % (accuracy[1]*100))



# Make Predictions

predictions = (model.predict(input_data) > 0.5).astype(int)

# Compare The Predicted Data with the Actual Data

def display(num):
    if num == 0:
        return "No Disease"
    else:
        return "Disease"


for i in range(50):
    print("%s => %s (expected %s)"% (input_data[i].tolist(),display(predictions[i]),display(output_data[i])))






