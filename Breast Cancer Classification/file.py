import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split

breast_cancer_data = sklearn.datasets.load_breast_cancer()

data_frame = pd.DataFrame(breast_cancer_data.data, columns=breast_cancer_data.feature_names)

## and now adding the target column into the dataframe 
data_frame['label'] = breast_cancer_data.target

data_frame.info()

data_frame.isnull().sum()

data_frame.describe()

data_frame['label'].value_counts()

data_frame.groupby('label').mean()

X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=2, test_size=0.2)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

## Building the neural netwoek 

import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras

## setting the layers of Neural Network

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])

## complining the neural network

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

## training the neural network

history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.legend(['training data', 'validation data'], loc = 'lower right')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.legend(['training data', 'validation data'], loc = 'upper right')

loss, accuracy = model.evaluate(X_test_std, Y_test)
print(accuracy)

print(X_test_std.shape)

print(X_test_std[0])

Y_pred = model.predict(X_test_std)

print(Y_pred.shape)
print(Y_pred[0]) # probability of 0 and 1

## argms function 

my_list = [10, 20, 30]

index_of_max_value = np.argmax(my_list)
print(my_list)
print(index_of_max_value)

## converting the predict probability to class labels

Y_pred_label = [np.argmax(i) for i in Y_pred]
print(Y_pred_label)

model.save('brest_cancer_model.h5')

