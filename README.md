# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Designing and implementing a neural network regression model aims to accurately predict a continuous target variable based on a set of input features from the provided dataset. The neural network learns complex relationships within the data through interconnected layers of neurons. The model architecture includes an input layer for the features, several hidden layers with non-linear activation functions like ReLU to capture complex patterns, and an output layer with a linear activation function to produce the continuous target prediction.


## Neural Network Model

![image](https://github.com/user-attachments/assets/d5fcd281-7218-43e5-b823-e796503665e7)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:M Pranathi
### Register Number:212222240064
```

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

Ai_Brain = Sequential([
Dense(units = 9, activation = 'relu',input_shape = [8]),
Dense(units = 9, activation = 'relu'),
Dense(units = 9, activation = 'relu'),
Dense(units = 1)
])

Ai_Brain.summary()

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('data1').sheet1

rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'input':'int'})
df.head()

dataset1 = pd.DataFrame(rows[1:],columns=rows[0])
dataset1 = dataset1.astype({'input':'int'})
dataset1 = dataset1.astype({'output':'int'})

X = dataset1[['input']].values
y = dataset1[['output']].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state=0)

Scaler = MinMaxScaler()

Scaler.fit(X_train)

X_train1 = Scaler.transform(X_train)

Ai_Brain = Sequential([
    Dense(8,activation='relu'),
    Dense(10,activation='relu'),
    Dense(1)
])

Ai_Brain.compile(optimizer = 'rmsprop', loss='mse')

Ai_Brain.fit(X_train1,y_train,epochs=2000)

loss_df=pd.DataFrame(Ai_Brain.history.history)
loss_df.plot()

X_test1=Scaler.transform(X_test)

Ai_Brain.evaluate(X_test1,y_test)

X_n1=[[10]]

X_n1_1=Scaler.transform(X_n1)

Ai_Brain.predict(X_n1_1)

```
## Dataset Information

![Screenshot 2024-09-01 181712](https://github.com/user-attachments/assets/5414be99-4671-46a6-bc31-466bc1bcaf84)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/dbe52a19-b867-4d20-8752-e4d3a0bb5c8d)


### Test Data Root Mean Squared Error

![image](https://github.com/user-attachments/assets/dc767f7a-bdc2-4d98-8d08-bd64d70f8cbe)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/f43571f8-a110-417f-9d7d-747527cdc1f1)


## RESULT

Thus, Neural network for Regression model is successfully Implemented
