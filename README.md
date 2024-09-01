# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

"Design and implement a neural network regression model to accurately predict a continuous target variable based on a set of input features within the provided dataset. The objective is to develop a robust and reliable predictive model that can capture complex relationships in the data, ultimately yielding accurate and precise predictions of the target variable. The model should be trained, validated, and tested to ensure its generalization capabilities on unseen data, with an emphasis on optimizing performance metrics such as mean squared error or mean absolute error. This regression model aims to provide valuable insights into the underlying patterns and trends within the dataset, facilitating enhanced decision-making and understanding of the target variable's behavior."

## Neural Network Model

![output](nn.svg)

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
### Name: AKASH KUMAR M.
### Register Number:212223230010
```python
# import the necesscary library
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```
```py
# fetch the data from excel and connect to sheets
auth.authenticate_user()
creds,_ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('ex-1').sheet1
rows = worksheet.get_all_values() 
```
```py
# Fetch the data and assain the data to variables
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'input':'float'})
df = df.astype({'output':'float'})
X = df[['input']].values
y = df[['output']].values
```
```py
# Create the model
AI_Brain = Sequential([
    Dense(units = 6, activation = 'relu', input_shape=[1]),
    Dense(units = 5, activation = 'relu'),
    Dense(units = 1)
])
```
```py
# Train the model
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
```
```py
# compile the model
AI_Brain.compile(optimizer= 'rmsprop', loss="mse")
AI_Brain.fit(X_train1,y_train,epochs=5000)
AI_Brain.summary()
```
```py
# plot the graph
loss_df = pd.DataFrame(AI_Brain.history.history)

loss_df.plot()
```
```py
X_test1 = Scaler.transform(X_test)
AI_Brain.evaluate(X_test1,y_test)
X_n1 = [[20]]
X_n1_1 = Scaler.transform(X_n1)
AI_Brain.predict(X_n1_1)
```
## Dataset Information

![output](./screen.png)

## OUTPUT

### Training Loss Vs Iteration Plot

![output](./plot.png)

### Test Data Root Mean Squared Error

![output](test.png)

### New Sample Data Prediction

![output](predict.png)

## RESULT


To develop a neural network regression model for the given dataset is created sucessfully.
