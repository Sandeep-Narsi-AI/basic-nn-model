# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

![newral_network](network.png)

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

```
NAME : P.SANDEEP
REG.NO : 212221230074
DEPT : AI-DS 
```

```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('dlexp1').sheet1

rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'input':'float'})
df = df.astype({'output':'float'})
df.head()

X = df[['input']].values
y = df[['output']].values
X

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.33,random_state=33)

from sklearn.preprocessing import MinMaxScaler
Scaler = MinMaxScaler()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

Scaler.fit(X_train)

X_train1 = Scaler.transform(X_train)

ai_brain = Sequential([Dense(8,activation = 'relu') , Dense(10,activation = 'relu') , Dense(1) ])

ai_brain.compile(optimizer = 'rmsprop' , loss = 'mse')

ai_brain.fit(X_train1,y_train,epochs = 2700)

loss_df = pd.DataFrame(ai_brain.history.history)

loss_df.plot()

X_test1 = Scaler.transform(X_test)

ai_brain.evaluate(X_test1,y_test)

X_n1 = [[4]]

X_n1_1 = Scaler.transform(X_n1)

ai_brain.predict(X_n1_1)
```

## Dataset Information

![data_input_output](data.png)

## OUTPUT

### Training Loss Vs Iteration Plot

![Training Loss Vs Iteration Plot](train_loss.png)

### Test Data Root Mean Squared Error

![loss1](loss1.png)
![loss2](loss2.png)

### New Sample Data Prediction

![data_prediction](datapred.png)

## RESULT

Thus a neural network regression model for the given dataset is written and executed successfully.


