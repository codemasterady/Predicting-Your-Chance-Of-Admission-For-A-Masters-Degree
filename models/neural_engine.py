# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Defining the Neural Engine Class


class NeuralEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.nn = Sequential()
        pass

    #! Data Preprocessing
    def __preprocess(self):
        # Data Preprocessing
        data_set = pd.read_csv(
            r'C:\Users\Selvaseetha\YouTube Codes\Predicting admission from important parameters\models\Admission_Predict_Ver1.1.csv')
        independent_variables = data_set.iloc[:, 1:8].values
        dependent_variable = data_set.iloc[:, -1].values
        # Scaling the data
        scaled_independent_variables = self.scaler.fit_transform(
            independent_variables)
        return np.array(scaled_independent_variables), np.array(dependent_variable)

    #! Train (Full Dataset)
    def mainTrain(self):
        X, y = self.__preprocess()
        # Defining the neural network
        self.nn.add(Dense(units=200, activation='relu'))
        self.nn.add(Dense(units=100, activation='relu'))
        self.nn.add(Dropout(rate=0.3))
        self.nn.add(Dense(units=50, activation='relu'))
        self.nn.add(Dense(units=25, activation='relu'))
        self.nn.add(Dense(units=12, activation='relu'))
        self.nn.add(Dense(units=6, activation='relu'))
        self.nn.add(Dropout(rate=0.6))
        self.nn.add(Dense(units=1, activation='sigmoid'))
        self.nn.compile(optimizer='adam', loss='mse')
        self.nn.fit(X, y, batch_size=60, epochs=40)

    #! Performance Evaluation
    def performanceEvaluation(self):
        X, y = self.__preprocess()
        # Splitting the data into the training and the test set for model performance evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3)
        # Defining the neural network
        eval_nn = Sequential()
        eval_nn.add(Dense(units=200, activation='relu'))
        eval_nn.add(Dense(units=100, activation='relu'))
        eval_nn.add(Dropout(rate=0.3))
        eval_nn.add(Dense(units=50, activation='relu'))
        eval_nn.add(Dense(units=25, activation='relu'))
        eval_nn.add(Dense(units=12, activation='relu'))
        eval_nn.add(Dense(units=6, activation='relu'))
        eval_nn.add(Dropout(rate=0.6))
        eval_nn.add(Dense(units=1, activation='sigmoid'))
        eval_nn.compile(optimizer='adam', loss='mse')
        eval_nn.fit(X_train, y_train, batch_size=60, epochs=40)
        # Extracting the evaluated values
        list_of_costs = []
        y_pred = eval_nn.predict(X_test)
        for i in range(0, 100):
            cost = y_test[i] - y_pred[i]
            # To ensure diff is always positive
            cost = abs(cost)
            list_of_costs.append(cost)
        # Plotting the results
        plt.plot(list_of_costs, 'r')
        plt.xlabel("Index (i)")
        plt.ylabel("Cost [Predicted value - Actual value] (%)")
        plt.show()

    #! Predicting the outcome using real world values
    def predictChances(self, input_array):
        input_array = np.array(input_array).reshape(1, 7)
        output = self.nn.predict(input_array)
        return output

    #! The initial private avoider (Only for testing)

    def avoidPrivate(self):
        return self.__preprocess()


# Testing
obj = NeuralEngine()
X, y = obj.avoidPrivate()

"""
                            Testing All The Possible Outcomes (Performance Report)
#! Initial Model
====================================================================================================================
X, y = self.__preprocess()
# Defining the neural network
self.nn.add(Dense(units=100, activation='relu'))
self.nn.add(Dense(units=50, activation='relu'))
self.nn.add(Dense(units=25, activation='relu'))
self.nn.add(Dense(units=12, activation='relu'))
self.nn.add(Dense(units=1, activation='sigmoid'))
self.nn.compile(optimizer='adam', loss='mse')
self.nn.fit(X, y, batch_size=50, epochs=40)
====================================================================================================================
#! Test A (Best Case)
input_arr = [340, 120, 5, 5, 5,  10, 1]
obj.mainTrain()
print(
    f"Best Case Scenario Has A Probability Of {str(obj.predictChances(input_arr))}")
#! Returned 1.0 (PASSED)
--------------------------------------------------------------------------------------------------------------------
#! Test B (Worst Case) (With Research)
input_arr = [0, 0, 5, 0, 0,  1, 0]
obj.mainTrain()
print(
    f"Worst Case Scenario Has A Probability Of {str(obj.predictChances(input_arr))}")
#! Returned 0.87353826 (FAILED)
--------------------------------------------------------------------------------------------------------------------
#! Test C (Average Case)
input_arr = [270, 60, 3, 3, 3,  1, 0]
obj.mainTrain()
print(
    f"Average Case Scenario Has A Probability Of {str(obj.predictChances(input_arr))}")
#! Returned 1.0 (Unknown)
--------------------------------------------------------------------------------------------------------------------
#! Test D (Average Case) (No Research)
input_arr = [270, 60, 3, 3, 3,  0, 0]
obj.mainTrain()
print(
    f"Average Case Scenario Has A Probability Of {str(obj.predictChances(input_arr))}")
#! Returned 1.0
--------------------------------------------------------------------------------------------------------------------
#! Test E (Worst Case)
input_arr = [0, 0, 5, 0, 0,  0, 0]
obj.mainTrain()
print(
    f"Average Case Scenario Has A Probability Of {str(obj.predictChances(input_arr))}")
#! Returned 0.8089772
--------------------------------------------------------------------------------------------------------------------
#! Test F (Worst Case) (With Bad University)
input_arr = [0, 0, 1, 0, 0,  0, 0]
obj.mainTrain()
print(
    f"Average Case Scenario Has A Probability Of {str(obj.predictChances(input_arr))}")
#! Returned 0.6375988
--------------------------------------------------------------------------------------------------------------------
#! Test G (Worst Case) (With Bad University) (With research)
input_arr = [0, 0, 1, 0, 0,  1, 0]
obj.mainTrain()
print(
    f"Average Case Scenario Has A Probability Of {str(obj.predictChances(input_arr))}")
#! Returned 0.74066675
--------------------------------------------------------------------------------------------------------------------
#! Test H (Average Case) (With Bad University)
input_arr = [270, 60, 1, 3, 3,  0, 0]
obj.mainTrain()
print(
    f"Average Case Scenario Has A Probability Of {str(obj.predictChances(input_arr))}")
#! Returned 1.
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

#! Model B
====================================================================================================================
nn = Sequential()
nn.add(Dense(units=200, activation='relu'))
nn.add(Dense(units=100, activation='relu'))
nn.add(Dense(units=50, activation='relu'))
nn.add(Dropout(rate=0.3))
nn.add(Dense(units=25, activation='relu'))
nn.add(Dense(units=12, activation='relu'))
nn.add(Dense(units=6, activation='relu'))
nn.add(Dropout(rate=0.6))
nn.add(Dense(units=1, activation='sigmoid'))
nn.compile(optimizer='adam', loss='mse')
====================================================================================================================
#! Test A (Best Case)
input_arr = [340, 120, 5, 5, 5,  10, 1]
obj.mainTrain()
print(
    f"Best Case Scenario Has A Probability Of {str(obj.predictChances(input_arr))}")
#! Returned 1.0 (PASSED)
--------------------------------------------------------------------------------------------------------------------
#! Test B (Worst Case) (With Research)
input_arr = [0, 0, 5, 0, 0,  1, 0]
obj.mainTrain()
print(
    f"Worst Case Scenario Has A Probability Of {str(obj.predictChances(input_arr))}")
#! Returned 0.73658943 (Improvement)
--------------------------------------------------------------------------------------------------------------------
#! Test C (Average Case)
input_arr = [270, 60, 3, 3, 3,  1, 0]
obj.mainTrain()
print(
    f"Average Case Scenario Has A Probability Of {str(obj.predictChances(input_arr))}")
#! Returned 1.0 (Unknown)
--------------------------------------------------------------------------------------------------------------------
#! Test D (Average Case) (No Research)
input_arr = [270, 60, 3, 3, 3,  0, 0]
obj.mainTrain()
print(
    f"Average Case Scenario Has A Probability Of {str(obj.predictChances(input_arr))}")
#! Returned 1.0
--------------------------------------------------------------------------------------------------------------------
#! Test E (Worst Case)
input_arr = [0, 0, 5, 0, 0,  0, 0]
obj.mainTrain()
print(
    f"Average Case Scenario Has A Probability Of {str(obj.predictChances(input_arr))}")
#! Returned 0.7697712 (Improvement)
--------------------------------------------------------------------------------------------------------------------
#! Test F (Worst Case) (With Bad University)
input_arr = [0, 0, 1, 0, 0,  0, 0]
obj.mainTrain()
print(
    f"Average Case Scenario Has A Probability Of {str(obj.predictChances(input_arr))}")
#! Returned 0.64155287 (Better??)
--------------------------------------------------------------------------------------------------------------------
#! Test G (Worst Case) (With Bad University) (With research)
input_arr = [0, 0, 1, 0, 0,  1, 0]
obj.mainTrain()
print(
    f"Average Case Scenario Has A Probability Of {str(obj.predictChances(input_arr))}")
#! Returned 0.7250743 (Improvement)
--------------------------------------------------------------------------------------------------------------------
#! Test H (Average Case) (With Bad University)
input_arr = [270, 60, 1, 3, 3,  0, 0]
obj.mainTrain()
print(
    f"Average Case Scenario Has A Probability Of {str(obj.predictChances(input_arr))}")
#! Returned 1.
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

#! Model C (Best Model)
====================================================================================================================
self.nn.add(Dense(units=200, activation='relu'))
self.nn.add(Dense(units=100, activation='relu'))
self.nn.add(Dropout(rate=0.3))
self.nn.add(Dense(units=50, activation='relu'))
self.nn.add(Dense(units=25, activation='relu'))
self.nn.add(Dense(units=12, activation='relu'))
self.nn.add(Dense(units=6, activation='relu'))
self.nn.add(Dropout(rate=0.6))
self.nn.add(Dense(units=1, activation='sigmoid'))
self.nn.compile(optimizer='adam', loss='mse')
self.nn.fit(X_train, y_train, batch_size=60, epochs=40)
====================================================================================================================
#! Test A (Best Case)
input_arr = [340, 120, 5, 5, 5,  10, 1]
obj.mainTrain()
print(
    f"Best Case Scenario Has A Probability Of {str(obj.predictChances(input_arr))}")
#! Returned 1.0 (PASSED)
--------------------------------------------------------------------------------------------------------------------
#! Test B (Worst Case) (With Research)
input_arr = [0, 0, 5, 0, 0,  1, 0]
obj.mainTrain()
print(
    f"Worst Case Scenario Has A Probability Of {str(obj.predictChances(input_arr))}")
#! Returned 0.71924317 (Improvement)
--------------------------------------------------------------------------------------------------------------------
#! Test C (Average Case)
input_arr = [270, 60, 3, 3, 3,  1, 0]
obj.mainTrain()
print(
    f"Average Case Scenario Has A Probability Of {str(obj.predictChances(input_arr))}")
#! Returned 1.0 (Unknown)
--------------------------------------------------------------------------------------------------------------------
#! Test D (Average Case) (No Research)
input_arr = [270, 60, 3, 3, 3,  0, 0]
obj.mainTrain()
print(
    f"Average Case Scenario Has A Probability Of {str(obj.predictChances(input_arr))}")
#! Returned 1.0
--------------------------------------------------------------------------------------------------------------------
#! Test E (Worst Case)
input_arr = [0, 0, 5, 0, 0,  0, 0]
obj.mainTrain()
print(
    f"Average Case Scenario Has A Probability Of {str(obj.predictChances(input_arr))}")
#! Returned 0.70957875 (Improvement)
--------------------------------------------------------------------------------------------------------------------
#! Test F (Worst Case) (With Bad University)
input_arr = [0, 0, 1, 0, 0,  0, 0]
obj.mainTrain()
print(
    f"Average Case Scenario Has A Probability Of {str(obj.predictChances(input_arr))}")
#! Returned 0.64277405 (Better)
--------------------------------------------------------------------------------------------------------------------
#! Test G (Worst Case) (With Bad University) (With research)
input_arr = [0, 0, 1, 0, 0,  1, 0]
obj.mainTrain()
print(
    f"Average Case Scenario Has A Probability Of {str(obj.predictChances(input_arr))}")
#! Returned 0.74086756 (Improvement??)
--------------------------------------------------------------------------------------------------------------------
#! Test H (Average Case) (With Bad University)
input_arr = [270, 60, 1, 3, 3,  0, 0]
obj.mainTrain()
print(
    f"Average Case Scenario Has A Probability Of {str(obj.predictChances(input_arr))}")
#! Returned 1.
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
"""
