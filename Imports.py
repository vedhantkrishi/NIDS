
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import pickle # saving and loading trained model
from os import path
import tensorflow as tf 

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# importing required libraries for normalizing data
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score # for calculating accuracy of model
from sklearn.model_selection import train_test_split # for splitting the dataset for training and testing
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier


from keras.models import Model
from keras.models import Sequential #importing Sequential layer
from keras.models import model_from_json

from keras.layers import Dense # importing dense layer
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Input


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from keras.utils import plot_model



import os