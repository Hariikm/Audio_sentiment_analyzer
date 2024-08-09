#GENERAL
import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import random
#PATH PROCESS
import os
import os.path
from pathlib import Path
import glob
#IMAGE PROCESS
from PIL import Image
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.applications.vgg16 import preprocess_input, decode_predictions
import imageio
from IPython.display import Image
import matplotlib.image as mpimg
#MUSIC PROCESS
import pydub
from scipy.io.wavfile import read, write
import librosa
import librosa.display
import IPython
from IPython.display import Audio
import scipy
#SCALER & TRANSFORMATION
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import regularizers
from sklearn.preprocessing import LabelEncoder
#ACCURACY CONTROL
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
#OPTIMIZER
from keras.optimizers import RMSprop,Adam,Optimizer,Optimizer, SGD
#MODEL LAYERS
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,MaxPooling2D,BatchNormalization,\
                        Permute, TimeDistributed, Bidirectional,GRU, SimpleRNN,\
LSTM, GlobalAveragePooling2D, SeparableConv2D, ZeroPadding2D, Convolution2D, ZeroPadding2D,Reshape,\
Conv2DTranspose, LeakyReLU, Conv1D, AveragePooling1D, MaxPooling1D
from keras import models
from keras import layers
import tensorflow as tf
from keras.applications import VGG16,VGG19,inception_v3
from keras import backend as K
from keras.utils import plot_model
from keras.datasets import mnist
import keras
#SKLEARN CLASSIFIER
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
#IGNORING WARNINGS
from warnings import filterwarnings
filterwarnings("ignore",category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=UserWarning)
import pickle
import wave
from pydub import AudioSegment


with open('files\scaler_data', 'rb') as file:
    scaler_data=pickle.load(file)

with open("files\encoder_label.pkl", 'rb') as file:
    encoder_label = pickle.load(file)

Model = tf.keras.models.load_model('files\Conv1D_Model.keras')



def add_noise(data):
    noise_value = 0.015 * np.random.uniform() * np.amax(data)
    data = data + noise_value * np.random.normal(size=data.shape[0])

    return data

def stretch_process(data,rate=0.8):

    return librosa.effects.time_stretch(data,rate= rate)


def shift_process(data):
    shift_range = int(np.random.uniform(low=-5,high=5) * 1000)

    return np.roll(data,shift_range)


def pitch_process(data,sampling_rate,pitch_factor=0.7):

    return librosa.effects.pitch_shift(data,sr= sampling_rate,n_steps= pitch_factor)






def extract_process(data, sample_rate):

    output_result = np.array([])
    mean_zero = np.mean(librosa.feature.zero_crossing_rate(y=data).T,axis=0)
    output_result = np.hstack((output_result,mean_zero))

    stft_out = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft_out,sr=sample_rate).T,axis=0)
    output_result = np.hstack((output_result,chroma_stft))

    mfcc_out = np.mean(librosa.feature.mfcc(y=data,sr=sample_rate).T,axis=0)
    output_result = np.hstack((output_result,mfcc_out))

    root_mean_out = np.mean(librosa.feature.rms(y=data).T,axis=0)
    output_result = np.hstack((output_result,root_mean_out))

    mel_spectogram = np.mean(librosa.feature.melspectrogram(y=data,sr=sample_rate).T,axis=0)
    output_result = np.hstack((output_result,mel_spectogram))

    return output_result



def export_process(path):

    data,sample_rate = librosa.load(path,duration=2.5,offset=0.6)

    output_1 = extract_process(data, sample_rate)
    result = np.array(output_1)

    noise_out = add_noise(data)
    output_2 = extract_process(noise_out, sample_rate)
    result = np.vstack((result,output_2))

    new_out = stretch_process(data)
    strectch_pitch = pitch_process(new_out, sample_rate)
    output_3 = extract_process(strectch_pitch, sample_rate)
    result = np.vstack((result,output_3))

    return result






def audio_sentiment(path):

    sound = AudioSegment.from_file(path)
    sound.export("./audio_wav.wav", format="wav")

    path_= './audio_wav.wav'
    x_Train=[]

    features = export_process(path_)

    for element in features:
        x_Train.append(element)

    new_predict_list= x_Train

    New_Predict_Feat = pd.DataFrame(new_predict_list)

    New_Predict_Feat = scaler_data.fit_transform(New_Predict_Feat)
    New_Predict_Feat = np.expand_dims(New_Predict_Feat,axis=2)


    prediction_nonseen = Model.predict(New_Predict_Feat)
    arg_prediction_nonseen = prediction_nonseen.argmax(axis=-1)
    y_prediction_nonseen = encoder_label.inverse_transform(prediction_nonseen)

    most_scored_index = np.argmax(arg_prediction_nonseen)

    # Get the corresponding class from y_prediction_nonseen
    most_scored_class = y_prediction_nonseen[most_scored_index]

 

    emotion_categories = {
    'neutral': ['OAF_neutral', 'YAF_neutral', 'OAF_Pleasant_surprise', 'YAF_pleasant_surprised'],
    'positive': ['YAF_happy', 'OAF_happy'],
    'negative': ['YAF_fear', 'OAF_angry', 'OAF_Fear', 'OAF_disgust', 'YAF_angry', 'OAF_Sad', 'YAF_disgust', 'YAF_sad']
    }

    def find_category(class_name):
        for category, classes in emotion_categories.items():
            if class_name in classes:
                return category
        return None

    # Example usage
    class_name = most_scored_class[0] # Replace with the class you want to check
    category = find_category(class_name)

    return category