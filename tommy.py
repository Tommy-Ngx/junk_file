import io
import os
import tensorflow as tf
import numpy as np

# Library
from PIL import Image
import shutil
import requests
from google.colab import files
from IPython.display import Image, display
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn import preprocessing
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 

from imblearn.combine import SMOTEENN 
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
from sklearn import preprocessing 
#sns.set(style="whitegrid")
np.random.seed(203)

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers

#py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
#warnings.filterwarnings('ignore')
from contextlib import contextmanager
import itertools





def plot_dimension_reduction(df,name):
    new_df = df.copy()
    X = new_df.drop('Class', axis=1)
    y = new_df['Class']
    # T-SNE Implementation
    t0 = time.time()
    X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)
    t1 = time.time()
    #print("T-SNE took {:.2} s".format(t1 - t0))
    # PCA Implementation
    t0 = time.time()
    X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)
    t1 = time.time()
    #print("PCA took {:.2} s".format(t1 - t0))

    # TruncatedSVD
    t0 = time.time()
    X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)
    t1 = time.time()
    #print("Truncated SVD took {:.2} s".format(t1 - t0))

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))
    # labels = ['No Fraud', 'Fraud']
    f.suptitle('Clusters using Dimensionality Reduction by {}'.format(name), fontsize=14)
    blue_patch = mpatches.Patch(color='#0A0AFF', label='No Cancer')
    red_patch = mpatches.Patch(color='#AF0000', label='Cancer')

    # t-SNE scatter plot
    ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Cancer', linewidths=2)
    ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Cancer', linewidths=2)
    ax1.set_title('t-SNE', fontsize=14)
    ax1.grid(True)
    ax1.legend(handles=[blue_patch, red_patch])

    # PCA scatter plot
    ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 0), cmap='coolwarm', label='No Cancer', linewidths=2)
    ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 1), cmap='coolwarm', label='Cancer', linewidths=2)
    ax2.set_title('PCA', fontsize=14)
    ax2.grid(True)
    ax2.legend(handles=[blue_patch, red_patch])

    # TruncatedSVD scatter plot
    ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 0), cmap='coolwarm', label='No Cancer', linewidths=2)
    ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 1), cmap='coolwarm', label='Cancer', linewidths=2)
    ax3.set_title('Truncated SVD', fontsize=14)
    ax3.grid(True)
    ax3.legend(handles=[blue_patch, red_patch])
    plt.show()

from matplotlib import cm

# Create a confusion matrix
def plot_confusion_matrix(cmodel, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    if normalize:
        cmodel = cmodel.astype('float') / cmodel.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cmodel)

    plt.imshow(cmodel, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cmodel.max() / 2.
    for i, j in itertools.product(range(cmodel.shape[0]), range(cmodel.shape[1])):
        plt.text(j, i, format(cmodel[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cmodel[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')







