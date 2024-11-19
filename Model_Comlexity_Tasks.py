import numpy as np
import pandas as pd
from sklearn import preprocessing
from pandas import read_csv
from matplotlib import pyplot
from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

winsconsin_headers = ['sample_code', 'c_thickness', 'uni_cell_size', 'uni_cell_shape', 'marg_adhesion', 'epi_cell_size', 'nuclei', 'bland_chromatin', 'mitoses', 'tumor_class']

wins_data = read_csv("Data/winsconsin_b_cancer.csv", names=winsconsin_headers)

wins_data.drop('sample_code', axis=1, inplace=True)

wins_data = wins_data.apply(pd.to_numeric, errors='coerce')

wincos_new_headers = ['c_thickness', 'uni_cell_size', 'uni_cell_shape', 'marg_adhesion', 'epi_cell_size', 'nuclei', 'bland_chromatin', 'mitoses', 'tumor_class']

wins_data[wincos_new_headers] = wins_data[wincos_new_headers].applymap(np.float)

