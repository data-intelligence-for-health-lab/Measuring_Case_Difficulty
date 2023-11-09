import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from numpy import array

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from keras.callbacks import EarlyStopping

from datetime import datetime

import itertools
from hyperopt.pyll.base import scope 
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
import logging


# Dataset Setting
log_file_name = './Telco.log'
file_name = './Telco.xlsx'
max_eval_a = 200
max_eval_b = 200

# Configure the logger
logging.basicConfig(filename=os.path.join(log_file_name), level=logging.DEBUG)
logging.info('Start of the app2_telco:' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
logging.info("max_eval_a: %s, max_eval_b: %s", max_eval_a, max_eval_b)

# Load Data
data = pd.read_csv('./WA_Fn-UseC_-Telco-Customer-Churn.csv') 
df = data

# Drop Customer ID
df = df.drop('customerID',axis=1)
# Convert SeniorCitizen Column to the object
df["SeniorCitizen"] = df["SeniorCitizen"].replace({0:"No",1:"Yes"})
# Convert Churn column to a numeric 0/1 variable
df["Churn"] = df["Churn"].replace({"No":0, "Yes":1})

# Missing data imputation use the same row information
df.loc[df['tenure'] == 0, 'TotalCharges'] = 0
df["TotalCharges"] = df["TotalCharges"].astype('float64')

# Categorical columns and numeric columns
Cat_col = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract', 'PaperlessBilling', 'PaymentMethod']
Num_col = ["tenure", "TotalCharges", "MonthlyCharges"]


# Label Encoder for categorical 
label_encoder = LabelEncoder()
df[Cat_col] = df[Cat_col].apply(label_encoder.fit_transform)


# X and y set
X = df.drop('Churn',axis=1)
y = df['Churn']


# One-hot encoding for categorical 
onehot_encoder = OneHotEncoder(handle_unknown='ignore') 
enc_data = pd.DataFrame(onehot_encoder.fit_transform(X[Cat_col]).toarray())
enc_data.columns = onehot_encoder.get_feature_names_out()
column_names = enc_data.columns.to_list()
names = Num_col + column_names


# Merge numeric columns and one-hot encoded columns
X.drop(X[Cat_col] ,axis=1, inplace=True)
X_OH_data = pd.concat([X, enc_data], axis=1)
X_OH_data


# Scaler Set
scaler_transformer = Pipeline(steps=[('scaler', StandardScaler())])
scaler_preprocessor = ColumnTransformer(transformers=[
    ('scaling', scaler_transformer, Num_col)], remainder='passthrough')


#Split X and y
telco_X = X_OH_data
telco_y = y
telco_X  = telco_X.to_numpy()
telco_y = telco_y.to_numpy()


# Split the data to five groups and save each index in fold_indices 
kfold = KFold(n_splits=5, shuffle=True, random_state= 0)
# Save the index of each fold in a list
fold_index = []
for train_index, test_index in kfold.split(telco_X):
    fold_index.append((test_index)) 

    
# Order of using folds
fold_order = []
lst = [0,1,2,3,4]
for order in range(len(lst)):
    fold_order.append(lst[order:] + lst[:order])
#print(fold_order)


numbers = [5, 10, 15, 20]
one_combinations_with_order = list(itertools.product(numbers))
two_combinations_with_order = list(itertools.product(numbers, repeat=2))
three_combinations = list(itertools.product(numbers, repeat=3))


layer_neuron_orders = []
layer_neuron_orders = one_combinations_with_order + two_combinations_with_order+three_combinations
#print(len(layer_neuron_orders))
#print(layer_neuron_orders)


# Neural network model of model A
def create_model_A(params):
    # Create model
    model = tf.keras.models.Sequential()
    # First hidden layer with input shape
    model.add(tf.keras.layers.Dense(params['hidden_layer_sizes'][0], input_shape=(train_x_model_A.shape[1],), activation=params['activation']))   
    for i in range(1,len(params['hidden_layer_sizes'])):
        # from second hidden layer to number of hidden layers
        model.add(tf.keras.layers.Dense(params['hidden_layer_sizes'][i], activation=params['activation']))
        # Ouput layer 
    model.add(tf.keras.layers.Dense(1,activation='sigmoid')) 
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=params['learnRate']), metrics=['accuracy'])  
   
    es = EarlyStopping(monitor='val_loss',mode='min', verbose=1, patience=100)  
    result = model.fit(train_x_model_A, train_y_model_A, verbose=0, validation_split=0.3, 
                       batch_size = params['batch_size'], epochs = 500, 
                       callbacks=[es])
    
    validation_loss = np.amin(result.history['val_loss'])
    validation_acc = np.amin(result.history['val_accuracy']) 
    #print('Best validation_loss of epoch:', validation_loss,'Best validation_acc of epoch:', validation_acc)
    
    return {'loss': validation_loss,
            'acc': validation_acc,
            'status': STATUS_OK, 
            'model': model,
            'params': params}  


# Neural network model of model B
def create_model_B(params):
    # Create model
    model = tf.keras.models.Sequential()
    # First hidden layer with input shape
    model.add(tf.keras.layers.Dense(params['hidden_layer_sizes'][0], input_shape=(model_B_train_x.shape[1],), activation=params['activation']))   
    for i in range(1,len(params['hidden_layer_sizes'])):
        # from second hidden layer to number of hidden layers
        model.add(tf.keras.layers.Dense(params['hidden_layer_sizes'][i], activation=params['activation']))
        # Ouput layer 
    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=params['learnRate']), metrics=['accuracy'])  

    es = EarlyStopping(monitor='val_loss',mode='min', verbose=1, patience=100)  
    result = model.fit(model_B_train_x, model_B_train_y, verbose=0, validation_split=0.3, 
                       batch_size = params['batch_size'], epochs = 500, 
                       callbacks=[es])
    
    validation_loss = np.amin(result.history['val_loss'])
    validation_acc = np.amin(result.history['val_accuracy']) 
    #print('Best validation_loss of epoch:', validation_loss,'Best validation_acc of epoch:', validation_acc)
    
    return {'loss': validation_loss,
            'acc': validation_acc, 
            'status': STATUS_OK, 
            'model': model,
            'params': params}  


def approach_2(telco_X,telco_y,folds):
    ## First and Second folds
    # First fold
    X_fold = telco_X[fold_index[folds[0]]]
    y_fold = telco_y[fold_index[folds[0]]]
    d = pd.DataFrame(X_fold,columns=names)
    d['label'] = y_fold   
    fold_data0 = d
    
    # Second fold
    X_fold = telco_X[fold_index[folds[1]]]
    y_fold = telco_y[fold_index[folds[1]]]
    d = pd.DataFrame(X_fold,columns=names)
    d['label'] = y_fold   
    fold_data1 = d

    # train_data_for_model_A is concat of fold 1, fold 2
    train_data_for_model_A = pd.concat([fold_data0,fold_data1],ignore_index=True)
    #print(train_data_for_model_A)
    
    #Check the label is balanced
    print("train_data_for_model_A:",train_data_for_model_A['label'].value_counts())
    

    global train_x_model_A, train_y_model_A    
    # train_data_for_model_A to x,y  
    train_x_model_A =  train_data_for_model_A.iloc[:,:-1]
    train_y_model_A =  train_data_for_model_A.iloc[:, -1]
    
    #StandardScaler be generated by fold 1 and fold 2
    train_x_model_A = scaler_preprocessor.fit_transform(train_x_model_A)
    
    # new search space
    search_space = {'learnRate': hp.choice('learnRate',[0.01,0.03,0.1]),
                    'batch_size': scope.int(hp.choice('batch_size',[32,64,128])),
                    'activation':hp.choice('activation',['relu','tanh']),
                    'hidden_layer_sizes':hp.choice('hidden_layer_sizes',layer_neuron_orders)}
    
    trials = Trials()   
    best = fmin(fn=create_model_A,
                space=search_space,
                algo=tpe.suggest,
                max_evals=max_eval_a,
                trials=trials,
                verbose=False)

    best_model_A = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
    best_params_A = trials.results[np.argmin([r['loss'] for r in trials.results])]['params']
    best_acc_A =  trials.results[np.argmin([r['loss'] for r in trials.results])]['acc']
    best_loss_A =  trials.results[np.argmin([r['loss'] for r in trials.results])]['loss']
    print("best_acc_A:",best_acc_A,"best_loss_A:",best_loss_A)
    print("best_params_A:",best_params_A)   
 
    
    ## Thire and Fourth folds
    # Third fold
    X_fold = telco_X[fold_index[folds[2]]]
    y_fold = telco_y[fold_index[folds[2]]]
    d = pd.DataFrame(X_fold,columns=names)
    d['label'] = y_fold   
    fold_data2 = d
    
    # Fourth fold
    X_fold = telco_X[fold_index[folds[3]]]
    y_fold = telco_y[fold_index[folds[3]]]
    d = pd.DataFrame(X_fold,columns=names)
    d['label'] = y_fold   
    fold_data3 = d
    
    # test data for model A with fold 3, fold 4
    test_data_for_model_A = pd.concat([fold_data2,fold_data3],ignore_index=True)
    print("test_data_for_model_A:",test_data_for_model_A['label'].value_counts())
    
    
    # test_data_for_model_A to x and answer
    model_A_X_test = test_data_for_model_A.iloc[:,:-1]
    answer = test_data_for_model_A.iloc[:, -1]

    #print("before:",model_A_X_test)
    #Generated StandardScaler used for fold 3 and fold 4
    model_A_X_test = scaler_preprocessor.transform(model_A_X_test)
    # Convert back to pandas DataFrame and assign original index
    model_A_X_test = pd.DataFrame(model_A_X_test)
    #print("after:",model_A_X_test)
    
    # Traind model A make predictions
    model_A_predictions = (best_model_A.predict(model_A_X_test) > 0.5).astype(int)
    
    # Compare between answer and predictions from model A for fold 3, 4
    id_df = pd.DataFrame()
    id_df["actual"] = answer
    id_df["predicted"] = model_A_predictions
    incorrect = id_df.loc[id_df.actual != id_df.predicted]
    
    incorrect_index = []
    incorrect_index = incorrect.index
    print("number of incorrect predictions:",len(incorrect_index),'\n',incorrect["actual"].value_counts())
    sum_number_incorrect.append(len(incorrect_index))

    # Provide 0 to wrong correctness, 1 to right correctness
    # Append correctness to the fold 3, 4 dataset and find the hyperparameter
    wrong_number = incorrect_index
    correctness = []
    for i in range(len(model_A_X_test)):
        if model_A_X_test.index[i] in wrong_number:
            correctness.append(0)
        elif model_A_X_test.index[i] not in wrong_number:
            correctness.append(1)
        else:
            print("error")
    correctness = pd.DataFrame(correctness,columns =['correctness'])
    print(correctness['correctness'].value_counts())
    

    # Get the raw test data from fold 3, fold 4 (Before StandardScaler)
    test_data_for_model_A = pd.concat([fold_data2,fold_data3],ignore_index=True)
    # Drop the y 
    model_A_X_test = test_data_for_model_A.iloc[:,:-1]
    #Generate New StandardScaler using fold 3 and fold 4 (will be used for fold 5)
    model_A_X_test = scaler_preprocessor.fit_transform(model_A_X_test)
    
    
    global model_B_train_x, model_B_train_y    
    # fold 3, 4 data with correctness
    model_B_train_x =  model_A_X_test
    model_B_train_y =  correctness    

    trials = Trials()   
    best = fmin(fn=create_model_B,
                space=search_space,
                algo=tpe.suggest,
                max_evals=max_eval_b,
                trials=trials,
                verbose=False)
    
    best_model_B = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
    best_params_B = trials.results[np.argmin([r['loss'] for r in trials.results])]['params']
    best_acc_B =  trials.results[np.argmin([r['loss'] for r in trials.results])]['acc']
    best_loss_B =  trials.results[np.argmin([r['loss'] for r in trials.results])]['loss']
    print("best_acc_B:",best_acc_B,"best_loss_B:",best_loss_B)
    print("best_params_B:",best_params_B)
    
    
    # Predict fold 5 label correctness
    X_fold = telco_X[fold_index[folds[4]]]
    y_fold = telco_y[fold_index[folds[4]]]
    d = pd.DataFrame(X_fold,columns=names)
    d['label'] = y_fold   
    difficulty_data_for_model_B = d
    print("difficulty_data_for_model_B:",difficulty_data_for_model_B['label'].value_counts())   
    
    
    difficulty_x = difficulty_data_for_model_B.iloc[:,:-1]
    
    #Generated StandardScaler from fold 3 and fold 4 used for fold 5
    difficulty_x = scaler_preprocessor.transform(difficulty_x)      
    
    
    predicted_difficulty = 1 - best_model_B.predict(difficulty_x) #By doing 1- Difficult case is closer to 1, Easy case is closer to 0
    predicted_difficulty = predicted_difficulty[:,0].tolist()
    
    return(difficulty_data_for_model_B, predicted_difficulty)


Final_data_save = []
Final_difficulty_save = []
sum_number_incorrect = []

overall_start_time = datetime.now()
for i in range(len(fold_order)):
    start_time = datetime.now()
    folds = fold_order[i]
    print('\n\n\nfold:',folds)
    logging.info('fold:%s',folds)
    difficulty_data_for_model_B,difficulty = approach_2(telco_X,telco_y,folds)
    Final_data_save.append(difficulty_data_for_model_B)
    Final_difficulty_save.append(difficulty)
    end_time = datetime.now()
    fold_calculation_time = end_time - start_time
    print("time per fold:",fold_calculation_time)
    logging.info("time per fold:%s",fold_calculation_time)

overall_end_time = datetime.now()
overall_calculation_time = overall_end_time - overall_start_time
print("Overall time per fold:",overall_calculation_time)
logging.info("Overall time per fold:%s",overall_calculation_time)

print(sum_number_incorrect)
print(sum(sum_number_incorrect))

dataframe_temp = Final_data_save
# Merge all the difficulties from the folds
for i in range(len(fold_order)):
    dataframe_temp[i]['difficulty'] = Final_difficulty_save[i]
    
    
new_column_names = names
new_column_names.extend(['label','difficulty'])

temp_df= pd.DataFrame(np.row_stack([dataframe_temp[0], dataframe_temp[1], dataframe_temp[2], dataframe_temp[3], dataframe_temp[4]]), 
                               columns=new_column_names)

writer = pd.ExcelWriter(file_name)
temp_df.to_excel(writer,header=True,index=False)
writer.close()