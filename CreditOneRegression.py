#!/usr/bin/env python
# coding: utf-8

# In[429]:


# imports
# numpy, pandas, scipy, math, matplotlib
import numpy as np
import pandas as pd
import scipy
from math import sqrt
import matplotlib.pyplot as plt

# estimators, sklearn is a portion of the Estimator Object
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import linear_model

# model metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

# cross validation
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

import os

os.getcwd()


# In[430]:


# data
rawData = pd.read_csv('default of credit card clients.csv', header=1)
rawData.head()


# In[431]:


rawData.info()


# In[432]:


print(rawData.iloc[1,1])
print(rawData['SEX'])
print(rawData.iloc[2:6,0:2])

# features
features = rawData.iloc[:,12:23]
print('Summary of features sample')
features.head()


# In[433]:


# dependent variable
depVar = rawData['PAY_AMT6']
print(depVar.iloc[1:3])


# In[434]:


# Training Set (Feature Space: X Training)
X_Train = (features[:1000])
X_Train.head()


# In[435]:


# Dependent Variable Training Set (y Training)
y_train = depVar[: 1000]
y_train_count = len(y_train.index)
print('The number of observations in the Y training set are:',str(y_train_count))
y_train.head()


# In[436]:


# Testing Set (X Testing)
X_test = features[-100:]
X_test_count = len(X_test.index)
print('The number of observations in the feature testing set is:',str(X_test_count))
print(X_test.head())


# In[437]:


# Ground Truth (y_test) 
y_test = depVar[-100:]
y_test_count = len(y_test.index)
print('The number of observations in the Y training set are:',str(y_test_count))
y_test.head()


# In[438]:


# 2. Select the Dependent Variable
# dependent variable
depVar = rawData['PAY_AMT6']

# 3. Establish the training set for the X-variables or Feature space 
# (first 1000 rows: only for this example you will still follow a 70/30 split for your final models)
# Training Set (Feature Space: X Training)
X_train = (features[: 1000])
X_train.head()

# 4. Establish the training set for the Y-variable or dependent variable (the number of rows much match the X-training set)
# Dependent Variable Training Set (y Training)
y_train = depVar[: 1000]
y_train_count = len(y_train.index)
print('The number of observations in the Y training set are:',str(y_train_count))
y_train.head()

# We can implement Cross Validation anytime we need to by simply running the following on the X and Y training sets:
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)

# We can also use the shape function to double check that the split was made as needed:
X_train.shape, X_test.shape

# We've already established out training and testing sets we can easily cross validate by 
# using sklearn.model_selection on our datasets as follows:
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)
X_train.shape, X_test.shape


# In[439]:


model = LinearRegression(n_jobs=10)


# In[440]:


########################################################################
#####      2. Building the Models and Choosing the Right Model     #####
########################################################################

# Models
modelSVR = SVR()
modelRF = RandomForestRegressor()
modelLR = LinearRegression()


# In[441]:


# features
features = rawData.iloc[:,12:23]
print('Summary of feature sample')
features.head()


# In[442]:


# dependent variable
print(depVar)


# In[443]:


model.fit(X_train,y_train)


# In[444]:


modelRF.fit(X_train,y_train)


# In[445]:


modelSVR.fit(X_train,y_train)


# In[446]:


modelLR.fit(X_train,y_train)


# In[447]:


############################################
#####     Choosing the Right Model     #####
############################################

print(cross_val_score(modelSVR, X_train, y_train)) 

#####################################################################################
# [-0.02738768 -0.06746915 -0.14855097 ...]
# These values correspond to the the following:
#   1st value:  The score array for test scores on each cv split. 
#               (Higher is an indicator of a better performing model)
#   2nd value: The time for fitting the estimator on the train set for each cv split.
#   3rd Value:  The time for scoring the estimator on the test set for each cv split. 
#####################################################################################


# In[448]:


print(cross_val_score(modelRF, X_train, y_train)) 


# In[449]:


print(cross_val_score(modelLR, X_train, y_train)) 


# In[450]:


# The default metric for model.score for Regression models in Sci-Kit Learn is R-Squared.
model.score(X_train,y_train)


# In[451]:


# 1. Random Forest Regression Model Fitting and Scoring
# Model Fitting
modelRF.fit(X_train,y_train)
print(cross_val_score(modelRF, X_train, y_train))
modelRF.score(X_train,y_train)


# In[452]:


# 2.Support Vector Regression Model Fitting and Scoring
modelSVR.fit(X_train,y_train)
print(cross_val_score(modelSVR, X_train, y_train)) 
modelSVR.score(X_train,y_train)


# In[453]:


# 3. Linear Regression Model Fitting and Scoring
modelLR.fit(X_train,y_train)
print(cross_val_score(modelLR, X_train, y_train)) 
modelLR.score(X_train,y_train)


# In[454]:


#####################################################################################
#####               Making Predictions and Evaluating the Results               #####
#####################################################################################

# Random Forest model your previously created
predictions = modelRF.predict(X_test)


# In[455]:


# Evaluating the Results
# 1. The sklearn.metrics Object is the main object that contains almost all of the metric 
# functions you will need. Verify that the first two are in your imported list of libraries 
# (you'll see some familiarity in their names):

# We can use the sqrt function and the mean_squared_error function to compose 
# your own function for calculating RMSE:
rmse = sqrt(mean_squared_error(y_test, predictions))

# We establish a variable and use the included function, the ground truth, and the predictions 
# to calculate R Squared as follows:
predRsquared = r2_score(y_test,predictions)


# In[456]:


# Here is how it all looks together:
#Make Predictions
predictions = modelRF.predict(X_test)
predRsquared = r2_score(y_test,predictions)
rmse = sqrt(mean_squared_error(y_test, predictions))
print('R Squared: %.3f' % predRsquared)
print('RMSE: %.3f' % rmse)


# In[457]:


# Plotting the Results
# Revisar por qué no funciona como parámetro en la función scatter color=['blue','green'],
plt.scatter(y_test, predictions, c='blue', alpha = 0.5)
plt.xlabel('Ground Truth')
plt.ylabel('Predictions')
plt.show();


# In[458]:


########################################################################
########################################################################
#####     Default of Credit Card Clients - Modelos Predictivos     #####
########################################################################
########################################################################

## Cargar paquetes utilizados en el proyecto

import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import gc
from datetime import datetime 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
get_ipython().system('pip install lightgbm')
import lightgbm as lgb
get_ipython().system('pip install xgboost')
import xgboost as xgb

pd.set_option('display.max_columns', 100)


# In[459]:


## Leer datos por defecto de los clientes de tarjetas de crédito

credit_df = pd.read_csv('default of credit card clients.csv', header=1)
print("Datos de las tarjetas de créditos de clientes - líneas:", credit_df.shape[0]," columnas:", credit_df.shape[1])


# In[460]:


credit_df.head()


# In[461]:


credit_df.columns


# In[462]:


credit_df.describe()


# In[463]:


credit_df.info()


# In[464]:


## Validar datos nulos

totalDatosNulos = credit_df.isnull().sum().sort_values(ascending = False)
porcentaje = (credit_df.isnull().sum()/credit_df.isnull().count()*100).sort_values(ascending = False)
pd.concat([totalDatosNulos, porcentaje], axis=1, keys=['Total', 'Porcenjate']).transpose()


# In[465]:


## Desequilibrio en los datos

temp = credit_df["default payment next month"].value_counts()

df = pd.DataFrame({'default payment next month': temp.index,'values': temp.values})
plt.figure(figsize = (6,6))

sns.set_color_codes("pastel")
sns.barplot(x = 'default payment next month', y="values", data=df)

# Add title and axis names
plt.title('Clientes de tarjeta de crédito predeterminados - valor objetivo - desequilibrio de datos\n (Default = 1, Not Default = 0)')
plt.xlabel('pago predeterminado el próximo mes')
plt.ylabel('valores desequilibrio')

locs, labels = plt.xticks()
plt.show()


# In[466]:


########################################################################
#####                     Explorando los datos                     #####
########################################################################

## Cantidad de límite de crédito
plt.figure(figsize = (14,6))
plt.title('Cantidad de límite de crédito - Gráfico de densidad')
sns.set_color_codes("pastel")

sns.distplot(credit_df['LIMIT_BAL'], bins=200, kde=True, color="black")

plt.xlabel('saldo límite')

plt.show()


# In[467]:


credit_df['LIMIT_BAL'].value_counts().shape


# In[468]:


credit_df['LIMIT_BAL'].value_counts().head()


# In[469]:


lb_low = credit_df["LIMIT_BAL"].quantile(0.01)
lb_hi  = credit_df["LIMIT_BAL"].quantile(0.99)

credit_df_filtered = credit_df[(credit_df["LIMIT_BAL"] < q_hi) & (credit_df["LIMIT_BAL"] > q_low)]

credit_df_filtered.describe()


# In[470]:


## Cantidad de límite de crédito

plt.figure(figsize = (14,6))
plt.title('Cantidad de límite de crédito - Gráfico de densidad')
sns.set_color_codes("pastel")

sns.distplot(credit_df_filtered['LIMIT_BAL'], bins=200, kde=True, color="black")

plt.xlabel('saldo límite')

plt.show()


# In[471]:


## Cantidad de límite de crédito agrupada por pago predeterminado el próximo mes

clsNoDefault = credit_df_filtered.loc[credit_df_filtered['default payment next month'] == 0]["LIMIT_BAL"]
clsDefault = credit_df_filtered.loc[credit_df_filtered['default payment next month'] == 1]["LIMIT_BAL"]

plt.figure(figsize = (14,6))

plt.title('Importe predeterminado del límite de crédito: agrupado por pago el próximo mes (Gráfico de densidad)')
sns.set_color_codes("pastel")
sns.distplot(clsDefault, kde=True, bins=200, color="red")
sns.distplot(clsNoDefault, kde=True, bins=200, color="green")
plt.xlabel('saldo límite')

plt.show()


# In[ ]:





# In[472]:


## Límite de crédito vs. sexo

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))

s = sns.boxplot(ax = ax1, x="SEX", y="LIMIT_BAL", hue="SEX", data=credit_df_filtered, palette="PRGn", showfliers=True)
s = sns.boxplot(ax = ax2, x="SEX", y="LIMIT_BAL", hue="SEX", data=credit_df_filtered, palette="PRGn", showfliers=False)

plt.show();


# In[473]:


## Características de correlación

varBillAMT = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']

plt.figure(figsize = (8,8))
plt.title('Cantidad de extracto de cuenta (abril-septiembre) gráfico de correlación')

corr = credit_df_filtered[varBillAMT].corr()

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.1, vmin=-1, vmax=1)

plt.show()


# In[474]:


varPayAMT = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

plt.figure(figsize = (8,8))
plt.title('Monto del pago anterior (abril-septiembre) \ n gráfico de correlación')

corr = credit_df_filtered[varPayAMT].corr()

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.1, vmin=-1, vmax=1)

plt.show()


# In[475]:


varPay = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

plt.figure(figsize = (8,8))
plt.title('Estado de reembolso (abril-septiembre) Gráfico de correlación')

corr = credit_df_filtered[varPay].corr()

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.1, vmin=-1, vmax=1)

plt.show()


# In[476]:


## Sexo, educación, edad y matrimonio

def boxplot_variation(f1, f2, f3, width=16):
    fig, ax1 = plt.subplots(ncols=1, figsize=(width,6))
    s = sns.boxplot(ax = ax1, x=f1, y=f2, hue=f3,
                data=credit_df_filtered, palette="PRGn", showfliers=False)
    s.set_xticklabels(s.get_xticklabels(), rotation=90)
    plt.show();
    
boxplot_variation('MARRIAGE','AGE', 'SEX',8)


# In[477]:


boxplot_variation('AGE','LIMIT_BAL', 'SEX',16)


# In[478]:


## Estado civil, nivel educativo y límite de crédito

boxplot_variation('MARRIAGE','LIMIT_BAL', 'EDUCATION',12)


# In[479]:


## Modelos predictivos
## Definir predictores y valores objetivo

varDependiente = 'default payment next month'
varIndependientes = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
                     'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
                     'BILL_AMT1','BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                     'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']


# In[480]:


## Dividir datos en tren y conjunto de validación

train_credit_df_filtered, val_credit_df_filtered = train_test_split(credit_df_filtered, test_size = 0.20, random_state = 2018, shuffle=True )

train_credit_df_filtered_bkp = train_credit_df_filtered.copy()
val_credit_df_filtered_bkp =val_credit_df_filtered.copy()


# In[481]:


## Random Forest Classifier

clfRF = RandomForestClassifier(n_jobs = 4, 
                               random_state = 2018,
                               criterion = 'gini',
                               n_estimators = 100,
                               verbose = False)


# In[482]:


clfRF.fit(train_credit_df_filtered[varIndependientes], train_credit_df_filtered[varDependiente].values)


# In[483]:


predictores = clfRF.predict(val_credit_df_filtered[varIndependientes])


# In[484]:


## Features importance

temporal = pd.DataFrame({'Feature': varIndependientes, 'Feature importance': clfRF.feature_importances_})
temporal = temporal.sort_values(by='Feature importance', ascending=False)

plt.figure(figsize = (7,4))
plt.title('Importancia de las características', fontsize=14)

s = sns.barplot(x='Feature', y='Feature importance', data=temporal)
s.set_xticklabels(s.get_xticklabels(), rotation=90)

plt.xlabel('Características')
plt.ylabel('Importancia de las características')

plt.show()  


# In[485]:


## Confusion matrix

ct = pd.crosstab(val_credit_df_filtered[varDependiente].values, predictores, rownames=['Actual'], colnames=['Predicted'])
fig, (ax1) = plt.subplots(ncols=1, figsize=(4,4))
sns.heatmap(ct, 
            xticklabels=['Not Default', 'Default'],
            yticklabels=['Not Default', 'Default'],
            annot=True,
            ax=ax1,
            linewidths=.2,
            linecolor="Darkblue", 
            cmap="Blues")
plt.title('Confusion Matrix', fontsize=10)
plt.show()


# In[486]:


roc_auc_score(val_credit_df_filtered[varDependiente].values, predictores)


# In[487]:


## RandomForrest con OneHotEncoder

cat_caracteristicas = ['EDUCATION', 'SEX', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

train_fea_df = pd.get_dummies(train_credit_df_filtered_bkp, columns = cat_caracteristicas)
val_fea_df = pd.get_dummies(val_credit_df_filtered_bkp, columns = cat_caracteristicas)

print("Valor predeterminado de datos de entrenamiento de clientes de tarjetas de crédito - filas:",train_fea_df.shape[0]," columnas:", train_fea_df.shape[1])
print("Valor predeterminado de los datos val de los clientes de la tarjeta de crédito - filas:",val_fea_df.shape[0]," columnas:", val_fea_df.shape[1])


# In[488]:


train_feaali_df, val_feaali_df = train_fea_df.align(val_fea_df, join='outer', axis=1, fill_value=0)

print("Valor predeterminado de datos de entrenamiento de clientes de tarjetas de crédito - filas:",train_feaali_df.shape[0]," columnas:", train_feaali_df.shape[1])
print("Valor predeterminado de los datos val de los clientes de la tarjeta de crédito - filas:",val_feaali_df.shape[0]," columnas:", val_feaali_df.shape[1])


# In[489]:


train_feaali_df.head(5)


# In[490]:


val_feaali_df.head(5)


# In[491]:


objetivo_f = 'default payment next month'
predictores_f = ['AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
                 'BILL_AMT5', 'BILL_AMT6', 'EDUCATION_0', 'EDUCATION_1',
                 'EDUCATION_2', 'EDUCATION_3', 'EDUCATION_4', 'EDUCATION_5',
                 'EDUCATION_6', 'LIMIT_BAL', 'MARRIAGE_0', 'MARRIAGE_1',
                 'MARRIAGE_2', 'MARRIAGE_3', 'PAY_0_-1', 'PAY_0_-2', 'PAY_0_0',
                 'PAY_0_1', 'PAY_0_2', 'PAY_0_3', 'PAY_0_4', 'PAY_0_5', 'PAY_0_6',
                 'PAY_0_7', 'PAY_0_8', 'PAY_2_-1', 'PAY_2_-2', 'PAY_2_0', 'PAY_2_1',
                 'PAY_2_2', 'PAY_2_3', 'PAY_2_4', 'PAY_2_5', 'PAY_2_6', 'PAY_2_7',
                 'PAY_2_8', 'PAY_3_-1', 'PAY_3_-2', 'PAY_3_0', 'PAY_3_1', 'PAY_3_2',
                 'PAY_3_3', 'PAY_3_4', 'PAY_3_5', 'PAY_3_6', 'PAY_3_7', 'PAY_3_8',
                 'PAY_4_-1', 'PAY_4_-2', 'PAY_4_0', 'PAY_4_1', 'PAY_4_2', 'PAY_4_3',
                 'PAY_4_4', 'PAY_4_5', 'PAY_4_6', 'PAY_4_7', 'PAY_4_8', 'PAY_5_-1',
                 'PAY_5_-2', 'PAY_5_0', 'PAY_5_2', 'PAY_5_3', 'PAY_5_4', 'PAY_5_5',
                 'PAY_5_6', 'PAY_5_7', 'PAY_5_8', 'PAY_6_-1', 'PAY_6_-2', 'PAY_6_0',
                 'PAY_6_2', 'PAY_6_3', 'PAY_6_4', 'PAY_6_5', 'PAY_6_6', 'PAY_6_7',
                 'PAY_6_8', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4',
                 'PAY_AMT5', 'PAY_AMT6', 'SEX_1', 'SEX_2']

clf.fit(train_feaali_df[predictores_f], train_credit_df_filtered[objetivo_f].values)


# In[492]:


preds = clf.predict(val_feaali_df[predictores_f])

temporal = pd.DataFrame({'Feature': predictores_f, 'Feature importance': clf.feature_importances_})
temporal = temporal.sort_values(by='Feature importance', ascending=False)

plt.figure(figsize = (16,4))
plt.title('Importancia de las características',fontsize=14)

s = sns.barplot(x='Feature', y='Feature importance', data=temporal)
s.set_xticklabels(s.get_xticklabels(),rotation=90)

plt.xlabel('Características')
plt.ylabel('Importancia de las características')

plt.show()


# In[493]:


ct = pd.crosstab(val_feaali_df[varDependiente].values, preds, rownames=['Actual'], colnames=['Predicted'])
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(ct, 
            xticklabels=['Not Default', 'Default'],
            yticklabels=['Not Default', 'Default'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Darkblue", cmap="Blues")
plt.title('Matriz de Confusión', fontsize=10)
plt.show()


# In[494]:


roc_auc_score(val_feaali_df[varDependiente].values, preds)


# In[495]:


## AdaBoostClassifier

clf = AdaBoostClassifier(random_state = 2018,
                         algorithm = 'SAMME.R',
                         learning_rate = 0.8,
                         n_estimators = 100)

clf.fit(train_credit_df_filtered[varIndependientes], train_credit_df_filtered[varDependiente].values)


# In[496]:


preds = clf.predict(val_credit_df_filtered[varIndependientes])

temporal = pd.DataFrame({'Feature': varIndependientes, 'Feature importance': clf.feature_importances_})
temporal = temporal.sort_values(by='Feature importance', ascending=False)

plt.figure(figsize = (7,4))
plt.title('Importancia de las características', fontsize=14)

s = sns.barplot(x='Feature', y='Feature importance', data=temporal)
s.set_xticklabels(s.get_xticklabels(), rotation=90)

plt.xlabel('Características')
plt.ylabel('Importancia de las características')

plt.show()  


# In[497]:


ct = pd.crosstab(val_credit_df_filtered[varDependiente].values, preds, rownames=['Actual'], colnames=['Predicted'])
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))

sns.heatmap(ct, 
            xticklabels=['Not Default', 'Default'],
            yticklabels=['Not Default', 'Default'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Darkblue", cmap="Blues")

plt.title('Matriz de Confusión', fontsize=14)
plt.show()


# In[498]:


roc_auc_score(val_credit_df_filtered[varDependiente].values, preds)


# In[499]:


## CatBoostClassifier
clf = CatBoostClassifier(iterations=500,
                         learning_rate=0.02,
                         depth=12,
                         eval_metric='AUC',
                         random_seed = 2018,
                         bagging_temperature = 0.2,
                         od_type='Iter',
                         metric_period = 50,
                         od_wait=100)

clf.fit(train_credit_df_filtered[varIndependientes], train_credit_df_filtered[varDependiente].values, verbose=True)


# In[500]:


preds = clf.predict(val_credit_df_filtered[varIndependientes])

temporal = pd.DataFrame({'Feature': varIndependientes, 'Feature importance': clf.feature_importances_})
temporal = temporal.sort_values(by='Feature importance',ascending=False)

plt.figure(figsize = (7,4))
plt.title('Importancia de las características',fontsize=14)

s = sns.barplot(x='Feature',y='Feature importance', data=temporal)
s.set_xticklabels(s.get_xticklabels(),rotation=90)

plt.xlabel('Características')
plt.ylabel('Importancia de las características')

plt.show()   


# In[501]:


ct = pd.crosstab(val_credit_df_filtered[varDependiente].values, preds, rownames=['Actual'], colnames=['Predicted'])
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(ct, 
            xticklabels=['No Fraudulento', 'Fraudulento'],
            yticklabels=['No Fraudulento', 'Fraudulento'],
            annot=True,
            ax=ax1,
            linewidths=.2,
            linecolor="Darkblue", 
            cmap="Blues")
plt.title('Matriz de Confusión', fontsize=10)
plt.show()


# In[502]:


roc_auc_score(val_credit_df_filtered[varDependiente].values, preds)


# In[503]:


## XGBoost
# Prepare the train and valid datasets
dtrain = xgb.DMatrix(train_credit_df_filtered[varIndependientes], train_credit_df_filtered[varDependiente].values)
dvalid = xgb.DMatrix(val_credit_df_filtered[varIndependientes], val_credit_df_filtered[varDependiente].values)

#What to monitor (in this case, **train** and **valid**)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

# Set xgboost parameters
params = {}
params['objective'] = 'binary:logistic'
params['eta'] = 0.039
params['silent'] = True
params['max_depth'] = 2
params['subsample'] = 0.8
params['colsample_bytree'] = 0.9
params['eval_metric'] = 'auc'
params['random_state'] = 2018

model = xgb.train(params, 
                  dtrain, 
                  1000, 
                  watchlist, 
                  early_stopping_rounds=50, 
                  maximize=True, 
                  verbose_eval=50)


# In[504]:


## Plot variable importance

fig, (ax) = plt.subplots(ncols=1, figsize=(8,5))
xgb.plot_importance(model, height=0.8, title="Importancia de las características (XGBoost)", ax=ax, color="green") 
plt.show()


# In[505]:


## LightGBM

params = {
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric':'auc',
          'learning_rate': 0.05,
          'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
          'max_depth': 4,  # -1 means no limit
          'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
          'max_bin': 100,  # Number of bucketed bin for feature values
          'subsample': 0.9,  # Subsample ratio of the training instance.
          'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
          'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
          'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
          'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
          'nthread': 8,
          'verbose': 0,
          'scale_pos_weight':50, # because training data is sightly unbalanced 
         }

categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE','PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

dtrain = lgb.Dataset(train_credit_df_filtered[varIndependientes].values, 
                     label = train_credit_df_filtered[varDependiente].values,
                     feature_name = varIndependientes,
                     categorical_feature = categorical_features)

dvalid = lgb.Dataset(val_credit_df_filtered[varIndependientes].values,
                     label = val_credit_df_filtered[varDependiente].values,
                     feature_name = varIndependientes,
                     categorical_feature = categorical_features)

## Run the model
evals_results = {}

model = lgb.train(params, 
                  dtrain, 
                  valid_sets=[dtrain, dvalid], 
                  valid_names=['train','valid'], 
                  evals_result=evals_results, 
                  num_boost_round=1000,
                  early_stopping_rounds=50,
                  verbose_eval=50, 
                  feval=None)

del dvalid
gc.collect()


# In[506]:


fig, (ax) = plt.subplots(ncols=1, figsize=(8,5))
lgb.plot_importance(model, height=0.8, title="Features importance (LightGBM)", ax=ax,color="red") 
plt.show()


# In[507]:


kf = KFold(n_splits = 5, random_state = 2018, shuffle = True)
for train_index, test_index in kf.split(credit_df_filtered):
    train_X, valid_X = credit_df_filtered.iloc[train_index], credit_df_filtered.iloc[test_index]

    dtrain = lgb.Dataset(train_X[varIndependientes].values, label=train_X[varDependiente].values,
                     feature_name=varIndependientes)

    dvalid = lgb.Dataset(valid_X[varIndependientes].values, label=valid_X[varDependiente].values,
                     feature_name=varIndependientes)

    evals_results = {}
    model =  lgb.train(params, 
                  dtrain, 
                  valid_sets=[dtrain, dvalid], 
                  valid_names=['train','valid'], 
                  evals_result=evals_results, 
                  num_boost_round=1000,
                  early_stopping_rounds=50,
                  verbose_eval=50, 
                  feval=None)


# In[508]:




