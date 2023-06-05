# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 11:57:01 2022

@author: 13347
"""



#Import relevant packages
import numpy as np
#import networkx
import matplotlib.pyplot as plt
import joblib
import sklearn
import pandas as pd
#import scprep
#import phate
import scipy
#import demap
from sklearn.metrics import roc_curve
import os
from scipy.stats import chi2
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import sys
#Importing logistic regression model
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import random 
os.chdir("C:\\Users\\13347\\Documents\\Yale\\MGH_Research\\Bortfield_Ali")
from base import *
from test import *
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

#Reading in Data
raw_data = pd.read_excel('Patient_and_Treatment_Characteristics.xls')
craw_data = pd.read_excel('Patient_and_Treatment_Characteristics.xls')

#--------------------# 
#----Data Preprocessing-----#
#--------------------#

#Dropping redundant or unnecessary columns
rdatacolumns = raw_data.columns
raw_data = raw_data.drop(columns = rdatacolumns[0])
raw_data = raw_data.drop(columns = rdatacolumns[3])
raw_data = raw_data.drop(columns = rdatacolumns[18])
rdatacolumns = raw_data.columns # Getting column names

#Changing Categorical data to numbered categorical
catlist = np.array([0,2,3,4,5,6,7,9,10,17,18,21,25,26,27,28,29,30,31,33,35,38,39,43,68,69,71,74,76,77,78,79])
for i in catlist:
    raw_data[rdatacolumns[i]] = raw_data[rdatacolumns[i]].astype('category')
    raw_data.iloc[:,i] = raw_data[rdatacolumns[i]].cat.codes

# Converting -1's back to nan's
for i in range(215):
    for j in range(80):
        if raw_data.iloc[i,j] == -1:
            raw_data.iloc[i,j] = np.nan

#Dropping Stop RT/Start RT and converting to one variable: Time to Start
nstart = pd.to_datetime(raw_data['Date of Diagnosis'].values) # Convert to datetime object
nstop = pd.to_datetime(raw_data['Date Start RT'].values) # Convert to datetime object
timedif = nstop - nstart
ntimedif = timedif/np.timedelta64(1,'D')
timetostart = ntimedif.astype(float)
dat2 = pd.DataFrame({'Time to Start Tx (Days)': timetostart})
raw_data = raw_data.join(dat2)


# Chagning Date Stop RT/date of recurrence to one variable: Time to recurrence 
nstart = pd.to_datetime(raw_data['Date Stop RT'].values) # Convert to datetime object
nstop = pd.to_datetime(raw_data['Date of recurrence'].values) # Convert to datetime object
timedif = nstop - nstart
ntimedif = timedif/np.timedelta64(1,'D')
timetorecur = ntimedif.astype(float)
dat2 = pd.DataFrame({'Time to Recurrence (Days)': timetorecur})
raw_data = raw_data.join(dat2)


# Converting start date to numbered categorical
start = raw_data['Date Start RT']
starter = start.values
nstart = pd.to_datetime(starter)
startdate = nstart.strftime("%Y")
nint = startdate.astype(int)
stdate = np.zeros((215))
a = 0
for i in range(215):
    
    if nint[i] < 2000:
        stdate[i] = 0
    
    if nint[i] > 1999 and nint[i] <2010:
        stdate[i] = 1
        
    if nint[i] > 2009:
        stdate[i] = 2
dat2 = pd.DataFrame({'Start Date': stdate})
raw_data = raw_data.join(dat2)


# Dropping more redundant categorical variables
raw_data = raw_data.drop(columns = 'CT sim date')
raw_data = raw_data.drop(columns = 'post-RT imaging date')
raw_data = raw_data.drop(columns = 'Pre-RT Imaging Date')
raw_data = raw_data.drop(columns = 'Date Start RT')
raw_data = raw_data.drop(columns = 'Date Stop RT')
raw_data = raw_data.drop(columns = 'Date of recurrence')
raw_data = raw_data.drop(columns = 'Date of Diagnosis')
raw_data = raw_data.drop(columns = 'Last Contact Date')
raw_data = raw_data.drop(columns = 'Recurrence imaging date')
raw_data = raw_data.drop(columns = 'Date Feeding tube placed')
raw_data = raw_data.drop(columns = 'Date Feeding tube removed')
raw_data = raw_data.drop(columns = 'Feeding tube note')
raw_data = raw_data.drop(columns = 'Follow up duration (month)')
raw_data = raw_data.drop(columns = 'Follow up duration (year)')
raw_data = raw_data.drop(columns = 'Overall Survival Censor')
raw_data = raw_data.drop(columns = 'Disease Specific Survival Censor')
raw_data = raw_data.drop(columns = 'Loco-regional Control Censor')


#Creating feature and target array
rdatacolumns = raw_data.columns 
labels = np.array([12,13,14,15,16,30,31,32,65]) # Manually selecting features

features = raw_data
for i in labels:
    print(rdatacolumns[i])
    features = features.drop(columns = rdatacolumns[i])
features = features.drop(columns = 'Height (m)') 

target = raw_data
for i in rdatacolumns:
    if i in np.array([rdatacolumns[labels]]):
        continue
    else:
        target = target.drop(columns = i)

#Changing 'Cause of Death' and 'Local Recurrence' Variables to be binary
for i in range(215):
    if target.iloc[i,2] == 1 or target.iloc[i,2] == 4:
        target.iloc[i,2] = 1
    else:
        target.iloc[i,2] = 0
 
for i in range(215):
    if target.iloc[i,4] == 2:
        target.iloc[i,4] = 1
    else:
        target.iloc[i,4] = 0
        

#Select categorical variables
columnfeatures = features.columns
catlist = np.array([0,2,3,4,5,6,7,8,9,12,13,14,15,16,17,18,20,22,23,24,46,47,50])
df_cat = features
for i in columnfeatures:
    if i in np.array([columnfeatures[catlist]]):
        continue
    else:
        df_cat = df_cat.drop(columns = i)
        
#Dropping columns with nan's       
df_cat = df_cat.drop(index = [95,110,71])
df_cat = df_cat.drop(columns = 'Recurrence imaging modality')


#Select numerical variables
numlist = np.array([1,11,19,21,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,54,55])
df_num = features
for i in columnfeatures:
    if i in np.array([columnfeatures[numlist]]):
        continue
    else:
        df_num = df_num.drop(columns = i)
        

# ******** --------- **********
#
# Here is where the two-sample analysis starts !!
#
# ******** --------- **********        
                

#Performing X^2
        

#Converting data to numpy arrays
df_catn = df_cat.to_numpy()
targetn = target.to_numpy()

# Defining X^2 function (NOTE: If there are Nan's, you'll run into trouble)
def chi_cat(predictor,targetc):
    
    table = np.zeros((len(np.unique(predictor)),2))
    x = np.unique(predictor) # Getting uniuqe values in predictor variables
    a = 0   
    
    for i in x:
        ind = np.where(predictor == i)# index where predictor variable equals unique value
        table[a,0] = len(np.where(targetc[ind[0]] == 1)[0])# number of target values of unique integer in predcitor variable = to 1
        table[a,1] = len(ind[0]) - len(np.where(targetc[ind[0]] == 1)[0]) # number of target values of unique integer in predictor variable= to 
        a = a + 1 # Iterating throught table

    stat, p, dof, expected = chi2_contingency(table)  
    return p


#Recording chi^2 values for Recurrence
catcolumns = df_cat.columns
catcol = df_cat.columns.to_numpy()
pvalREC = []
ncat = np.ma.size(df_cat,axis = 0)
for j in range(ncat):
    pvalREC.append(chi_cat(df_catn[:,j],targetn[:,2]))
    
#Storing sorted categorical variables for local recurrence
dfRECX = pd.DataFrame(np.sort(np.vstack((catcolumns,pvalREC)).T,axis = 0))


#Recording chi^2 values for Cause of Death
catcolumns = df_cat.columns
catcol = df_cat.columns.to_numpy()
pvalCOD = []
ncat = np.ma.size(df_cat,axis = 0)
for j in range(ncat):
    pvalCOD.append(chi_cat(df_catn[:,j],targetn[:,4]))
    
dfCODX = pd.DataFrame(np.sort(np.vstack((catcolumns,pvalCOD)).T,axis = 0))


# Performing Logistic regression - two-sample testing
df_num = df_num.drop(columns ='Time to Recurrence (Days)')
df_numn = df_num.to_numpy()

# Need to remove all the nan's from numerical variables in numpy array

ronan = np.argwhere(np.isnan(df_numn))[:,0] # Getting row where nan occured
print(ronan)


#df_numn = np.delete(df_numn, (54),axis = 0)
#targetn = np.delete(targetn, (54),axis = 0)
#df_numn = np.delete(df_numn, (28),axis = 0)
#targetn = np.delete(targetn, (28),axis = 0)
#df_numn = np.delete(df_numn, (211),axis = 0)
#targetn = np.delete(targetn, (211),axis = 0)
#df_numn = np.delete(df_numn, (210),axis = 0)
#targetn = np.delete(targetn, (210),axis = 0)


# Cause of Death
logcolumns = df_num.columns
logcol = df_num.columns.to_numpy()
logpvalCOD = []
for h in range(len(df_numn[0,:])):
    X = df_numn[:,h]
    y = targetn[:,2]
    X = sm.add_constant(X)
    model = sm.Logit(y,X)
    result = model.fit(method='newton')
    logpvalCOD.append((result.pvalues[1]))
       
dfCODlog = pd.DataFrame(np.sort(np.vstack((logcolumns,np.abs(logpvalCOD))).T,axis = 0))

#Local Recurrence
logpvalREC = []
for h in range(len(df_numn[0,:])):
    X = df_numn[:,h]
    y = targetn[:,4]
    X = sm.add_constant(X)
    model = sm.Logit(y,X)
    result = model.fit(method='newton')
    logpvalREC.append((result.pvalues[1]))
dfREClog = pd.DataFrame(np.sort(np.vstack((logcolumns,np.abs(logpvalREC))).T,axis = 0))



#### ****************************** ####
##---- Machine Learning Starts Here !!!--- ##
#### ****************************** ####

#Getting number of incidences:
nincdREC = np.where(targetn[:,4] == 1)
nRECind = len(nincdREC[0])
nincdCOD = np.where(targetn[:,2] == 1)
nCODind = len(nincdCOD[0])
# Recurrence has 20 incidences;Cause of Death has 55 incidences

# Defining target variables
CODtarg = targetn[:,2]
RECtarg = targetn[:,4]


#Sort recurrence/cause of death by pvalue

dfCOD = pd.concat([dfCODlog,dfCODX]).sort_values(by=1)
dfREC = pd.concat([dfREClog,dfRECX]).sort_values(by=1)



#Obtaining training, validation, and testing data


def process_train_data(numbpred,targetv,sortedfeat,predvar):
    
    """
    numpred = number of desired predictor variables
    targetv = specify target variable 
    sortedfeat = sorted data frame of features from two-sample testing
    predvar = dataframe of predictor variables
    
    """
    
    #Creating numpy data matrix in order of most importance based on pvalue
    X = np.empty((215,0))
    for i in range(numbpred): # NOTE:There are 48 useful predictor variables
        ind = np.where(predvar.columns == sortedfeat.iloc[i,0])[0]
        temp = predvar.iloc[:,ind[0]].to_numpy().reshape(215,1)
        X = np.append(X,temp,axis = 1)
    

    
    #Checking for nan's and deleting rows where nan's exist
    ronan = np.argwhere(np.isnan(X))[:,0] # Getting row where nan occured
    if len(ronan) >0:
        X = np.delete(X, (ronan),axis=0)
        targetv = np.delete(targetv, (ronan))

        
        
    #Shuffle observations in feature matrix and target vector seperately 
    indpos = np.argwhere(targetv == 1).squeeze()
    indneg = np.argwhere(targetv == 0).squeeze()
    random.shuffle(indpos)
    random.shuffle(indneg)
    
    #Defining percentages (test percentage can be inferred)
    trnper = 0.72
    vadper = 0.18
    
    #Get equal train/vad/test percent of BOTH positive and negative incidences
    
    postrain = round(trnper*len(indpos))
    negtrain = round(trnper*len(indneg))
    posvad = round(vadper*len(indpos))
    negvad = round(vadper*len(indneg))
    
    trnindpos = indpos[0:postrain]
    vadindpos = indpos[postrain:postrain+posvad]
    testindpos = indpos[postrain+posvad:-1]
    testindpos = np.append(testindpos,indpos[-1])# Appending last element (for some reason indexing doesn't grab)
    
    trnindneg = indneg[0:negtrain]
    vadindneg = indneg[negtrain:negtrain+negvad]
    testindneg = indneg[negtrain+negvad:-1]
    testindpos = np.append(testindpos,indneg[-1]) # Appending last element(for some reason indexing doesn't grab)
    
    #Concatenate positive/negative indeces
    trainind = np.concatenate((trnindpos,trnindneg))
    vadind = np.concatenate(([vadindpos,vadindneg]))
    testind = np.concatenate(([testindpos,testindneg]))
    
    #Create train, validation, and test data
    X_training = X[trainind,:]
    y_training = targetv[trainind]
    X_validation = X[vadind,:]
    y_validation = targetv[vadind]
    X_testing = X[testind,:]
    y_testing = targetv[testind]

    return X_training, y_training, X_validation, y_validation, X_testing, y_testing


X_train, y_train, X_vad, y_vad, X_test, y_test = process_train_data(dfREC.shape[0], RECtarg, dfREC, features)



#Begin Logistic Regression ML


#Model 1
clf = LogisticRegression(random_state=0,class_weight='balanced').fit(X_train[:,0:5], y_train)
pred_prob = clf.predict_proba(X_vad[:,0:5]) # Getting probabilities
#Get roc curve 
fpr,tpr,thresh = roc_curve(y_vad,pred_prob[:,1]) #NOTE need to feed probabilities of positive class
baseone = metrics.auc(fpr,tpr)
print("AUC",metrics.auc(fpr,tpr))

#Plotting
plt.figure()
plt.title("ROC Curve-Default (5 predictors)")
plt.plot(fpr,tpr)
plt.plot(fpr,fpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(["Logistic","No Skill"])


#Model 2
clf = LogisticRegression(random_state=0,class_weight='balanced',solver = 'saga',penalty='l1').fit(X_train[:,0:5], y_train)
pred_prob = clf.predict_proba(X_vad[:,0:5]) # Getting probabilities
#Get roc curve 
fpr,tpr,thresh = roc_curve(y_vad,pred_prob[:,1]) #NOTE need to feed probabilities of positive class
basetwo = metrics.auc(fpr,tpr)
print("AUC",metrics.auc(fpr,tpr))

#Plotting
plt.figure()
plt.title("ROC Curve-Default (5 predictors) - L1")
plt.plot(fpr,tpr)
plt.plot(fpr,fpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(["Logistic","No Skill"])


#Model 3
clf = LogisticRegression(random_state=0,class_weight='balanced',solver = 'liblinear',penalty='l1').fit(X_train[:,0:5], y_train)
pred_prob = clf.predict_proba(X_vad[:,0:5]) # Getting probabilities
#Get roc curve 
fpr,tpr,thresh = roc_curve(y_vad,pred_prob[:,1]) #NOTE need to feed probabilities of positive class
basethree = metrics.auc(fpr,tpr)
print("AUC",metrics.auc(fpr,tpr))

#Plotting
plt.figure()
plt.title("ROC Curve-Default (5 predictors) - L1")
plt.plot(fpr,tpr)
plt.plot(fpr,fpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(["Logistic","No Skill"])

#Model 4
clf = LogisticRegression(random_state=0,class_weight='balanced',penalty='l2').fit(X_train[:,0:5], y_train)
pred_prob = clf.predict_proba(X_vad[:,0:5]) # Getting probabilities
#Get roc curve 
fpr,tpr,thresh = roc_curve(y_vad,pred_prob[:,1]) #NOTE need to feed probabilities of positive class
basefour = metrics.auc(fpr,tpr)
print("AUC",metrics.auc(fpr,tpr))

#Plotting
plt.figure()
plt.title("ROC Curve-Default (5 predictors) - L2")
plt.plot(fpr,tpr)
plt.plot(fpr,fpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(["Logistic","No Skill"])


#Need to perform stratified K-fold between training/validation data for feature selection/hyperparameter tuning

#Merge training and validation data such that cross-validation can be performed

X_trainvad = np.concatenate((X_train,X_vad),axis=0)
y_trainvad = np.concatenate((y_train,y_vad),axis=0)
print(np.shape(X_trainvad))

"""
#Splitting based on percentages

#Model forward selection
maxAUCindfwd = []
skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
for train_index, vad_index, in skf.split(X_trainvad,y_trainvad):
    
    AUCforward = []
    for i in range(5,dfREC.shape[0]): 
        
        X_train, X_vad = X_trainvad[train_index,0:i], X_trainvad[vad_index,0:i]
        y_train, y_vad = y_trainvad[train_index], y_trainvad[vad_index]
        #!!!!!!!!!!!!!!!!!
        #NOTE 
        #Change model here
        #clf = LogisticRegression(random_state=0,class_weight='balanced').fit(X_train, y_train) # Model 1
        #clf = LogisticRegression(random_state=0,class_weight='balanced',solver = 'saga',penalty='l1').fit(X_train, y_train) # Model 2
        clf = LogisticRegression(random_state=0,class_weight='balanced',solver = 'liblinear',penalty='l1').fit(X_train, y_train) #Model 3
        #clf = LogisticRegression(random_state=0,class_weight='balanced',penalty='l2').fit(X_train, y_train) #Model 4
        #!!!!!!!!!!!!!!!

        pred_prob = clf.predict_proba(X_vad)# Getting probabilities
        fpr,tpr,thresh = roc_curve(y_vad,pred_prob[:,1])
        AUCforward.append(metrics.auc(fpr,tpr)) # Storing AUC values as a function of number of predictor variables
        
        
    maxAUCindfwd.append(np.argmax(AUCforward)+5)
    
round(np.mean(maxAUCindfwd))
optnumbfeatfwd = round(np.mean(maxAUCindfwd))
print("Average index of max AUC:",optnumbfeatfwd)


#Testing out forward selection with optimal number of features
#clf = LogisticRegression(random_state=0,class_weight='balanced').fit(X_train[:,0:optnumbfeatfwd], y_train) #Model 1
#clf = LogisticRegression(random_state=0,class_weight='balanced',solver = 'saga',penalty='l1').fit(X_train[:,0:optnumbfeatfwd], y_train) # Model 2
clf = LogisticRegression(random_state=0,class_weight='balanced',solver = 'liblinear',penalty='l1').fit(X_train[:,0:optnumbfeatfwd], y_train) # Model 3
#clf = LogisticRegression(random_state=0,class_weight='balanced',penalty='l2').fit(X_train[:,0:optnumbfeatfwd], y_train) #Model 4
pred_prob = clf.predict_proba(X_vad[:,0:optnumbfeatfwd])# Getting probabilitiese 
fpr,tpr,thresh = roc_curve(y_vad,pred_prob[:,1])

optfwdauc = metrics.auc(fpr,tpr)
print("Optimal AUC using forward selection",optfwdauc) # Storing AUC values as a function of number of predictor variables

        
plt.figure()
plt.title("ROC Curve-Optimized forwards selection")
plt.plot(fpr,tpr)
plt.plot(fpr,fpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(["Logistic","No Skill"])



# Backwards selection to determine optimal number of features
#Model backwards selection
maxAUCindback = []
for train_index, vad_index, in skf.split(X_trainvad,y_trainvad):
    
    
    AUCback = []
    for i in range(dfREC.shape[0],5,-1): 
                
        X_train, X_vad = X_trainvad[train_index,0:i], X_trainvad[vad_index,0:i]
        y_train, y_vad = y_trainvad[train_index], y_trainvad[vad_index]
        
        #!!!!!!!!!!!!!!!!!
        #NOTE 
        #Change model here
        #clf = LogisticRegression(random_state=0,class_weight='balanced').fit(X_train, y_train) #Model 1
        #clf = LogisticRegression(random_state=0,class_weight='balanced',solver = 'saga',penalty='l1').fit(X_train, y_train) #Model 2
        clf = LogisticRegression(random_state=0,class_weight='balanced',solver = 'liblinear',penalty='l1').fit(X_train, y_train) #Model 3
        #clf = LogisticRegression(random_state=0,class_weight='balanced',penalty='l2').fit(X_train, y_train) # Model 4
        #!!!!!!!!!!!!!!!


        pred_prob = clf.predict_proba(X_vad)# Getting probabilities
        fpr,tpr,thresh = roc_curve(y_vad,pred_prob[:,1])
        AUCback.append(metrics.auc(fpr,tpr)) # Storing AUC values as a function of number of predictor variables
        
    maxAUCindback.append(np.argmax(AUCback))
    
round(np.mean(maxAUCindback))
optnumbfeatback = round(np.mean(maxAUCindback))
print("Average index of max AUC:",optnumbfeatback)

#Testing out forward selection with optimal number of features
#clf = LogisticRegression(random_state=0,class_weight='balanced').fit(X_train[:,0:optnumbfeatback], y_train) # Model 1
#clf = LogisticRegression(random_state=0,class_weight='balanced',solver = 'saga',penalty='l1').fit(X_train[:,0:optnumbfeatback], y_train) #Model 2
clf = LogisticRegression(random_state=0,class_weight='balanced',solver = 'liblinear',penalty='l1').fit(X_train[:,0:optnumbfeatback], y_train) #Model 3
#clf = LogisticRegression(random_state=0,class_weight='balanced',penalty='l2').fit(X_train[:,0:optnumbfeatback], y_train) # Model 4
pred_prob = clf.predict_proba(X_vad[:,0:optnumbfeatback])# Getting probabilities
fpr,tpr,thresh = roc_curve(y_vad,pred_prob[:,1])

optbackauc = metrics.auc(fpr,tpr)
print("Optimal AUC using Backwards selection selection",optbackauc) # Storing AUC values as a function of number of predictor variables
        
plt.figure()
plt.title("ROC Curve-Optimized backwards selection")
plt.plot(fpr,tpr)
plt.plot(fpr,fpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(["Logistic","No Skill"])


#Getting optimal number of indeces from forward/backwards selection based on AUC
maxauc = np.argmax([optbackauc,optfwdauc])

if maxauc == 0:
    numbfeat = optnumbfeatback 
if maxauc == 1:
    numbfeat = optnumbfeatfwd

"""

# Performing forward/backward selection with no feature importance
    
from sklearn.feature_selection import SequentialFeatureSelector

#Model 1
clf = LogisticRegression(random_state=0,class_weight='balanced')
sfs = SequentialFeatureSelector(clf,n_features_to_select="auto",scoring='roc_auc')
sfs.fit(X_trainvad,y_trainvad)
findexmone = np.argwhere(sfs.get_support() == True)

sfs = SequentialFeatureSelector(clf,n_features_to_select=10,direction='backward')
sfs.fit(X_trainvad,y_trainvad)
bindexmone = np.argwhere(sfs.get_support() == True)

#Model 2















#Hyperparameter Tuning


#Define parameter grid dictionary
param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(1e-5, 100, ),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000, 2500, 5000]
    }
]




#Perform random search
#logistic = LogisticRegression(random_state=0,class_weight='balanced') # Model 1
#logistic = LogisticRegression(random_state=0,class_weight='balanced',solver = 'saga',penalty='l1') #Model 2
logistic = LogisticRegression(random_state=0,class_weight='balanced',solver = 'liblinear',penalty='l1') #Model 3
#logistic = LogisticRegression(random_state=0,class_weight='balanced',penalty='l2') #Model 4
clf = RandomizedSearchCV(logistic,param_grid,random_state=0,n_iter=60) # Prior from probablity theory tells us 60 iterations is optimal
search = clf.fit(X_trainvad[train_index,0:numbfeat],y_trainvad[train_index])
print("Best parameters",search.best_params_)


#Test optimized model with optimal hyperparameters/optimal number of predictor variables

#Model test
clf = LogisticRegression(random_state=0,class_weight='balanced',solver = search.best_params_["solver"],C = search.best_params_["C"],max_iter = search.best_params_["max_iter"],penalty = search.best_params_["penalty"]).fit(X_trainvad[train_index,0:numbfeat], y_trainvad[train_index])
pred_prob = clf.predict_proba(X_trainvad[vad_index,0:numbfeat]) # Getting probabilities
#Get roc curve 
fpr,tpr,thresh = roc_curve(y_trainvad[vad_index],pred_prob[:,1]) #NOTE need to feed probabilities of positive class
hypertunedauc = metrics.auc(fpr,tpr)
print("AUC",metrics.auc(fpr,tpr))

#Plotting
plt.figure()
plt.title("ROC Curve - Optimized number of features/optimized hyperparameters ")
plt.plot(fpr,tpr)
plt.plot(fpr,fpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(["Logistic","No Skill"])

bestAUC = np.max([baseone,basetwo,basethree,basefour,optbackauc,optfwdauc,hypertunedauc])
print(bestAUC)

sys.exit()

































































# Shuffling

#import random
#shfind = np.arange(len(X[:,0]))
#random.shuffle(shfind)

# Splitting into testing/training 
NX = X[shfind,:] # Selecting training observations
NY = targetn[shfind,2] # Selecting target observations (2 = Cause of Death)/(4 = Reccurrence)

nhold = int(.10*len(NX[:,0])) # Holding out 10 percent of data for testing
ntrainfull = (len(X[:,0]) - nhold) # Setting aside 90% of data for training
ntrain = int(.8*ntrainfull) # Selecting 80% of training data for actual training
nvad = ntrainfull - ntrain # Selecting 20% of training data for validation

# Getting data
x_test = NX[0:nhold,:]
x_train = NX[nhold:int(nhold + ntrain),:]
x_vad = NX[int(nhold+ntrain):-1,:]

# Getting target
y_test = NY[0:nhold]
y_train = NY[nhold:int(nhold + ntrain)]
y_vad = NY[int(nhold+ntrain):-1]



# Get AUC, sensitivity, accuracy
#fpr, tpr, thresholds = metrics.roc_curve(y_vad, predictions)

classifier = 0 #Variable to change if I want a classifier
clf = LogisticRegression(random_state=0,class_weight='balanced',solver='liblinear',penalty="l1").fit(x_train,y_train)
coef = clf.coef_

if classifier == 1:
    

    predictions = clf.predict(x_vad) 
    print(np.shape(predictions))

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for i in range(len(predictions)):
        if predictions[i] == 1 and y_vad[i] == 1:
            tp = tp + 1
        if predictions[i] == 1 and y_vad[i] == 0: 
            fp = fp + 1
        if predictions[i] == 0 and y_vad[i] == 0:
            tn = tn + 1
        if predictions[i] == 0 and y_vad[i] == 1:
            fn = fn + 1
    
    sensitivity = tp/(tp + fn)
    specificity = tn/(tn + fp)
    fpr = fp/(fp + tn)
    auc = roc_auc_score(predictions,y_vad)
    
    #Printing metrics
    print("Sensitivity:",sensitivity)
    print("Specificty:",specificity)
    print("AUC:", auc)
    
if classifier == 0:
    
    predictions = clf.predict_proba(x_vad)
    threshold = np.arange(.5,np.max(predictions[:,1]),.02)
    sensitivity = []
    specificity = []
    fpr = []
    auc = []
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for j in range(len(threshold)):
        
            
        npred = np.zeros(len(predictions))            
        tempred = predictions[:,1]
        
        for i in range(len(tempred)):
            if tempred[i] > threshold[j]:
                npred[i] = 1
        
        
        
        for i in range(len(npred)):
            
            
            
            if npred[i] == 1 and y_vad[i] == 1:
                tp = tp + 1
            if npred[i] == 1 and y_vad[i] == 0: 
                fp = fp + 1
            if npred[i] == 0 and y_vad[i] == 0:
                tn = tn + 1
            if npred[i] == 0 and y_vad[i] == 1:
                fn = fn + 1
        
            
        sensitivity.append(tp/(tp + fn))
        specificity.append(tn/(tn + fp))
        fpr.append(fp/(fp + tn))
        auc.append(roc_auc_score(npred,y_vad))
        
        
    print("Sensitivity:",sensitivity[0])
    print("Specificty:",specificity[0])
    print("AUC:", auc[0])
        

fpr = np.array(fpr)
sensitivity = np.array(sensitivity)



plt.figure()
fprind = np.argsort(fpr)
fpr = np.sort(fpr)
xdef = np.linspace(0,1)
#ydef = np.linspace(np.min(sensitivity[fprind]),np.max(fpr))
ydef = xdef

plt.title("Receiver Operating Characteristic Curve (Local Recurrence - 5 features)")
plt.plot(fpr,sensitivity[fprind])
plt.plot(xdef,ydef)
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.legend(["Logistic Regression","No skill"])

    
        
    
    
    
    
    
    

    
#print("Accuracy:", (len(np.where(predictions == y_vad)[0])/len(y_vad)) )


# Performing Grid Search Cross Validation for different models

"""

# Cross-validation python
logreg = LogisticRegression(random_state=0,class_weight='balanced')
#Defining grid of paramters
parameters = {
        'penalty' : ['l2','none'],
        'C' : np.logspace(-4,4,20),
        'solver' : ['newton-cg'],
        }
nclf = GridSearchCV(logreg,parameters,cv = 5)



tot_xtrain = np.concatenate((x_train,x_vad),axis=0)
tot_ytrain = np.concatenate((y_train,y_vad),axis=0)
nclf.fit(tot_xtrain,tot_ytrain)
print("Tuned parameters",nclf.best_params_)




#running best logistic regression model
best_params = nclf.best_params_
best_logreg = LogisticRegression(random_state=0,**best_params,class_weight = 'balanced')
# Fitting our model to the train set
fit_log = best_logreg.fit(tot_xtrain, tot_ytrain)
# Creating predicted variables to compare against y_test
predictions = fit_log.predict(x_test)

tp = 0
fp = 0
tn = 0
fn = 0

for i in range(len(predictions)):
    if predictions[i] == 1 and y_test[i] == 1:
        tp = tp + 1
    if predictions[i] == 1 and y_test[i] == 0: 
        fp = fp + 1
    if predictions[i] == 0 and y_test[i] == 0:
        tn = tn + 1
    if predictions[i] == 0 and y_test[i] == 1:
        fn = fn + 1

sensitivity = tp/(tp + fn)
specificity = tn/(tn + fp)
fpr = fp/(fp + tn)
auc = roc_auc_score(predictions,y_test)

#Printing metrics
print("Sensitivity:",sensitivity)
print("Specificty:",specificity)
print("AUC:", auc)
print("Accuracy:", (len(np.where(predictions == y_vad)[0])/len(y_vad)) )


"""



