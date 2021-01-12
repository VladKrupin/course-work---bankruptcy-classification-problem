from scipy.io import arff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score, accuracy_score, auc
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


#reading the arff files
data, meta = arff.loadarff('1year.arff')
data2, meta2 = arff.loadarff('2year.arff')
data3, meta3 = arff.loadarff('3year.arff')
data4, meta4 = arff.loadarff('4year.arff')
data5, meta5 = arff.loadarff('5year.arff')

col_name=[]
for i in meta:
  col_name.append(i)

dataset=pd.DataFrame(data, columns=col_name)
dataset2=pd.DataFrame(data2, columns=col_name)
dataset3=pd.DataFrame(data3, columns=col_name)
dataset4=pd.DataFrame(data4, columns=col_name)
dataset5=pd.DataFrame(data5, columns=col_name)

#joining arff files into one dataframe  
dataset_full=pd.concat((dataset, dataset2, dataset3, dataset4, dataset5))
dataset_full['class'][dataset_full['class']==b'0'] = 0 
dataset_full['class'][dataset_full['class']==b'1'] = 1

#distribution of classes
plt.pie(dataset_full['class'].value_counts(),autopct='%.1f')
plt.title('Розподіл класів у % значенні')
plt.legend(labels=np.unique(dataset_full['class']))

#data preprocessing
dataset_full.info()
del dataset_full['Attr37']

#split dataset into two datasets (class 0 and class 1)
dataset_full_0=dataset_full[dataset_full['class']==0]
dataset_full_1=dataset_full[dataset_full['class']==1]
del dataset_full_0['class']
del dataset_full_1['class']

#replace NAN by mean for each class
for i in dataset_full_0.columns:
    dataset_full_0[i][dataset_full_0[i].isnull()]= dataset_full_0[i].mean()
    dataset_full_1[i][dataset_full_1[i].isnull()]= dataset_full_1[i].mean()

#delete the outliers (values that are bigger than 3 std)
dataset_Y_0 = dataset_full[dataset_full['class']==0]['class'][dataset_full_0.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
dataset_Y_1 = dataset_full[dataset_full['class']==1]['class'][dataset_full_1.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
dataset_full_0 = dataset_full_0[dataset_full_0.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
dataset_full_1 = dataset_full_1[dataset_full_1.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]

dataset_X=pd.concat((dataset_full_0, dataset_full_1))
Y=pd.concat((dataset_Y_0, dataset_Y_1))
Y.value_counts()


df=pd.concat((dataset_X, Y), axis=1)

#multicollinearity (delete values)
correlations_data_X=df.corr()
CorField = []
for i in correlations_data_X:
    for j in correlations_data_X.index[correlations_data_X[i] > 0.9]:
        if i != j and j not in CorField and i not in CorField:
            CorField.append(j)
            print ("%s-->%s: r^2=%f" % (i,j, correlations_data_X[i][correlations_data_X.index==j].values[0]))
for i in CorField:
  del df[i]

#normalization
sc = MinMaxScaler()
df.loc[:,df.columns[:-1]] = sc.fit_transform(df.loc[:,df.columns[:-1]])

X=df.iloc[:,:-1]
Y=df.iloc[:,df.shape[1]-1]

#split into the train and test
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, random_state=0)

#resampling
X = pd.concat([X_train, Y_train], axis=1)
not_bad=X[X['class']==0]
bad=X[X['class']==1]
bad_unsampled = resample(bad,
                          replace=True,
                          n_samples=len(not_bad),
                          random_state=27)

upsampled = pd.concat([not_bad, bad_unsampled])
X_train=upsampled.iloc[:,:-1]
Y_train=upsampled.iloc[:,-1]
Y_train=Y_train.astype(int)
Y_test=Y_test.astype(int)


plt.pie(Y_train.value_counts(),autopct='%.1f')
plt.title('Розподіл класів у % значенні')
plt.legend(labels=np.unique(dataset_full['class']))

#fitting the models (Logistic Regression, Random Forest, XGBoost)
######### model 1 - LogisticRegression
classifier_lr=LogisticRegression(max_iter=20000)
classifier_lr.fit(X_train, Y_train)
y_prob_lr=classifier_lr.predict_proba(X_test)
y_pred_lr=np.where(y_prob_lr>0.5, 1 , 0)
cm_lr=confusion_matrix(Y_test, y_pred_lr[:,1])
print('accuracy=',accuracy_score(Y_test, y_pred_lr[:,1]))
print('recall=',recall_score(Y_test, y_pred_lr[:,1]))
print('precision=',precision_score(Y_test, y_pred_lr[:,1]))
print('roc_auc=',roc_auc_score(Y_test, y_pred_lr[:,1])) 

########## model 2 - RandomForest
classifier_rf=RandomForestClassifier(n_estimators = 1000)
classifier_rf.fit(X_train, Y_train)
y_prob_rf=classifier_rf.predict_proba(X_test)
#y_pred_rf=classifier_rf.predict(X_test)
y_pred_rf=np.where(y_prob_rf>0.5, 1 , 0)
cm_rf=confusion_matrix(Y_test, y_pred_rf[:,1])
print('accuracy=',accuracy_score(Y_test, y_pred_rf[:,1]))
print('recall=',recall_score(Y_test, y_pred_rf[:,1]))
print('precision=',precision_score(Y_test, y_pred_rf[:,1]))
print('roc_auc=',roc_auc_score(Y_test, y_pred_rf[:,1])) 


######## model 3 - XGBoost
classifier_xgb=XGBClassifier(n_estimators=1000)
classifier_xgb.fit(X_train, Y_train)
y_prob_xgb=classifier_xgb.predict_proba(X_test)
y_pred_xgb=np.where(y_prob_xgb>0.5, 1 , 0)
cm_xgb=confusion_matrix(Y_test, y_pred_xgb[:,1])
print('accuracy=',accuracy_score(Y_test, y_pred_xgb[:,1]))
print('recall=',recall_score(Y_test, y_pred_xgb[:,1]))
print('precision=',precision_score(Y_test, y_pred_xgb[:,1]))
print('roc_auc=',roc_auc_score(Y_test, y_pred_xgb[:,1])) 

#plot ROC_AUC
fpr_lr, tpr_lr, _ = metrics.roc_curve(Y_test,  y_pred_lr[:,1])
fpr_rf, tpr_rf, _ = metrics.roc_curve(Y_test,  y_pred_rf[:,1])
fpr_xgb, tpr_xgb, _ = metrics.roc_curve(Y_test,  y_pred_xgb[:,1])
plt.plot(fpr_lr,tpr_lr,label="Logistic Regression, auc="+str(round(auc(fpr_lr, tpr_lr),3)))
plt.plot(fpr_rf,tpr_rf,label="Random Forest, auc="+str(round(auc(fpr_rf, tpr_rf),3)))
plt.plot(fpr_xgb,tpr_xgb,label="XGBoost, auc="+str(round(auc(fpr_xgb, tpr_xgb),3)))
plt.legend(loc=4)
plt.title('ROC_AUC ДЛЯ КОЖНОЇ МОДЕЛІ')
plt.show()

#distribution of probabilities
#Logisitic Regression
sns.kdeplot(y_prob_lr[:,0], shade=True, label='class 0')
sns.kdeplot(y_prob_lr[:,1], shade=True, label='class 1')
plt.title('Розподіл ймовірностей приналежності до класу 0 або 1 (LogReg)')

#Random Forest
sns.kdeplot(y_prob_rf[:,0], shade=True, label='class 0')
sns.kdeplot(y_prob_rf[:,1], shade=True, label='class 1')
plt.title('Розподіл ймовірностей приналежності до класу 0 або 1 (RF)')

#XGBoost
sns.kdeplot(y_prob_xgb[:,0], shade=True, label='class 0')
sns.kdeplot(y_prob_xgb[:,1], shade=True, label='class 1')
plt.title('Розподіл ймовірностей приналежності до класу 0 або 1 (XGBoost)')

#the most important features
feat_importances=pd.Series(list(classifier_xgb.feature_importances_), index=X_train.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.title('Вагомість предикторів за XGBoost')
