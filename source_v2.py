#!/usr/bin/env python
# coding: utf-8

# # Step1: Import necessary libraries for training, preprocessing, exploratory analysis, metrics ...

# In[209]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from warnings import filterwarnings
filterwarnings('ignore')


# In[210]:


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.autoscale()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# # Step2: Import data into dataframe

# In[211]:


train_features = pd.read_csv("./train/X_train.txt",header=None, delimiter=r"\s+")
test_features  = pd.read_csv("./test/X_test.txt",header=None, delimiter=r"\s+")


# In[212]:


train_features.info()


# In[213]:


test_features.info()


# # Data pre-processing

# <b>Check if any duplicate rows are available </b>

# In[214]:


train_features[train_features.duplicated()].count().sum()


# <b>Check if any null values are present and impute them with column mean (if the count is insignificant)  </b>

# In[215]:


Imputer = SimpleImputer(missing_values=np.nan, strategy="mean")


# In[216]:


train_features = Imputer.fit_transform(train_features.values)
train_features = pd.DataFrame(train_features)


# In[217]:


train_features.isnull().sum().sum()


# <b>Transform features by scaling each feature between [-1 1]</b>

# In[218]:


scaler = MinMaxScaler(feature_range=(-1,1))


# In[219]:


train_features = pd.DataFrame(scaler.fit_transform(train_features.values))


# In[220]:


train_features.describe()


# In[223]:


fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(train_features[0],train_labels)
plt.xlabel(" Feature Values ")
plt.ylabel(" Class Labels ")
plt.show()


# In[224]:


fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(train_features[51],train_labels)
plt.xlabel(" Feature Values ")
plt.ylabel(" Class Labels ")
plt.show()


# In[227]:


fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(train_features[280],train_labels)
plt.xlabel(" Feature Values ")
plt.ylabel(" Class Labels ")
plt.show()


# In[228]:


fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(train_features[560],train_labels)
plt.xlabel(" Feature Values ")
plt.ylabel(" Class Labels ")
plt.show()


# <b>Remove outliers using zscore, before removing check how many to delete</b>

# In[243]:


prev_rows = len(train_features)
data_copy = train_features.copy()
z_score = np.abs(stats.zscore(train_features))


# In[244]:


data_copy = data_copy[(z_score < 10).all(axis=1)]
print("Before removing outliers , rows - ", prev_rows)
print("After removing outliers , rows -", len(data_copy))
print("Number of records deleted - ", (prev_rows - len(data_copy)))


# <b> Repeat same preprocessing steps for test_features as well </b>

# In[245]:


test_features = Imputer.fit_transform(test_features.values)
test_features = pd.DataFrame(test_features)
# Gives a final sum across dataframe
test_features.isnull().sum().sum()


# In[246]:


scaler = MinMaxScaler(feature_range=(-1,1))
test_features = pd.DataFrame(scaler.fit_transform(test_features.values))
test_features.describe()


# In[248]:


train_labels=pd.read_csv("./train/y_train.txt",header=None)
train_labels.columns=['label']
test_labels=pd.read_csv("./test/y_test.txt",header=None)
test_labels.columns=['label']


# In[249]:


train_labels.info()


# In[250]:


test_labels.info()


# In[251]:


train_labels.describe()


# In[252]:


test_labels.describe()


# In[253]:


activities=['WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING']


# In[254]:


train_labels_copy = train_labels.copy()
train_labels_copy['label'] = train_labels_copy['label'].apply(lambda x: activities[x - 1])
test_labels_copy = test_labels.copy()
test_labels_copy['label'] = test_labels_copy['label'].apply(lambda x: activities[x - 1])


# In[255]:


train_labels_copy['label'].value_counts()


# In[256]:


test_labels_copy['label'].value_counts()


# In[257]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
axis = sns.countplot(x='label',data=train_labels_copy,)
axis.set_xticklabels(axis.get_xticklabels(), rotation=40)
plt.title("Training data count per class label")
plt.show()


# # Training models

# # 1. Logistic Regression

# In[95]:


#X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=0)


# In[70]:


lr_params = {'C':np.logspace(-1, 1, 10),
             'max_iter':[10,100]}
lr = LogisticRegression(solver='lbfgs')
gridcv_lr = GridSearchCV(lr, lr_params, verbose=1, cv=3)


# In[71]:



gridcv_lr.fit(train_features,train_labels['label'].ravel())


# In[72]:


gridcv_lr.best_params_


# In[73]:


lr_predictions = gridcv_lr.predict(test_features)


# In[74]:


print(classification_report(test_labels,lr_predictions))


# In[75]:


plot_confusion_matrix(confusion_matrix(test_labels,lr_predictions),
                      normalize    = False,
                      target_names = activities,
                      title        = "Confusion Matrix for logistic classifier")


# # 2. KNN

# In[258]:


knn_params = {'n_neighbors':np.array(range(9,18))}
knn = KNeighborsClassifier()
gridcv_knn = GridSearchCV(knn, knn_params, verbose=False, cv=3)


# In[259]:



gridcv_knn.fit(train_features,train_labels['label'].ravel())


# In[217]:


gridcv_knn.best_params_


# In[218]:


knn_predictions = gridcv_knn.predict(test_features)


# In[220]:


print(classification_report(test_labels,knn_predictions))


# In[221]:


plot_confusion_matrix(confusion_matrix(test_labels,knn_predictions),
                      normalize    = False,
                      target_names = activities,
                      title        = "Confusion Matrix for KNN")


# # 3.Random Forest

# In[89]:


rf_params = {'n_estimators' : np.arange(10,30,10),'max_depth' : np.arange(1,6,2)}
rf = RandomForestClassifier(random_state=0)
gridcv_rf = GridSearchCV(rf, rf_params, verbose=False, cv=3)


# In[90]:


gridcv_rf.fit(train_features,train_labels['label'].ravel())


# In[91]:


gridcv_rf.best_params_


# In[92]:


rf_predictions = gridcv_rf.predict(test_features)


# In[93]:


print(classification_report(test_labels,rf_predictions))


# In[127]:


plot_confusion_matrix(confusion_matrix(test_labels,rf_predictions),
                      normalize    = False,
                      target_names = activities,
                      title        = "Confusion Matrix for Random Forest")


# # selecting important features using random forest model

# In[260]:


new_rf = RandomForestClassifier(random_state=0,max_depth=5,n_estimators=20)


# In[261]:


new_rf.fit(train_features,train_labels['label'].ravel())


# In[262]:


newrf_predictions = new_rf.predict(test_features)


# In[263]:


print(classification_report(test_labels,newrf_predictions))


# In[265]:


imp_features= new_rf.feature_importances_


# In[266]:


remove_col = []
for i in range(len(imp_features)):
    if imp_features[i] == 0:
        remove_col.append(i)


# In[268]:


train_copy = train_features.copy()


# In[269]:



train_copy = train_copy.drop(remove_col,axis=1)


# In[270]:


train_copy.describe()


# In[134]:


train_copy.info()


# In[271]:


test_copy = test_features.copy()


# In[272]:


test_copy = test_copy.drop(remove_col,axis=1)


# In[273]:


test_copy.describe()


# In[274]:


test_copy.info()


# In[275]:


new_knn = KNeighborsClassifier(n_neighbors=17)
new_knn.fit(train_copy,train_labels['label'].ravel())


# In[276]:


new_knn_predictions = new_knn.predict(test_copy)
print(classification_report(test_labels,new_knn_predictions))


# In[278]:


rf_params = {'n_estimators' : np.arange(10,30,10),'max_depth' : np.arange(1,6,2)}
new_rf = RandomForestClassifier(random_state=0)
new_gridcv_rf = GridSearchCV(new_rf, rf_params, verbose=False, cv=3)


# In[279]:


new_gridcv_rf.fit(train_copy,train_labels['label'].ravel())


# In[280]:



new_gridcv_rf.best_params_


# In[281]:



newrf_predictions = new_gridcv_rf.predict(test_copy)


# In[282]:



print(classification_report(test_labels,newrf_predictions))


# In[ ]:





# # 4. Final result

# In[236]:


knn_res = [gridcv_knn.score(train_features,train_labels),
           gridcv_knn.score(test_features,test_labels),
           precision_score(test_labels,knn_predictions,average ='weighted'),
           recall_score(test_labels,knn_predictions,average ='weighted'),
           f1_score(test_labels,knn_predictions,average ='weighted')]


# In[265]:


result = pd.DataFrame(np.array(knn_res).reshape(-1,5))


# In[266]:


lr_res = [gridcv_lr.score(train_features,train_labels),
         gridcv_lr.score(test_features,test_labels),
         precision_score(test_labels,lr_predictions,average ='weighted'),
         recall_score(test_labels,lr_predictions,average ='weighted'),
         f1_score(test_labels,lr_predictions,average ='weighted')]
lr_res = pd.DataFrame(np.array(lr_res).reshape(-1,5))


# In[267]:


result = pd.concat([result,lr_res])


# In[268]:


rf_res = [gridcv_rf.score(train_features,train_labels),
       gridcv_rf.score(test_features,test_labels),
       precision_score(test_labels,rf_predictions,average ='weighted'),
       recall_score(test_labels,rf_predictions,average ='weighted'),
       f1_score(test_labels,rf_predictions,average ='weighted')]
rf_res = pd.DataFrame(np.array(rf_res).reshape(-1,5))


# In[269]:


result = pd.concat([result,rf_res])


# In[270]:


lab=['Train Accuracy','Test Accuracy','Precision','Recall','F1 score']
models =['KNN','Logistic Regression','Random Forest']
idx=[1,2,3]
result.index = [idx,models]
result.columns=lab


# In[271]:


result


# In[ ]:




