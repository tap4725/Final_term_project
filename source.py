#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import brier_score_loss, roc_auc_score, auc, roc_curve

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# In[59]:


df = pd.read_csv("data/loan_data.csv")


# # Data Visualization and Preprocessing

# In[60]:


df.describe()


# In[61]:


df.info()


# In[62]:


df["loan_status"].unique()


# In[63]:


df.isnull().any()


# In[64]:


df["loan_status"].value_counts()


# Sampling

# In[65]:


df = df.groupby('loan_status').sample(frac=0.2, random_state=42)
df["loan_status"].value_counts()


# In[66]:


features = df.drop("loan_status", axis = 1)


# In[67]:


df = df.reset_index().drop(["index"], axis = 1)
df


# In[68]:


encoder=LabelEncoder()
for col in df.columns[df.dtypes=='object']:
    df[col]=encoder.fit_transform(df[col])


# In[69]:


X_raw = df.drop("loan_status", axis = 1)
y = df["loan_status"]


# In[70]:


X = (X_raw - X_raw.mean()) / X_raw.std()


# In[71]:


fig, axis = plt.subplots(figsize=(8, 8))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=True, linewidths=.5, fmt='.2f', ax=axis)
plt.show()


# ### Histogram

# In[72]:


plt.figure(figsize=(15,15))
for ax, col in enumerate(X.columns):

    plt.subplot(5,4, ax+1)
    plt.title(col)
    sns.kdeplot(x=df[col],shade=True, hue=df["loan_status"])
    plt.legend()

plt.tight_layout()


# ### Density graph of labels.

# In[73]:


sns.kdeplot(y, shade = True)


# ## Matrics Calculations

# In[74]:


def matrics_cal(y_test, y_pred, y_proba = None):
    matrics = {}
    matrics["TP"] = sum(np.where(y_test & y_pred, 1, 0))
    matrics["TN"] = sum(np.where( (y_test == 0) & (y_pred == 0), 1, 0))
    matrics["FP"] = sum(np.where( (y_test == 0) & (y_pred == 1), 1, 0))
    matrics["FN"] = sum(np.where( (y_test == 1) & (y_pred == 0), 1, 0))

    matrics["TPR"] =  round(matrics["TP"] / (matrics["TP"] +  matrics["FN"]),3)
    matrics["TNR"] =  round(matrics["TN"] / (matrics["TN"] +  matrics["FP"]),3)
    matrics["FPR"] =  round(matrics["FP"] / (matrics["FP"] +  matrics["TN"]),3)
    matrics["FNR"] =  round(matrics["FN"] / (matrics["TP"] +  matrics["FN"]),3)

    matrics["Accuracy"] = round((matrics["TP"] + matrics["TN"]) / (matrics["TP"] + matrics["TN"] + matrics["FP"] + matrics["FN"]),3)
    matrics["Precision"] = round(matrics["TP"] / (matrics["TP"] +  matrics["FP"]),3)
    matrics["F1"] = 2 * round(((matrics["Precision"] * matrics["TPR"]) / (matrics["Precision"] + matrics["TPR"])),3)

    matrics["brier_score"] = round(brier_score_loss(y_test, y_proba),3)
    matrics["AUC"] =  round(roc_auc_score(y_test, y_proba),3)
    reference_prob = np.mean(y_test)
    reference_brier_score = brier_score_loss(y_test, [reference_prob] * len(y_test))
    matrics["BSS"] = round(1 - (matrics["brier_score"] / reference_brier_score),3)
		
    return matrics


# ## Common training function

# In[75]:


def train(clf, X, y):
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics_list = []
    
    for i, (train_index, test_index) in enumerate(kf.split(X), start=1):
        # Splitting the data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        clf.fit(X_train, y_train)
    
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        
        mat = matrics_cal(y_test, y_pred, y_pred_proba)
        print(f"Fold {i}: {mat}")
       
        metrics_list.append(mat)

    return metrics_list, y_pred_proba


# In[76]:


def plot_matrics(matrics):
    plt.figure(figsize=(15,15))
    for ax, col in enumerate(matrics.columns):
        plt.subplot(5,4, ax+1)
        plt.title(col)
        sns.lineplot(data=matrics, x=matrics.index, y=col)
        plt.xlabel("Folds")
        plt.legend()
    
    plt.tight_layout()


# ## Random Forest

# In[77]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)


# In[78]:

print("\nRendom Forest Training")
rnf_matrics, rnf_pred_proba = train(clf, X, y)


# In[79]:


rnf_matrics = pd.DataFrame(rnf_matrics)
rnf_matrics


# In[80]:


plot_matrics(rnf_matrics)


# In[81]:


avg_rnf_matrics = rnf_matrics.mean()


# Naive Bayes

# In[82]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

print("\nKNN Training")
knn_matrics, knn_pred_proba = train(knn, X, y)


# In[83]:


knn_matrics = pd.DataFrame(knn_matrics)
knn_matrics


# In[84]:


plot_matrics(knn_matrics)


# ## LSTM

# In[85]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=64,activation='relu', input_shape=(X.shape[1], 1), return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))  # Change based on your task
model.compile(optimizer=Adam(), loss='mean_squared_error')


kf = KFold(n_splits=10, shuffle=True, random_state=42)
metrics_list = []
print("\nLSTM Training")

for i, (train_index, test_index) in enumerate(kf.split(X), start=1):
	# Splitting the data
	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
	y_train, y_test = y.iloc[train_index], y.iloc[test_index]

	model.fit(X_train, y_train, validation_data=(X_test, y_test))

	lstm_pred_proba = model.predict(X_test)
	pred_labels = lstm_pred_proba > 0.5
	y_pred = pred_labels.astype(int).reshape(-1)

	mat = matrics_cal(y_test, y_pred, lstm_pred_proba)
	print(f"Fold {i}: {mat}")

	metrics_list.append(mat)

lstm_metrics = pd.DataFrame(metrics_list)
lstm_metrics
    


# In[86]:


plot_matrics(lstm_metrics)


# # Metrics Comparison

# In[87]:

print("\n")
print(rnf_matrics)


# In[88]:

print("\n")
print(knn_matrics)


# In[89]:

print("\n")
print(lstm_metrics)


# ### Average Comparison

# In[90]:

print("\n")
print(pd.concat([rnf_matrics.mean() , knn_matrics.mean(), lstm_metrics.mean()], axis = 1).rename(columns = {0: "Random Forest", 1: "KNN", 2: "LSTM"}))


# ### Fold wise comparison

# In[91]:


for i in range(10):
    print("\n")
    print(f"Fold :{i+1}")
    print(pd.DataFrame([rnf_matrics.iloc[i] , knn_matrics.iloc[i], lstm_metrics.iloc[i]]).reset_index().drop("index", axis = 1).T.rename(columns = {0: "Random Forest", 1: "KNN", 2: "LSTM"}))


# ### ROC curves

# In[92]:


def plot_roc_curve(y_test, pred_proba):
	fpr, tpr, thresholds = roc_curve(y_test, pred_proba)
	roc_auc = auc(fpr, tpr)

	# Plot the ROC curve
	plt.figure()
	plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic')
	plt.legend(loc='lower right')
	plt.show()



# Random Forest

# In[93]:


plot_roc_curve(y_test, rnf_pred_proba)


# KNN

# In[94]:


plot_roc_curve(y_test, knn_pred_proba)


# LSTM 

# In[95]:


plot_roc_curve(y_test, lstm_pred_proba)

