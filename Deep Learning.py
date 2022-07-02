#build a  binary classification model for predicting arrival delay >15 min
# Deep Learning Model (Forward Neural Nework) :FNN
# without using departure delay as an input feature
#Read the Flight data and explore it using pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
flight=pd.read_csv("Covid-19 airline C&D.csv")
print(flight.head())
#clean and prepare the data
print(flight.shape)
#rows 1048575 and columns 47
print(flight.isna().sum())
print(flight.columns)
#wrangle data
#Handel class imbalance through sampling
class_distribution=flight["ARR_DEL15"].value_counts()
print("Class imbalance:")
print(class_distribution)
zero=flight[flight["ARR_DEL15"]==0].tail(class_distribution.min())
one=flight[flight["ARR_DEL15"]==1]
data=zero.append(one)
print(data.shape)
del zero,one
data.sort_values(['YEAR','MONTH','DAY_OF_MONTH','DAY_OF_WEEK'],ascending=[False,False,False,False])
print('Class imbalance evened out:')
print(data['ARR_DEL15'].value_counts())
print(len(data.columns))
print(data.head(2)) #showing top 2 rows as sampled data is ordered by  time with older data at the top
print(data.tail(2))#showing bottom 2 rows as sampled data is ordered by time with latest data at the bootom

#checking for missing values :
print(data.isna().sum())
remove=data.drop(['CANCELLATION_CODE','CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY'],
          axis=1, inplace=True) #drop the missing values columns
print("Dimension Reduced to:")
print(len(data.columns))
#sparsity per variable
print("Sparsity Per Variable")
print((len(data.index)-data.count())/len(data.index))
#to be excluded as per to the task
data.drop(['DEP_DELAY','DEP_DEL15','ARR_DELAY','DEP_DELAY_NEW','ARR_DELAY_NEW'],axis=1,inplace=True)
print(len(data.columns))
print(data.dtypes)
#Explantory Analysis
#Proportion of Late flight per category based on all other flights
print(len(data["ARR_DEL15"]))
avglate= np.sum(data["ARR_DEL15"])/len(data["ARR_DEL15"])
attributes=["MONTH","DAY_OF_WEEK","DAY_OF_MONTH","DEP_TIME_BLK","ARR_TIME_BLK","MKT_UNIQUE_CARRIER",
            "ARR_DELAY_GROUP","DEP_DELAY_GROUP"]
for i,pred in enumerate(attributes):
    plt.figure(i,figsize=(15,5))
    group=data.groupby([pred],as_index=False).aggregate(np.mean)[[pred,"ARR_DEL15"]]
    group.sort_values(by=pred,inplace=True)
    group.plot.bar(x=pred,y="ARR_DEL15")
    plt.axhline(y=avglate,label="Average")
    plt.ylabel("Percent of Flights that arrive late")
    plt.title(pred)
    plt.legend().remove()

#Label Encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["MKT_UNIQUE_CARRIER"]=le.fit_transform(data["MKT_UNIQUE_CARRIER"])
UniqueCarrier = list(le.classes_)
data["TAIL_NUM"]=le.fit_transform(data["TAIL_NUM"])
TailNum = list(le.classes_)
data["ORIGIN"] = le.fit_transform(data["ORIGIN"])
Origin = list(le.classes_)
data["ORIGIN_CITY_NAME"]=le.fit_transform(data["ORIGIN_CITY_NAME"])
OriginCityName = list(le.classes_)
data["ORIGIN_STATE_ABR"]=le.fit_transform(data["ORIGIN_STATE_ABR"])
OriginState = list(le.classes_)
data["ORIGIN_STATE_NM"]=le.fit_transform(data["ORIGIN_STATE_NM"])
OriginStateName = list(le.classes_)
data["DEST"]=le.fit_transform(data["DEST"])
Dest = list(le.classes_)
data["DEST_CITY_NAME"]=le.fit_transform(data["DEST_CITY_NAME"])
DestCityName = list(le.classes_)
data["DEST_STATE_ABR"]=le.fit_transform(data["DEST_STATE_ABR"])
DestState = list(le.classes_)
data["DEST_STATE_NM"]=le.fit_transform(data["DEST_STATE_NM"])
DestStateName = list(le.classes_)
data["DEP_TIME_BLK"]=le.fit_transform(data["DEP_TIME_BLK"])
DepTimeBlk = list(le.classes_)
data["ARR_TIME_BLK"]=le.fit_transform(data["ARR_TIME_BLK"])
ArrTimeBlk = list(le.classes_)
#Removing delay details as per the task
data.drop(['DEP_DELAY_GROUP','ARR_DELAY_GROUP'], axis=1, inplace=True)
print("Dimensions Reduced to :")
print(len(data.columns))
d=data.describe()
print(d)
print(data["ARR_DEL15"].describe())
print(data.shape)
# Removing the Target Variable into dependent one
from sklearn.model_selection import train_test_split
Delay_YN=data["ARR_DEL15"]
data.drop(['ARR_DEL15'],axis=1,inplace=True)#Removing target variable
data.drop(['FL_DATE','DEST_STATE_ABR','ORIGIN_STATE_ABR'],axis=1, inplace=True)
data.drop(['CANCELLED'],axis=1, inplace=True)
X_train,X_test,Y_train,Y_test=train_test_split(data,Delay_YN,test_size=0.2,random_state=42)
print(X_train.shape)
print(X_test.shape)
#Feature Scaling (within the same range)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)




#Build The deep Learning Model
from keras.models import Sequential
from keras.layers import Dense,Dropout
model=Sequential()
#Input Layer
model.add(Dense(units=20,activation='relu',input_dim=29))
model.add(Dropout(rate=0.1))
#Hidden Layer
model.add(Dense(units=20,activation='relu'))
model.add(Dropout(rate=0.1))
#Output Layer
model.add(Dense(units=1,activation='sigmoid'))
model.compile(optimizer='SGD',loss='binary_crossentropy',metrics=["accuracy"])
model.fit(X_train,Y_train,batch_size=10,epochs=200)
#(244204/10)*2*200=9,768,160 process will be done during the run (Forward and Backward)
y_pred=model.predict(X_test)
labels=[0,1]
y_pred=y_pred>0.5
from sklearn.metrics import confusion_matrix,roc_curve
cm=confusion_matrix(Y_test,y_pred)
print('Accuracy: ' + str(np.round(100*float(cm[0][0]+cm[1][1])/float((cm[0][0]+cm[1][1] + cm[1][0] + cm[0][1])),2))+'%')
print('Precsion: ' + str(np.round(100*float((cm[1][1]))/float((cm[0][1]+cm[1][1])),2))+'%')
print('Recall: ' + str(np.round(100*float((cm[1][1]))/float((cm[1][0]+cm[1][1])),2))+'%')
p=(cm[1][1])/((cm[0][1]+cm[1][1]))
r=(cm[1][1])/((cm[1][0]+cm[1][1]))
f1=2*(p*r)/(p+r)
print("F1-score:",f1*100)
print('Confusion matrix:')
print(cm)
fpr, tpr, _ = roc_curve(Y_test, y_pred)
auc = np.trapz(fpr,tpr)
print('Area under the ROC curve: ' + str(auc))
#Draw Confusion Matrix
fig=plt.figure(1)
ax=fig.add_subplot(111)
cax=ax.matshow(cm)
plt.title('Confusion matrix for Deep Learning Model (FNN)')
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
from itertools import product
for i,j in product(range(cm.shape[0]),range(cm.shape[1])):
    plt.text(j,i,format(cm[i,j],'d'),horizontalalignment='center',color='white',weight="bold")
plt.show()

#Draw the ROC Curve
fig = plt.figure(2)
plt.plot(fpr,tpr,color='green')
plt.xlabel('False positive rate (FPR)')
plt.ylabel('True positive rate (TPR)')
plt.title('Receiver operating characteristic (ROC)')


#Actual Vs Predicted table
error=pd.DataFrame(np.array(Y_test).flatten(),columns=['actual'])
error['predicted']=np.array(y_pred)
print(error)

from keras.wrappers.scikit_learn \
    import KerasClassifier
from sklearn.model_selection import GridSearchCV
#grid Search(Define search space as a grid of hyperparameter values and evaluate every position in the grid)
def build_cls(optimizer):
    model = Sequential( )
    # Input Layer
    model.add(Dense(units=20, activation='relu', input_dim=29))
    model.add(Dropout(rate=0.1))
    # Hidden Layer
    model.add(Dense(units=20, activation='relu'))
    model.add(Dropout(rate=0.1))
    # Output Layer
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])
    return model

gs_clf=KerasClassifier(build_fn=build_cls)
params={'batch_size':[10,25,50],
        'epochs':[50,100,200],
        'optimizer':['adam','SGD']}

#Keras Classifier : Compatability between keras and sklearn
gs=GridSearchCV(estimator=gs_clf,param_grid=params,scoring='accuracy',cv=10)
gs=gs.fit(X_train,Y_train)

print(gs.best_params_)
print(gs.best_score_)
