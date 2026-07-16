import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#----------------------- Dataset trining Library -------------------------------- 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder #used to convert strings to unique integer


# ---------------------------------------------------------XXXX-------------------------------

train_pd=pd.read_csv('./train.csv')
test_pd=pd.read_csv('./test.csv')
# print(train_pd.head(10))


survived =train_pd['Survived']
train_pd = train_pd.drop(['Survived'],axis=1)
# print(survived)


train_index =train_pd['PassengerId']
test_idx=test_pd['PassengerId']



# ################################# REMOVING NA VALUES #####################################3




combined_pd= pd.concat([train_pd,test_pd]).reset_index(drop =True)
# print(combined_pd)


# print("Null Values:")
# print(combined_pd.isnull().sum())

y=combined_pd['Age'].value_counts()
x=y.index.values
# print(y)
# print(x)


""" plt.figure(figsize=(10,10))
plt.bar(x,y)
plt.xlabel('Ages')
plt.ylabel('Index Values')
plt.title("All Ages")
plt.show() """


class_age=combined_pd[['Pclass','Age']]
# fig, axes = plt.subplots(2, 2, figsize=(4, 4))
""" for i in range(1,4):
  plt.figure(figsize=(4,4))
  plt.title("Class" + str(i))
  y=class_age[class_age['Pclass']==i]['Age'].value_counts()
  x=y.index.values
  plt.bar(x,y)
plt.show()
 """

medians = class_age.groupby('Pclass').median()
# print(medians)

for i in range(3):
  idx=np.where( (combined_pd['Pclass']==i+1) & (combined_pd['Age'].isnull()))[0]
  combined_pd.loc[idx,'Age'] = medians.values[i][0] 
  
# print(combined_pd[combined_pd['Fare'].isnull()])

# -------------------removing fares NA values------------------------
similar_fares=combined_pd[ (combined_pd['Pclass'] == 3) & ( combined_pd['Embarked'] == 'S') ]['Fare']
"""
# print(similar_fares)
# print("Medians", similar_fares.median()) """

""" plt.hist(similar_fares)
# plt.show()
"""
combined_pd['Fare']=combined_pd['Fare'].fillna(similar_fares.median())

# print(combined_pd[combined_pd['Fare']].isnull())

# --------------------Removing EMbarked NA---------------
# print(combined_pd[ combined_pd['Embarked'].isnull() ] )
similar_embarked= combined_pd[( combined_pd['Fare']>=70 ) & (combined_pd['Fare'] <= 90 ) & (combined_pd['Pclass'] == 1)] 
# print(similar_embarked['Embarked'].groupby(similar_embarked['Embarked']).value_counts())
combined_pd['Embarked']=combined_pd['Embarked'].fillna('C')
# print(combined_pd[combined_pd['PassengerId']==62],'\n',combined_pd[combined_pd['PassengerId']==830])
# print()

#---------------------Removing Ticket NA---------------------------

# print(combined_pd['Ticket'].isnull().value_counts())
combined_pd['Cabin']=combined_pd['Cabin'].fillna('M')
combined_pd['Cabin']= combined_pd['Cabin'].str[0]
idx=np.where(combined_pd['Cabin']=='T')[0]
combined_pd.loc[idx,'Cabin'] = 'M'
# print(combined_pd['Cabin'].value_counts())
# print(combined_pd.isnull().sum()) 


################################################## MAKING DATASET FURTHER TO LOOK EASY TO CREATE A MODEL########################################
# --------------------------------------------GROUPING TITLES OF NAMES-----------------------------------------------------
# print(combined_pd.nunique()) #unique()->number of uniques
# print(combined_pd['Pclass'].unique())
# print(combined_pd['Name'])
names=combined_pd['Name']
# print(combined_pd['Name'].unique())
last_names=[]
first_names=[]
titles=[]
for name in names:
  if ', ' not in name:
    last_names.append("")
  else:
    last, name=name.split(', ',1)
    last_names.append(last)
  
  if '. ' not in name:
    titles.append("")
  else:
    title,first= name.split('.',1)
    titles.append(title)
    first_names.append(first)

last_names=np.array(last_names)
first_names=np.array(first_names)
titles=np.array(titles)
# print()

idx=np.where(np.isin(titles,['Capt','Col','Major']))
titles[idx] = 'Military'

idx=np.where(np.isin(titles,['Don','Dona','Jonkheer','Lady','Sir','Master','the Countess']))
titles[idx] = 'Nobility'

idx=np.where(np.isin(titles,['Miss','Mlle','Ms']))
titles[idx] = 'Ms'

idx=np.where(np.isin(titles,['Mrs','Mme']))
titles[idx] = 'Mrs'
# print(np.unique(titles, return_counts=True))

combined_pd['Title'] = titles
# print(combined_pd['Title'].value_counts())


# ------------------------------------------------ADJUSTING AGES -----------------------------------------------
# print(combined_pd['Age'].min(), combined_pd['Age'].max())

bins =np.array([0,10,20,30,40,50,60,70,80])
 
combined_pd['Age_Bins'] = pd.cut(combined_pd['Age'],bins) #pd.cut() is used to divide continious neuerical data into intervals(bins) and assigns each value to category.

# ---------------------------------------------------Tickets Adjustments----------------------------------------------
tickets_dict=dict(combined_pd['Ticket'].value_counts())
combined_pd['Tkt_count'] = combined_pd['Ticket'].map(tickets_dict)

combined_pd['Fare_per_ticket'] = combined_pd['Fare'] / combined_pd['Tkt_count']

# print(combined_pd['Fare_per_ticket'].min(),combined_pd['Fare_per_ticket'].max())

bins = np.array([0,20,40,60,80,150])

combined_pd['Fare_bins'] = pd.cut(combined_pd['Fare_per_ticket'],bins)


combined_pd['Num_family'] = combined_pd['SibSp'] + combined_pd['Parch'] +1


combined_pd.drop(['Name', 'Age', 'Ticket', 'Fare', 'Tkt_count','Fare_per_ticket'],axis=1, inplace=True)

# --------------- Label Encoding ---------------------
combined_pd.set_index('PassengerId', inplace = True)
# print(combined_pd)

label_enc =combined_pd.copy()
label_enc=label_enc.astype(str)

label_enc =label_enc.apply(LabelEncoder().fit_transform)
# print(label_enc)


one_hot = label_enc.copy()
one_hot = pd.get_dummies(one_hot,columns=['Sex','Embarked','Title'])

# print(one_hot)

x = one_hot.loc[train_index].values
y=survived.values

scaler = MinMaxScaler()
scaler.fit(x)
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = .2, random_state=0)

# print(x_train.shape,x_test.shape, y_train.shape, y_test.shape)


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Random Forest Algorithm XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
"""
# -------------------Training Classifier------------------------

classifier = RandomForestClassifier(random_state = 0)
print("Random Forest Classifier score: ",classifier.fit(x_train, y_train).score(x_test, y_test))

#------------------ Hyper Parameters ------------------------

params = {
   'criterion':['gini','entropy'],
   'n_estimators':[20,50,100,200,300,400,500,800,1000],
   'max_depth':np.arange(3,50),
   'min_samples_split':np.arange(1,1000),
   'max_features':['sqrt','log2'],
   'max_samples':np.linspace(0.1,0.9,10)
 }
# -------------------------------- Using RanomisedSearchCV for more accuracy ------------------------------------------------
rand_search = RandomizedSearchCV(RandomForestClassifier(random_state=0), params,scoring = 'accuracy', random_state = 0, cv = 5) #CV= cross validation
rand_search.fit(x_train, y_train)

rand_params = rand_search.best_params_
print(rand_params, '\n')

print('Train Accuracy: ', rand_search.best_score_)

preds = rand_search.predict(x_test)

print("Test Accuracy : ", accuracy_score(preds, y_test))


# ------------------------- Using GridSearchCV For better accuracy -------------------------

n_estimators = np.linspace(rand_params['n_estimators']-10, rand_params['n_estimators']+10 , 3).astype(int)

min_samples_split = np.arange(rand_params['min_samples_split']-3, rand_params['min_samples_split']+3).astype(int)

max_samples = np.linspace(rand_params['max_samples']-.05, rand_params['max_samples']+.05, 6)

max_depth = np.arange(rand_params['max_depth'] - 5, rand_params['max_depth']+5).astype(int)

params ={
  'criterion':[rand_params['criterion']],
  'n_estimators':n_estimators,
  'max_depth':max_depth,
  'min_samples_split':min_samples_split,
  'max_features':[rand_params['max_features']],
  'max_samples':max_samples
}

grid_search = GridSearchCV(RandomForestClassifier(random_state = 0), params, scoring = 'accuracy' , cv =5)
grid_search.fit(x_train,y_train)

grid_params = grid_search.best_params_
print(grid_params,'\n')

print('Training Accuracy  using Grid Search cross validation:: ', grid_search.best_score_)
preds = grid_search.predict(x_test)

print('Test Accuracy using Grid Search cross validation: ', accuracy_score(preds, y_test))

# print(classification_report(y_test, preds)) """


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX USING ADABOOST (Model) ALGORITHMS XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

classifier = AdaBoostClassifier(random_state = 0)
classifier.fit(x_train, y_train).score(x_test, y_test)

params = {
  'n_estimators':[50,100,200,300,400,500,800,1000],
  'learning_rate':[1.0, 0.1,0.01, 0.001]
}

rand_search = RandomizedSearchCV(AdaBoostClassifier(random_state = 0), params, scoring ='accuracy', random_state = 0, cv =5)
rand_search.fit(x_train, y_train)

rand_params =rand_search.best_params_
print(rand_params, '\n')

print('Train Accuracy: ',rand_search.best_score_)
preds = rand_search.predict(x_test)
print('Test Accuracy: ',accuracy_score(preds, y_test))
# -----------------------------------------------------------------------

n_estimators = np.linspace(rand_params['n_estimators']-50, rand_params['n_estimators']+50, 3).astype(int)
learning_rate = np.linspace(rand_params['learning_rate']*.9, rand_params['learning_rate']*1.1, 3)

params ={
  'n_estimators':n_estimators,
  'learning_rate':learning_rate
}

grid_search = GridSearchCV(AdaBoostClassifier(random_state = 0),params, scoring ='accuracy', cv =5)
grid_search.fit(x_train, y_train)
grid_params = grid_search.best_params_
print(grid_params, '\n')

print('Grid Train Accuracy: ',grid_search.best_score_)
preds = grid_search.predict(x_test)
print('Grid Test Accuracy: ',accuracy_score(preds, y_test))









