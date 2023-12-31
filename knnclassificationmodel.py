import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

# Load the training data from CSV
train_data = pd.read_csv('Doceree-HCP_Train.csv',encoding = 'latin-1')
train_data.dropna()

# Select the column(s) containing categorical variables
categorical_columns = ['ID','DEVICETYPE', 'BIDREQUESTIP','USERCITY', 'USERPLATFORMUID',
                       'USERAGENT','USERZIPCODE','PLATFORMTYPE','CHANNELTYPE',
                       'URL','KEYWORDS','TAXONOMY','IS_HCP'
]

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

dictionary_taxonomy = {}

# Encode categorical variables in the selected columns
for col in categorical_columns:
    value = train_data['TAXONOMY']
    if col == 'TAXONOMY':
        #print(train_data['TAXONOMY'].size)
        for index in range(value.size):
            #if train_data['TAXONOMY'][index] != 'NaN':
            dictionary_taxonomy.setdefault(value[index],index)
    train_data[col] = label_encoder.fit_transform(train_data[col])

#print(dictionary_taxonomy)

# Extract the input features and target variable
X_train_hcp = train_data.drop('IS_HCP', axis=1)  # Assuming 'IS_HCP' is the column name for the target variable
y_train_hcp = train_data['IS_HCP']

X_train_taxonomy = train_data.drop('TAXONOMY', axis=1)  # Assuming 'IS_HCP' is the column name for the target variable
y_train_taxonomy = train_data['TAXONOMY']

# Load the testing data from CSV
test_data = pd.read_csv('Doceree-HCP-Test.csv',encoding = 'latin-1')
test_data.dropna()

# Select the column(s) containing categorical variables
categorical_columns = ['ID','DEVICETYPE', 'BIDREQUESTIP','USERCITY', 'USERPLATFORMUID',
                       'USERAGENT','USERZIPCODE','PLATFORMTYPE','CHANNELTYPE',
                       'URL','KEYWORDS','TAXONOMY','IS_HCP'
]

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Encode categorical variables in the selected columns
for col in categorical_columns:
    test_data[col] = label_encoder.fit_transform(test_data[col])

# Extract the input features and target variable
#print(test_data.head())
X_test_hcp = test_data.drop('IS_HCP', axis=1)  # Assuming 'IS_HCP' is the column name for the target variable
y_test_hcp = test_data['IS_HCP']

X_test_taxonomy = test_data.drop('TAXONOMY', axis=1)  # Assuming 'IS_HCP' is the column name for the target variable
y_test_taxonomy = test_data['TAXONOMY']

#model_hcp = LogisticRegression()
#model_hcp = HistGradientBoostingClassifier()
#model_hcp = DecisionTreeClassifier()
#model_hcp = RandomForestClassifier()
#model_hcp = SVC()
#model_hcp = GaussianNB()
#Accuracy for the MultinomialNB algorithm is 0.37053883834849544
#model_hcp = MultinomialNB()
#model_hcp = BernoulliNB()


#Accuracy for the KNeighborsClassifier algorithm 
# is 0.8931070678796361 with neighbors 4
max_accuracy_hcp = 0
max_accuracy_taxonomy = 0
index_hcp = 0
index_taxonomy = 0
#for i in range(1000):
model_hcp = KNeighborsClassifier(n_neighbors=1000)
model_taxonomy = KNeighborsClassifier(n_neighbors=1000)

X_test_hcp = X_test_hcp.fillna(-1)
y_test_hcp = y_test_hcp.fillna(-1)

X_test_taxonomy = X_test_taxonomy.fillna(-1)
y_test_taxonomy = y_test_taxonomy.fillna(-1)


X_train_hcp = X_train_hcp.fillna(-1)
y_train_hcp = y_train_hcp.fillna(-1)

X_train_taxonomy = X_train_taxonomy.fillna(-1)
y_train_taxonomy = y_train_taxonomy.fillna(-1)

# Train the model on the training data
model_hcp.fit(X_train_hcp, y_train_hcp)

# Make predictions on the test data
y_pred_hcp = model_hcp.predict(X_test_hcp)

# Calculate the accuracy of the model
max_accuracy_hcp = accuracy_score(y_test_hcp, y_pred_hcp)
    #print(f'Accuracy: {accuracy}')

# Train the model on the training data
model_taxonomy.fit(X_train_taxonomy, y_train_taxonomy)

# Make predictions on the test data
y_pred_taxonomy = model_taxonomy.predict(X_test_taxonomy)

# Calculate the accuracy of the model
max_accuracy_taxonomy = accuracy_score(y_test_taxonomy, y_pred_taxonomy)
    #print(f'Accuracy: {accuracy}')



    # if(accuracy_hcp >= max_accuracy_hcp):
    #     max_accuracy_hcp = accuracy_hcp
    #     index_hcp = i
    #     if( max_accuracy_hcp == 1.0):
    #         break

    # if(accuracy_taxonomy >= max_accuracy_taxonomy):
    #     max_accuracy_taxonomy = accuracy_taxonomy
    #     index_taxonomy = i
    #     if( max_accuracy_taxonomy == 1.0):
    #         break

result = test_data
result['TAXONOMY'] = y_pred_taxonomy
result = result.filter(['ID','TAXONOMY','IS_HCP'])


result['IS_HCP'] = y_pred_hcp
result = result.filter(['ID','TAXONOMY','IS_HCP'])
print(f'Max Accuracy for hcp: {max_accuracy_hcp}')
print(f'Max Accuracy for taxonomy: {max_accuracy_taxonomy}')
result.to_csv('output.csv')
