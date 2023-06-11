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

# Encode categorical variables in the selected columns
for col in categorical_columns:
    train_data[col] = label_encoder.fit_transform(train_data[col])

#print(train_data.describe())

# Extract the input features and target variable
X_train = train_data.drop('IS_HCP', axis=1)  # Assuming 'IS_HCP' is the column name for the target variable
y_train = train_data['IS_HCP']

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
X_test = test_data.drop('IS_HCP', axis=1)  # Assuming 'IS_HCP' is the column name for the target variable
y_test = test_data['IS_HCP']

#model = LogisticRegression()
#model = HistGradientBoostingClassifier()
#model = DecisionTreeClassifier()
#model = RandomForestClassifier()
#model = SVC()
#model = GaussianNB()
#Accuracy for the MultinomialNB algorithm is 0.37053883834849544
#model = MultinomialNB()
#model = BernoulliNB()


#Accuracy for the KNeighborsClassifier algorithm 
# is 0.8931070678796361 with neighbors 4
model = KNeighborsClassifier(n_neighbors=4)

X_test = X_test.fillna(-1)
y_test = y_test.fillna(-1)


X_train = X_train.fillna(-1)
y_train = y_train.fillna(-1)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

result = test_data
result = result.filter(['ID','TAXONOMY','IS_HCP'])


result['IS_HCP'] = y_pred
result = result.filter(['ID','TAXONOMY','IS_HCP'])
result.to_csv('output.csv')