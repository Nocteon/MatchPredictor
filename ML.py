# Import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import packages for classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#Import the spreadsheet's data
match_data = pd.read_csv("match_data.csv")
"""
This spreadsheet contains data from the first 10 minutes of League of Legends matches.
For more info, visit the source at:
https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min
"""


#Combine Towers and Elite Monsters into total Objectives Taken
#Together, these provide a more inclusive look into a team's objective progression status
match_data['blueObjectives'] = match_data['blueTowersDestroyed'] + match_data['blueEliteMonsters']
match_data['redObjectives'] = match_data['redTowersDestroyed'] + match_data['redEliteMonsters']

#Calculate differences in team statistics for the modesls to predict on
match_data['blueObjDiff'] = match_data['blueObjectives'] - match_data['redObjectives']
match_data['blueExpDiff'] = match_data['blueTotalExperience'] - match_data['redTotalExperience']

# Testing Both Teams' Data
X = match_data[["blueGoldDiff","blueExpDiff","blueObjDiff"]]
# For Testing One Team's Data
#X = match_data[["blueTotalGold","blueTotalExperience","blueObjectives"]]
# For Testing all features in the Dataset
#md = match_data.drop("blueWins",axis=1)
#X = md

y = match_data[["blueWins"]]

#Splitting data into training/validation and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)

#Initializing classifier models
rfc = RandomForestClassifier(max_depth=3,min_samples_leaf=100,max_features=None)
#Max depth did not particularly affect accuracy from 1 - 20
#min_samples_leaf did not affect accuracy from 10 - 100
#max_features None to allow it to use all of the limited features used

knn = KNeighborsClassifier(n_neighbors=125,weights='distance')
#closer neighbors have more of an impact with weights set to distance
#higher k-values were more accurate up to a point of severe diminishing returns around k=71

#Fitting models
rfc.fit(X_train,np.ravel(y_train))
knn.fit(X_train,np.ravel(y_train))

#Setting predictions from the models
y_pred_rfc = rfc.predict(X_test)
y_pred_knn = knn.predict(X_test)

#Scores for the data
print('Random Forest Classifier Statistics')
accuracy_core_rfc = metrics.accuracy_score(y_pred_rfc, y_test)
print('Confusion Matrix:')
print(metrics.confusion_matrix(y_pred_rfc, y_test))
print('Accuracy score is ', end="")
print('%.3f' % accuracy_core_rfc)
mse_rfc = metrics.mean_squared_error(y_pred_rfc, y_test)
print('Error score is: ', end="")
print('%.3f\n' % mse_rfc)

print('K-Nearest Neighbors Statistics')
accuracy_core_knn = metrics.accuracy_score(y_pred_knn, y_test)
print('Confusion Matrix:')
print(metrics.confusion_matrix(y_pred_knn, y_test))
print('Accuracy score is ', end="")
print('%.3f' % accuracy_core_knn)
mse_knn = metrics.mean_squared_error(y_pred_knn, y_test)
print('Error score is: ', end="")
print('%.3f' % mse_knn)

#Box plot data visualization

#goldDiffCheck = match_data[['blueGoldDiff']]# Compare wins to gold difference
#objCheck = match_data[['blueObjDiff']]# Compare wins to gold difference
#expCheck = match_data[['blueExpDiff']]# Compare wins to gold difference

#boxplot = goldDiffCheck.boxplot(column='blueGoldDiff')
#boxplot2 = objCheck.boxplot(column='blueObjDiff')
#boxplot3 = expCheck.boxplot()
#plt.show()