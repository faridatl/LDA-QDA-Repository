# -*- coding: utf-8 -*-
# =============================================================================
# Faridat Lawal

# The Iris Species dataset includes three types of iris species with 50 samples
# each as well as some properties about each flower. Therer are 4 features/att-
# ributes and three classes.

# A 5-Fold cross-validation to evaluate the classification performance of a 
# Linear Discriminant Analysis and Quadratic Discriminant Analysis classifier 
# will be carried out. The best classifier will be determined based on model
# performance as well as taking into account each classifiers Confusion Matrix,
# sensitivity, specificity, total accuracy, F1-score, ROC & AUC curve, and 
# overall model performance.
# =============================================================================

"""Libraries"""
import pandas as pd # save package name under variable name for easier access/usage
from sklearn.model_selection import KFold # split data into train and test data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis # import package/library to conduct linear and quadratic discriminant analysis classifier 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score # compute accuracy_score, classification report and confusion matrix for LDA and QDA classifier model; will aid in creation and calculation of auc and roc
import matplotlib.pyplot as plt # save package name as variable for easier acces/usage
from sklearn.preprocessing import LabelEncoder #import package to transform target classes into integer vlaues 
from sklearn.multiclass import OneVsRestClassifier # import store package to aid in plot for multiclass roc graph

"""Dataset"""
iris = pd.read_csv('/Users/faridatlawal/DTSC710/Assignment4/iris.csv') # read in dataset from file on computer and save to a variable name for access in python
le = LabelEncoder() # save function name for easier usuage 
iris['species']=le.fit_transform(iris['species']) # transform target variables to integers: Setosa is 0, Versicolor is 1 and Virginica is 2
iris.isnull().sum() # check dataset for any null values 

"""Split dataset into X and y for training and testing"""
X = iris.iloc[:,:-1] # creates a dataframe of all features needed for training and testing in the iris dataset this will only include 4 columns 
y = iris.iloc[:,-1] # creates a dataframe for target variable/cases 

"""K-Fold Cross Validation"""
k=5 # save number of folds to varaibel name for easier acces
cv = KFold(k, random_state=33, shuffle=True) # create 5-fold cross validation that will be used to train and test our feature and target datasets

"""Linear Discriminant Analysis"""
lda = LinearDiscriminantAnalysis() # save function name to variable for easier access

acc_score =[] # empty array that will hold accuracy score for each fold
for train_index, test_index in cv.split(X): # creates for loop that will loop through each instance in the feature dataset and apply Kfold function to create training and testing sets
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:] # creates X train and test datasets and stores them
    y_train, y_test = y[train_index], y[test_index] # creates y train and test datasets and stores them
    lda.fit(X_train, y_train) # fit the x and y training sets to the linear discriminant analysis classifier
    y_preds = lda.predict(X_test) # predict the y target varaible/value from the provided X testing set
    acc = accuracy_score(y_test, y_preds) # calculates the overall accuracy of the lda classifier model performance for each fold
    acc_score.append(acc) # appends/stores this accuracy value for each fold into empty array created above

acc_score # returns list of five overall model accuracies for each fold
avg_acc = sum(acc_score)/k # calculates and stores the overall average accuracy score of lda for all 5 folds
cm_lda = confusion_matrix(y_test, y_preds) # creates and stores the confusion matrix for the lda classifier
print(cm_lda) # prints contents of confusion matrix
print(classification_report(y_test, y_preds)) # prints classification report on lda model; includes Sensitivity and F1 scores for each class as well as precision, recall and model accuracy
print(avg_acc) # prints overall accuracy score for LDA model classifier

""" Assiging TP, TN, FN, & FP values to the proper class"""
TP1 = cm_lda[0,0] # true positive value for setosa from confusion matrix for access/usage
TP2 = cm_lda[1,1] # true positive value for versicolor from confusion matrix for access/usage
TP3 = cm_lda[2,2] # true positive value for virginica from confusion matrix for access/usage

FN1 = cm_lda[0,1] + cm_lda[0,2] # false negative value for setosa from confusion matrix for access/usage
FN2 = cm_lda[1,0] + cm_lda[1,2] # false negative value for versicolor from confusion matrix for access/usage
FN3 = cm_lda[2,0] + cm_lda[2,1] # false negative value for virginica from confusion matrix for access/usage

TN1 = TP2 + cm_lda[1,2] + cm_lda[2,1] + TP3 # true negative value for setosafrom confusion matrix for access/usage
TN2 = TP1 + cm_lda[0,2] + cm_lda[2,0] + TP3 # true negative value for versicolor from confusion matrix for access/usage
TN3 = TP1 + TP2 + cm_lda[0,1] + cm_lda[1,0] # true negative value for virginica from confusion matrix for access/usage

FP1 = cm_lda[1,0] + cm_lda[2,0] # false positive value for setosa from confusion matrix for access/usage
FP2 = cm_lda[0,1] + cm_lda[2,1] # false positive value for versicolor from confusion matrix for access/usage
FP3 = cm_lda[0,2] + cm_lda[1,2] # false positive value for virginica from confusion matrix for access/usage

""" Specificity TN/TN+FP"""
print("Class 1 Specificity:", TN1/(TN1 + FP1)) # calculates and prints the specificity rate of setosa for model performance based on values from confusion matrix
print("Class 2 Specificity:", TN2/(TN2 + FP2)) # calculates and prints the specificity rate of versicolor for model performance based on values from confusion matrix
print("Class 3 Specificity:", TN3/(TN3 + FP3)) # calculates and prints the specificity rate of virginica for model performance based on values from confusion matrix


""" Quadratic Discriminant Analysis"""
qda = QuadraticDiscriminantAnalysis() # save function name to variable for easier access

acc_score_qda = [] # empty array that will hold accuracy score for each fold
for train_index, test_index in cv.split(X): # creates for loop that will loop through each instance in the feature dataset and apply Kfold function to create training and testing sets
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:] # creates X train and test datasets and stores them
    y_train, y_test = y[train_index], y[test_index] # creates y train and test datasets and stores them
    qda.fit(X_train, y_train) # fit the x and y training sets to the quadratic discriminant analysis classifier
    y_preds_qda = qda.predict(X_test) # predict the y target varaible/value from the provided X testing set
    acc_qda = accuracy_score(y_test, y_preds_qda) # calculates the overall accuracy of the qda classifier model performance for each fold 
    acc_score_qda.append(acc_qda) # appends/stores this accuracy value for each fold into empty array created above

acc_score_qda # returns list of five overall model accuracies for each fold
avg_acc_qda = sum(acc_score_qda)/k # calculates and stores the overall average accuracy score of qda for all 5 folds
cm_qda = confusion_matrix(y_test, y_preds_qda) # creates and stores the confusion matrix for the lda classifier
print(cm_qda) # prints contents of confusion matrix
print(classification_report(y_test, y_preds_qda)) # prints classification report on qda model; includes Sensitivity and F1 scores for each class as well as precision, recall and model accuracy
print(avg_acc_qda) # prints overall accuracy score for QDA model classifier

""" Assiging TP, TN, FN, & FP values to the proper class"""
TP1_q = cm_qda[0,0] # true positive value for setosa from confusion matrix for access/usage
TP2_q = cm_qda[1,1] # true positive value for versicolor from confusion matrix for access/usage
TP3_q = cm_qda[2,2] # true positive value for virginica from confusion matrix for access/usage

FN1_q = cm_qda[0,1] + cm_lda[0,2] # false negative value for setosa from confusion matrix for access/usage
FN2_q = cm_qda[1,0] + cm_lda[1,2] # false negative value for versicolor from confusion matrix for access/usage
FN3_q = cm_qda[2,0] + cm_lda[2,1] # false negative value for virginica from confusion matrix for access/usage

TN1_q = TP2_q + cm_qda[1,2] + cm_qda[2,1] + TP3_q # true negative value for setosafrom confusion matrix for access/usage
TN2_q = TP1_q + cm_qda[0,2] + cm_qda[2,0] + TP3_q # true negative value for versicolor from confusion matrix for access/usage
TN3_q = TP1_q + TP2_q + cm_qda[0,1] + cm_qda[1,0] # true negative value for virginica from confusion matrix for access/usage

FP1_q = cm_lda[1,0] + cm_lda[2,0] # false positive value for setosa from confusion matrix for access/usage
FP2_q = cm_lda[0,1] + cm_lda[2,1] # false positive value for versicolor from confusion matrix for access/usage
FP3_q = cm_lda[0,2] + cm_lda[1,2] # false positive value for virginica from confusion matrix for access/usage

""" Specificity QDA: TN/TN+FP"""
print("Class 1 Specificity:", TN1_q/(TN1_q + FP1_q)) # calculates and prints the specificity rate of setosa for model performance based on values from confusion matrix
print("Class 2 Specificity:", TN2_q/(TN2_q + FP2_q)) # calculates and prints the specificity rate of versicolor for model performance based on values from confusion matrix
print("Class 3 Specificity:", TN3_q/(TN3_q + FP3_q)) # calculates and prints the specificity rate of virginica for model performance based on values from confusion matrix



""" ROC & AUC: LDA"""
lda_probs = lda.predict_proba(X_test) # create probablitlies for y predictions because this is a multiclass model
clf_lda = OneVsRestClassifier(lda) # save function name for easier access with lda as its estimator
clf_lda.fit(X_train, y_train) # fit the estimator for multiclass with X and y training sets
pred = clf_lda.predict(X_test) # predict the y outomes based on X testing dataset
pred_prob_lda = clf_lda.predict_proba(X_test) # calculate the probabilities of y predictions; probablilities are needed for multiclass

fpr ={} # creates empty list/array to hold false positive rate values
tpr = {} # creates empty list/array to hold true positive rate values
thr = {} # creates empty list/array to hold threshold values
n_class = 3 # create variable with number of classes stored 


for i in range(n_class): # creates for loop for that will loop only 3 times (becasue this is the number of classes we have for iris dataset)
    fpr[i], tpr[i], thr[i] = roc_curve(y_test, pred_prob_lda[:,i], pos_label=i) # calculates, creates and stores the roc, fpr, tpr and threshold for each class in the iris dataset
    

figdt, ax1 = plt.subplots() # create empty plot/graph
ax1.plot(fpr[0], tpr[0], color='orange', label='Setosa vs Rest', ) # create ROC for lda classifier Setosa class then add to graph 
ax1.plot(fpr[1], tpr[1], color='green', label='Versicolor  vs Rest') # create ROC for lda classifier Versicolor class then add to graph 
ax1.plot(fpr[2], tpr[2], color='blue', label='Virginica vs Rest') # create ROC for lda classifier Virginica class then add to graph 
ax1.plot([0,1],[0,1], linestyle='--') #plot baseline
ax1.set_title('Multiclass ROC curve') # add title to graph
ax1.set_xlabel('False Positive Rate') # add ylabel to graph
ax1.set_ylabel('True Positive rate') # add xlabel to graph
ax1.legend(loc='lower right') #add legend and set location as bottom right of graph
figdt # display/show the graph created

roc_auc_score(y_test, lda_probs, average='weighted', multi_class='ovr', labels=[0,1,2]) # calculates the AUC score for each class in the lda classifier 

# Note: LDA and QDA did not have a large difference in overall accuracy but based on their scores, LDA was the better model. 




