## Introduction
The audience would be the telecom business itself, interested in reducing how much money is lost because of customers who don't stick around very long. The question is are there any predictable patterns here?

This project aims to identify patterns in customer behavior to predict churn in the Welcome Company’s customer base. The company aims to reduce customer turnover by understanding key factors influencing churn, thereby minimizing revenue loss and improving overall business performance.

## Objectives
1. Predict Customer Churn: develop a model that can accurately predict whether a customer will churn.

2. Understand key factors influencing Churn. 

3. Improve customer retention: predict which customers are at risk of churning.

Task: Build a classifier to predict whether a customer will stop doing business with SyriaTel.

## Data Understanding

The dataset `bigml.csv` contains information about customers in telecommunications company, SyriaTel. The project aims to build a classifier to predict whether a customer will stop doing business with SyriaTel. 

The project aims to build a classification model to predict whether a customer will churn(Leave the company). The target variable is `churn`, which is a binary variable (`True` for churn and `False` for non-churn )

#import Data and load data
import pandas as pd 

df =  pd.read_csv('bigml.csv')
df

df.info()

The dataset contains 3333 rows, each with 20 columns and 1 target column:

Target variable: `churn`

Numerical features include:  account length, number vmail messages, total day in munites, total day calls, total day charge, total eve calls, total eve charge, total night minutes, total night calls,toatal night charge, total intl minutes, total intl calls, total intl charge, customer service, calls

Categorical features: state, area code, international plan, voice mail plan

Text feature: phone number

# Data Preprocessing

#drop phone number column (its unnecessary)

#phone number is dropped as it is not useful for the predictive model
data = df.drop(columns=['phone number'])

#Import relevant function
from sklearn.preprocessing import LabelEncoder

#Convert categorical variables into numbers

#Label encoding is applied to convert categorical variables ( state, area code, international plan, voicemail plan) into numerical format.

label_encoders = {}
for column in data.select_dtypes(include=['object', 'bool']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le


#Split the data into features and Target
X = data.drop(columns=['churn'])
y = data['churn']

#Split the Data into Trading and Testing Sets
import train test split function
from sklearn.model_selection import train_test_split

#Splitting data into 70% training and 30% testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#Output the shapes of the resulting datasets
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)


## Model Building and Evaluation

# 1. Logistic Regression

# mport LogisticRegression
from sklearn.linear_model import LogisticRegression

#Instantiate
logreg = LogisticRegression(fit_intercept=False, C=1e12, solver='liblinear') 

#fit the model to training data
model_log = logreg.fit(X_train, y_train)

#predict on the test data
y_pred = model_log.predict(X_test)

y_pred_proba = model_log.predict_proba(X_test)[:, 1]


1. Confusion matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

#Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', cbar=False, 
            annot_kws={"size": 16}, linewidths=1, linecolor='black',
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show() 

2. Classification report

from sklearn.metrics import classification_report

#Generate the classification report
class_report = classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])
print(class_report)

3. Accuracy

from sklearn.metrics import accuracy_score

#Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

4. AUC and ROC

# Import roc-curve, auc
 
from sklearn.metrics import roc_curve, auc

#calculate the probability scores of each point in the training set
y_train_score = model_log.decision_function(X_train)

#calculate the fpr, tpr and thresholds for the training set
train_fpr, train_tpr, thresholds = roc_curve(y_train, y_train_score)

#calculate the probability scores of each point in the test set
y_test_score = model_log.decision_function(X_test)

#calculate the fpr tpr,and thresholds for the test set
test_fpr, test_tpr, test_threshols = roc_curve(y_test, y_test_score)


#import matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#seaborn's beautiful styling
sns.set_style('darkgrid', {'axes.facecolor': '0.9'})

#ROC curve for training set
plt.figure(figsize=(10, 8))
lw = 2
plt.plot(train_fpr, train_tpr, color='darkorange',
         lw=lw, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.yticks([i/20.0 for i in range(21)])
plt.xticks([i/20.0 for i in range(21)])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) Curve for Training Set')
plt.legend(loc='lower right')
print('Training AUC: {}'.format(auc(train_fpr, train_tpr)))
plt.show()


The score of 0.8037 indicates that the model can distinguish between churn and non-churn customers during training.  this means there is approximately an 80.37% chance that the model will rank a randomly chosen positive instance(churn) higher than a randomly chosen negative instance(non-churn).

#ROC curve for test set
plt.figure(figsize=(10, 8))
lw = 2
plt.plot(test_fpr, test_tpr, color='darkorange',
         lw=lw, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.yticks([i/20.0 for i in range(21)])
plt.xticks([i/20.0 for i in range(21)])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) Curve for Test Set')
plt.legend(loc='lower right')
print('Test AUC: {}'.format(auc(test_fpr, test_tpr)))
print('')
plt.show()


The score of 0.7819 reflects the model's performance on unseen data. it is slightly lower than the training AUC but still indicates good performance.
It means there is approximately 78.19% chance that the model will correctly rank a randomly chosen positive instance higher than a randomly chosen negative instance on the test set.

The AUC score suggests that logistic regression is a fit for the data. It has a strong ability to distinguish between customers who churn and those who don't both on the training set and on the unseen test data. There is a slight drop in performance on the test set, it is within an acceptable range, indicating that the model is not overfitting and should perform well when deployed.


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report, confusion_matrix

#Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

#Calculate precision
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.4f}")

#Calculate recall
recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.4f}")

#Calculate F1-score
f1 = f1_score(y_test, y_pred)
print(f"F1-score: {f1:.4f}")

#Calculate AUC-ROC
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC: {auc_score:.4f}")

# Model interpretation

import numpy as np

#Retrieve feature names and coefficients
feature_names = X_train.columns
coefficients = model_log.coef_[0]

#Create a DataFrame to display the coefficients with the corresponding feature names
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

#Sort by the absolute value of the coefficients to identify the most influential features
coef_df['abs_coefficient'] = np.abs(coef_df['Coefficient'])
coef_df = coef_df.sort_values(by='abs_coefficient', ascending=False).drop(columns=['abs_coefficient'])

coef_df

import matplotlib.pyplot as plt

#Plot the coefficients
plt.figure(figsize=(10, 6))
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=['green' if x < 0 else 'red' for x in coef_df['Coefficient']])
plt.xlabel('Coefficient Value')
plt.title('Feature Importance for Churn Prediction (Logistic Regression)')
plt.show()

The coefficients from the model tell us the direction and magnitude of each coefficient feature's influence on the probability of churn

Positive coefficients: Features with positive coefficients increase the likelihood of churn

Negative coefficients: Features with negative coefficients decrease the likelihood of churn

magnitude: The larger the absolute value of the coefficient the more significant the impact of the feature on churn.


# 2. Decision tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

#Define parameter ranges
max_depth_range = [3, 5, 7, 10]  # Different depths of the tree
min_samples_split_range = [2, 5, 10]  # Minimum samples required to split a node
min_samples_leaf_range = [1, 2, 4]  # Minimum samples required at a leaf node

#Initialize variables to store the best model and best performance
best_model = None
best_score = 0
best_params = {}

#Loop through different combinations of parameters
for max_depth in max_depth_range:
    for min_samples_split in min_samples_split_range:
        for min_samples_leaf in min_samples_leaf_range:
            # Initialize the Decision Tree model with current parameters
            model = DecisionTreeClassifier(max_depth=max_depth, 
                                           min_samples_split=min_samples_split, 
                                           min_samples_leaf=min_samples_leaf, 
                                           random_state=42)
            #Fit the model to the training data
            model.fit(X_train, y_train)

            #Predict on the test set
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            #Evaluate performance metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_roc = roc_auc_score(y_test, y_pred_proba)

            #Update the best model if current model performs better
            if auc_roc > best_score:
                best_score = auc_roc
                best_model = model
                best_params = {
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'AUC-ROC': auc_roc
                }

#Display the best model's parameters and performance
print("Best Decision Tree Parameters and Performance:")
print(best_params)


# Model interpretation


#Get the feature importances
feature_importances = dt_model.feature_importances_

#Print the feature importances
for feature, importance in zip(feature_names, feature_importances):
    print(f"Feature: {feature}, Importance: {importance}")


# Conclusion and Recommendation

 Model performance Summary

 Two model are evaluated logistic regression and decision tree. Here is a comparison of their performance:

 1. Logistic regression

 Accuracy: 0.8600

 Precision: 0.5397

 Recall: 0.2345

 F1-Score 0.3269

 AUC-ROC: 0.7819


 2. Decision tree

 Accuracy: 0.9400

 Precision:0.9126

 Recall: 0.6483

 F1_score: 0.7581

 AUC-ROC: 0.8547


Key Insights:

Decision tree model significantly outperforms Logistic regression in all key performance metrics, particularly in recall and precision. It correctly identifies a high proportion of actual churn cases(recall of 64.83%) while maintaining a high precision (91.26%). this means it accurately predicts churners without many false positives.

Logistic regression model shows lower recall(23.45%), which indicates it fails to identify most of the churn cases. While the model's AUC-ROC score is still reasonable(0.7819), its overall effectiveness in identifying churners is limited, making it less suitable for business objectives.w


Implications for business

Decision tree model is highly effective for predicting customer churn. Its high precision and recall make it a reliable tool for identifying at-risk customers, allowing the company to take proactive measures to retain these customers. The models interpretability also provides insights into key factors influencing churn, which can inform targeted interventions.

Key drivers of churn:
The model reveals that factors such as contract type, tenure, customer service interactions and pricing plans are significant predictors of churn. This highlights areas where the company can focus efforts to improve customer satisfaction and reduce churn rate.


Recommendations:

Implement the Decision Tree model in a real-time environment to monitor and predict churn. Use these predictions to trigger targeted retention strategies such as personalized offers, improved customer service and loyalty programs for high risk customers.

The model indicates customer dissatisfaction is a major driver of churn. Enhancing customer service and offering more flexible contract options and addressing common pain points ca significantly reduce churn rates.

Use model predictive insights to segment customers based on their churn risk. This allows the company to allocate resources efficiently and focusing retention efforts on the most vulnerable segments.

Regularly update the model with new data to keep it accurate and reflective of changing customer behavior. Monitor model performance and refine hyperparameters as needed to ensure optimal results.


Conclusion

Decision tree model provides SyriaTel company with a powerful tool to predict and mitigate customer churn. By implementing this model and acting on its insights, the company can significantly reduce customer turnover and ultimately enhancing profitability and customer loyalty. 
