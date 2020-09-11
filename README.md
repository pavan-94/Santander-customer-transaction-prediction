# Santander-customer-transaction-prediction
Introduction:

This project is intended to review the work carried out by different kagglers on a dataset and identify the underlying gaps to be filled. The dataset chosen for this project is called Santander Customer Prediction (https://www.kaggle.com/lakshmi25npathi/santander-customer-transaction-prediction-dataset). This dataset set was released for a competition in Kaggle where the aim is to predict if a customer will make a transaction or not. It has two csv files one is training set and other is testing set. There is a total of 202 columns and 2,00,000 rows in the training set including target and index column, whereas in test set there are 201 columns and 2,00,000 rows and it doesn’t  have the target column. This is a binary classification problem where the class labels to be predicted is either ‘0’(customer does not make transaction) or ‘1’(customer makes transaction). All the columns are numeric and there are no categorical variables. 

Gaps and Plan:

From the review of existing work, it was identified that the boosting algorithms like LightGBM and XGBoost show better performance than other models for binary classification with imbalanced data. There do exist other boosting algorithm like CatBoost which is much faster than its counterparts for bigger datasets and none of the kernels have showed implementation using this model. Also, for such boosting algorithms it is essential to select a proper set of hyperparameters for getting better performance. Though one of the kernels did use grid search on hyperparameters to tune the model but not all the important parameters like max_depth, learning rate were identified and grid search for finding optimal parameters is very time consuming and requires more computational power. To overcome the complexity of the Grid search random search of hyperparameters can be used which randomly subsets a set of parameter values and it is faster compared to grid search. The results showed that models performed best for imbalanced dataset, but for any machine learning model it is essential to balance the dataset. Hence to get an optimal prediction of the target following is the plan which is implemented:

•	The data is checked for skewness and kurtosis, based on the result the data is standardized.

•	Target attribute is balanced using ROSE.

•	In total 8 different Xgboost and CatBoost models are built 

•	Two variants of Xgboost and Two Variants of CatBoost models are built one each for balanced and unbalanced dataset using grid search for hyper parameter tuning.

•	Two variants of Xgboost and Two Variants of CatBoost models are built one each for balanced 

and unbalanced dataset using random search for hyper parameter tuning.

•	The models are build using both balanced and unbalanced dataset to monitor how the model performs in both the cases. The metric used for evaluation is AUC. 

•	Further to improve the model performance 3-fold cross validation is done

References 
N, L. (2020). Santander Customer Transaction Prediction Dataset. Retrieved 13 May 2020, from https://www.kaggle.com/lakshmi25npathi/santander-customer-transaction-prediction-dataset

Prabhakaran, S. (2020). Caret Package – A Practical Guide to Machine Learning in R – Machine Learning Plus. Retrieved 13 May 2020, from https://www.machinelearningplus.com/machine-learning/caret-package/ 
How to produce a confusion matrix and find the misclassification rate of the Naïve Bayes Classifier?. (2020). Retrieved 13 May 2020, from https://stackoverflow.com/questions/46063234/how-to-produce-a-confusion-matrix-and-find-the-misclassification-rate-of-the-na%C3%AF 

Usage examples - CatBoost. Documentation. (2020). Retrieved 13 May 2020, from https://catboost.ai/docs/concepts/r-usages-examples.html#selecting-hyperparameters 

Berhane, F. (2020). Extreme Gradient Boosting with R. Retrieved 13 May 2020, from https://datascienceplus.com/extreme-gradient-boosting-with-r/

Beginners Tutorial on XGBoost and Parameter Tuning in R Tutorials & Notes | Machine Learning | HackerEarth. (2020). Retrieved 13 May 2020, from https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/

Menardi, G., & Torelli, N. (2012). Training and assessing classification rules with imbalanced data. Data Mining And Knowledge Discovery, 28(1), 92-122. doi: 10.1007/s10618-012-0295-5


