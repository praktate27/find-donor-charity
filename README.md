# find-donor-charity
This project will employ several Supervised Algorithms to accurately model individuals' income using data collected from the 1994 U.S. The Aim of project is to construct a model that accurately predicts whether an individual makes more than $50,000. This sort of task can arise in a non-profit setting, where organizations survive on donations. Understanding an individual's income can help a non-profit better understand how large of a donation to request, or whether or not they should reach out to begin with.
The dataset for this project originates from the UCI Machine Learning Repository. The datset was donated by Ron Kohavi and Barry Becker.

# Steps:
1. Preprocessing of data is done to check if data needs cleaning, formatting and restructuring. In dataset, there are no invalid or empty entries .

2. Apply logarithmic transformation to reduce the range of values caused by outliers. Then normalize numerical features. Here we will be using 'capital-gain' and 'capital-loss' for transformation and normalization.

3. Data has some non-numeric fields and learning algorithm requires numeric data. Hence we need to convert non-numeric data to numberic by one-hot encoding scheme. As we will be referring to 'income' which has values either '<=50K' or '>50K', we will not be doing one-hot endcoding scheme rather we will be converting it to '<=50K' to '0' and '>50K' to '1'.

4. Now we will be splitting data 80% for training sets and 20% of it for testing sets.

5. Supervised Learning Models
    The following are some of the supervised learning models that are currently available in scikit-learn that you may choose     from:

      Gaussian Naive Bayes (GaussianNB)
      Decision Trees
      Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
      K-Nearest Neighbors (KNeighbors)
      Stochastic Gradient Descent Classifier (SGDC)
      Support Vector Machines (SVM)
      Logistic Regression
We will be implementing  following 3 models and will check which one is more efficient
1) AdaBoost Classifier
2) Support Vector Machines(SVM)
3) Decision Tree Classifier

6. To properly evaluate performance of each model we will be doing following steps :
  --> Import fbeta_score and accuracy_score from sklearn.metrics.
  --> Fit the learner to the sampled training data and record the training time.
  --> Perform predictions on the test data X_test, and also on the first 300 training points X_train[:300].
        --> Record the total prediction time.
  --> Calculate the accuracy score for both the training subset and testing set.
  --> Calculate the F-score for both the training subset and testing set.
  
7. Following things to be done to evaluate model
  --> Import the three supervised learning models you've discussed in the previous section.
  --> Initialize the three models and store them in 'clf_A', 'clf_B', and 'clf_C'.
  --> Use a 'random_state' for each model you use, if provided.
  --> Calculate the number of records equal to 1%, 10%, and 100% of the training data.
  --> Store those values in 'samples_1', 'samples_10', and 'samples_100' respectively.
  
# From performance metrics
we can predict that 'AdaBoost Classifier' is the most appropriate model out of the three for the task of identifying individuals that make more than $50,000. AdaBoost Classifier out performs both Decission Tree Classifier and LinearSVC since F-score on testing when 100 percent of training data is used is highest when compared to other two models. Model Predicting time and trainig time is higghest under all cases(Less is better). With high accuracy and F-score, AdaBoost Classifier is the most approriate supervised learning model for this problem.

8. Final Model Tuning. Following steps to take:
  -->  Import sklearn.grid_search.GridSearchCV and sklearn.metrics.make_scorer.
  --> Initialize the classifier you've chosen and store it in clf.
  --> Set a random_state if one is available to the same state you set before.
  --> Create a dictionary of parameters you wish to tune for the chosen model.
  --> Use make_scorer to create an fbeta_score scoring object (with [Math Processing Error]).
  --> Perform grid search on the classifier clf using the 'scorer', and store it in grid_obj.
  --> Fit the grid search object to the training data (X_train, y_train), and store it in grid_fit.
# Output of the above steps :
Unoptimized model
------
Accuracy score on testing data: 0.8576
F-score on testing data: 0.7246

Optimized Model
------
Final accuracy score on the testing data: 0.8606
Final F-score on the testing data: 0.7316

# Result:
The optimized model's accuracy and F-score on testing data is 0.8606 and 0.7316 respectively. These scores are slightly better than the unoptimized model. There is a huge improvement from the naive predictor benchmarks (Accuracy: 0.2478 F-score: 0.2917), the optimized model gives almost 86 percent accurate results.

# Feature importance for prediction:
1) Age 
2) hours-per-week 
3) education_level 
4) occupation 
5) capital-gain 
Age being the most important of all, can be a crucial feature because children and senior citizens will most likely not be earning more than $50,000. Hours per week is a likely indicator of income for the same reason, as part-time or unemployed people will almost always have below 50K income. Education level can also help to differentiate since people who are more educated are more likely to earn more than 50K. Occupation assists in further sorting the working-age population, and gives the ability to identify "high-paying" and "low-paying" occupations across multiple job-sectors. Finally capital gain, people with more capital gain are more likely to earn more than 50K and vice-versa.

If we implement our model with with these 5 top features, training and prediction time is much lower.

Final Model trained on full data
------
Accuracy on testing data: 0.8606
F-score on testing data: 0.7316

Final Model trained on reduced data
------
Accuracy on testing data: 0.8325
F-score on testing data: 0.6752

The reduced model performes worse than the optimized, full-data model. The thing that is worth noting is that the unoptimized model with default parameters also outperformed the reduced data configuration. Because of this, I prefer not to use the reduced data version. Even if we consider training time unoptimized model was relatively fast, training and making predictions on the full dataset was under 1 second.
