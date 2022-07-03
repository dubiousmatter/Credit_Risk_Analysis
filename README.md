# Credit_Risk_Analysis

## Overview

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, we needed to employ different techniques to train and evaluate models with unbalanced classes. In this analysis we used imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, we oversampled the data using the RandomOverSampler and SMOTE algorithms, and undersampled the data using the ClusterCentroids algorithm. Then, we used a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, we compared two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk.

## Results

![Random Forest](https://user-images.githubusercontent.com/101157423/177043348-8292e207-094b-4305-bdd2-67f02ba7362e.png)

As seen in the summary above, the random forest classifier scores without AdaBoost were:
- Precision - 0.03
- Recall - 0.70
- F1 - 0.06

![Random Forest AdaBoost](https://user-images.githubusercontent.com/101157423/177043712-7f038a26-81d8-4436-b911-b124e5e32d7c.png)

As seen in the above summary, the random forest classifier scores with AdaBoost were:
- Precision - 0.09
- Recall - 0.92
- F1 - 0.16

![Naive Random](https://user-images.githubusercontent.com/101157423/177044054-5adbfbe4-22ba-4efc-8222-bbda2c9f4b9f.png)

As seen in the above summary, the naive random oversampling scores were:
- Precision - 0.01
- Recall - 0.74
- F1 - 0.02

![Smote](https://user-images.githubusercontent.com/101157423/177044157-b6553a6a-b2d3-496b-9338-13f90cd2dcb5.png)

As seen in the above summary, the SMOTE oversampling scores were:
- Precision - 0.01
- Recall - 0.63
- F1 - 0.02

![Undersampling](https://user-images.githubusercontent.com/101157423/177044338-67d3bc2e-80bf-409f-99a7-59b7a1f2a4b6.png)

As seen in the above summary, the undersampling scores were:
- Precision - 0.01
- Recall - 0.67
- F1 - 0.01

![Combination](https://user-images.githubusercontent.com/101157423/177044538-ce1875b1-7ba2-401c-bdb3-1172050b5d1a.png)

As seen in the above summary, the combination sampling scores were:
- Precision - 0.01
- Recall - 0.70
- F1 - 0.02

## Summary

The random forest classifier, with and without AdaBoost, failed to achieve useable performance. The balanced random forest classifier's precision is 0.03, meaning that in 100 loan applications that were flagged to be bad, only 3 were actually bad loan applications. The model's recall is 0.70, meaning that it detected 70% of bad loan applications. The F1 score is low at 0.06, since either a low precision or recall will result in a lower F1 score.

The random forest classifier with AdaBoost, while achieving better results, still suffered from inadequate predictive power. Its precision score is 0.09 and its recall 0.92. The F1 score, again, is skewed low at 0.16 by the low precision score.

The performances of both models are insufficient for commercial application.

Oversampling, both naive and with SMOTE, did not yield a useable model for the prediction of bad loans. Naive oversampling resulted in a precision score of 0.01 and recall of 0.74. The F1 score of 0.02 reflected the low precision score. SMOTE oversampling similarly resulted in a precision score of 0.01 and recall of 0.63, and a F1 score of 0.02.

A model with undersampling yielded similarly poor results, with a precision score of 0.01, recall of 0.67, and F1 score of 0.01.

Combination sampling resulted in a precision score of 0.01 and recall of 0.70, and a F1 score of 0.02. While the recall score is marginally better than that of the other models, the overall performance of the model is still poor.

These models are inadequately predictive, and are not recommended for commercial application.
