# credit-risk-default-prediction
Predicting borrower default risk using Logistic Regression and Random Forest on a 150K-row credit dataset
Overview

This project builds and compares machine learning models to predict borrower default risk using a real-world, 150,000-row credit history dataset. The goal is to identify high-risk borrowers and support data-driven lending decisions.

Problem

Financial institutions face significant risk when lending to borrowers. Accurately predicting default probability helps reduce losses and improve credit decision-making.

The real challenge in this dataset: only about 7% of borrowers actually default. This kind of class imbalance is common in risk modeling (rare disease detection, insurance claims, fraud detection) and makes plain accuracy a misleading metric, a model that predicts "no default" for everyone would still be 93% accurate while catching zero real defaulters.

Approach: 

Cleaned and preprocessed the dataset, handling missing values in MonthlyIncome and NumberOfDependents

Clipped extreme outliers in MonthlyIncome, DebtRatio, and RevolvingUtilizationOfUnsecuredLines at the 99th percentile

Explored which variables differed most between defaulters and non-defaulters (age, income, credit utilization, payment history)

Split the data with stratified sampling to preserve class balance in train/test sets

Built and compared three models: 
Baseline Logistic Regression (no adjustment for class imbalance)
Rebalanced Logistic Regression (using class_weight='balanced')
Random Forest (also balanced)

Results

Model	Accuracy	Recall (Default Class)

Baseline Logistic Regression	93.5%	5.6%

Rebalanced Logistic Regression	76.1%	75%

Random Forest (balanced)	93.6%	15%

Why I Chose the Rebalanced Logistic Regression

In a real lending context, missing an actual defaulter (a false negative) is far more costly than incorrectly flagging a safe borrower (a false positive) for review. The baseline model and Random Forest both have high accuracy, but they miss the vast majority of real defaulters, that's not useful for a lender trying to manage risk.

The rebalanced Logistic Regression sacrifices overall accuracy but catches 75% of actual defaulters, compared to just 6% (baseline) or 15% (Random Forest). Late payment history and credit utilization were the strongest signals separating defaulters from non-defaulters throughout the analysis.

Tech Stack

Python (pandas, NumPy, scikit-learn), seaborn, matplotlib, Jupyter Notebook

Impact

This project demonstrates how thoughtful model selection, prioritizing the right metric for the business problem, not just the highest accuracy can meaningfully improve credit risk assessment and support more accurate, responsible lending decisions.
