# LoanApprovalPrediction Using Machine Learning Models ( LR,KNN,SVC,RF-DT Voting)
---
## Problem Statement:
* To develop machine learning models to make accurate prediction of whether to approave loan for an application or considering the data being provided.

## Datasets Used:
* Kaggle Datasets for Loan Approaval Prediction

## Setup Required:

* Install the following library using the command promt:
    * ```bash
         pip install numpy pandas seaborn matpltlib scikit-learn
      ``` 
## Results Achieved:
    ### Model Performance Table

| Model                     | Accuracy | Precision | Recall   | F1-Score | ROC-AUC Score | Cross-Validation Score |
|---------------------------|----------|-----------|----------|----------|----------------|-------------------------|
| Decision Tree             | 0.978469 | 0.988350  | 0.976967 | 0.982625 | 0.978960       | 0.976959                |
| Random Forest             | 0.978469 | 0.990253  | 0.975048 | 0.982592 | 0.998708       | 0.976958                |
| Soft-Voting Classifier    | 0.970096 | 0.988189  | 0.963532 | 0.975705 | 0.995393       | 0.973667                |
| Hard-Voting Classifier    | 0.950957 | 0.974308  | 0.946257 | 0.960078 | 0.995393       | 0.959303                |
| SVC                       | 0.941388 | 0.959144  | 0.946257 | 0.952657 | 0.983036       | 0.939853                |
| KNN                       | 0.918660 | 0.952096  | 0.915547 | 0.933464 | 0.971892       | 0.919805                |
| Logistic Regression       | 0.912679 | 0.929119  | 0.930902 | 0.930010 | 0.967736       | 0.918907                |


