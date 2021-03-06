
# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree its the first project in this nano degree program.

In this project, we manufacture and advance an Azure ML pipeline utilizing the Python SDK and a gave Scikit-learn model. 

The hyper-boundaries of the Scikit-learn model will be tuned utilizing Azure HyperDrive functionality.This model is then contrasted with an Azure AutoML run and then we compare performance of both of the methods using different metrics. 

## Summary

- In this problem the dataset contains data about the financial and personal details of the customers of a Portugese bank. We seek to predict if the customer will subscribe to bank term deposit or not. <br>
- First , a Scikit-learn based LogisticRegression model is trained using cleaned data set and then we tune hyperparameter of it using azure hyper drive  <br>
-At that point, the equivalent dataset is given to Azure AutoML to attempt to locate the best model utilizing its usefulness. <br>
- We came to know that Soft Voting Ensemble was the best performing model out of all the models found using AutoML.

## Scikit-learn Pipeline

### Pipeline Architecture
- In the Pipeline, first the dataset is recovered from the given url utilizing AzureDataFactory class in the train.py file. <br>
- Then we create a compute instance for our model to train on it.
- At that point the information is cleaned utilizing clean_data function in which some preprocessing steps were performed like changing straight out factor over to two fold encoding, one hot encoding, etc and afterward the dataset is part in proportion of 70:30 (train/test) for training and testing and sklearn's LogisticRegression Class is utilized to characterize Logistic Regression model. <br>
- The train.py content contains all the means expected to prepare and test the model which are information recovery, information cleaning and pre-handling, information parting into train and test information, characterizing the scikit-learn model and preparing the model on train information and foreseeing it on the test information to get the precision and afterward sparing the model. <br>
- A SKLearn estimator is created in which we pass it train.py script and the compute on which training of model should occur. 
- Then we create HyperDriveConfig by passing estimator, policy, hyperparameter sampling and primary metric name on which our model will be measured
- The hyperparameters which are should have been tuned are characterized in the boundary sampler. The hyperparameters that can be tuned here are C and max_iter. C is the converse regularization boundary and max_iter is the greatest number of emphasess. <br>
- Finally, the best model is saved using joblib <br>

### Benefits of parameter sampler
- The parameter sampler is utilized to give various decisions of hyperparameters to look over and have a go at during hyperparameter tuning utilizing hyperdrive. <br>
- I have utilized Random Parameter Sampling in the parameter sampler with the goal that it work very well and may be utilized to give irregular examining over a hyperparameter search space.
- For our problem, the hyperparameters gave in the hyperparamete search space are C and max_iter.The various decisions for the estimations of C and max_iter are given so that the hyperdrive can attempt all the blends of decisions to do the hyperparameter tuning to get the best model with the greatest exactness.

### Benefits of Early Stopping policy
- One can characterize an Early Stopping strategy in HyperDriveConfig and it is valuable in halting the HyperDrive run if the precision of the model isn't improving from the best exactness by a specific characterized sum after each given number of emphasess <br>
- In this model, we have characterized a Bandit Policy for early stopping with the boundaries slack_factor and evaluation_interval which are characterized as :
  - slack_factor :  The measure of slack permitted as for the best performing preparing run. This factor determines the leeway as a proportion. <br>
  - evaluation_interval : The recurrence for applying the policy. Each time the preparation content logs the essential measurement considers one span.<br>
- Early stop save a lot of compute resources by stopping the model on right time.

## AutoML
- AutoML implies Automated ML which implies it can automate all the cycle associated with a Machine Learning measure. For instance, we can automate feature engineering, hyperparameter determination, model preparing, and tuning and can prepare and send 100 models in a day all with the assistance of AutoML.
- At the point when I applied AutoML to our concern, it did an extraordinary undertaking and I was astonished to see that AutoML attempted so a wide range of models in quite a brief timeframe some of which I was unable to try and consider attempting or executing. The models attempted via AutoML were RandomForests,BoostedTrees,XGBoost,LightGBM,SGDClassifier,VotingEnsemble, and so on AutoML utilized various information preprocessing standardization like Standard Scaling, Min Max Scaling, Sparse Normalizer, MaxAbsScaler, and so on It has likewise dealt with class irregularity very well without anyone else. <br>
- To run AutoML, one necessities to utilize AutoMLConfig class simply like HyperdriveConfig class and need to characterize an automl_config item and setting different boundaries in it which are expected to run the AutoML. A portion of these boundaries are: <br>
    - task : Type of task needed to perform (regression or classification) <br>
    - training_data : Data to train the autoML model. <br>
    - label_column_name : output label from data the model will yield. <br>
    - iterations : The number of iterations the autoML should run. <br>
    - primary_metric : the evaluation metric for the models for example we used accuracy<br>
    - n_cross_validations :  Cross validations needed to perform in each model <br>
    - experiment_timeout_minutes : The wait time after which autoML will stop. <br>
- List of all the models used in the process:

![alt_text](7.PNG)

## Pipeline comparison

- Overall,the contrast in exactness between the AutoML model and the Hyperdrive tuned custom model isn't excessively. AutoML exactness was 0.9163 while the Hyperdrive precision was 0.9096

- With Respect to engineering AutoML was in a way that is better than hyperdrive in light of the fact that it attempted many models, which was very difficult to do with Hyperdrive on the grounds that for that we need to make pipeline for each model.

- There was very little distinction in exactness perhaps on account of the informational collection yet AutoML truly attempted and figured some intricate models to get the best outcome and model out of the given dataset.

The best run/model of HyperDrive : 

![alt_text](4.PNG)

The best run/model in AutoML :

![alt_text](6.PNG)

Some of the top features in our dataset as learnt by the best AutoML model are :

![alt_text](10.PNG)

## Future work

- For the future I would like to use Bayesian Sampler instead of Random Sampler.
- I would like to use different primary metric to test the performance of the model. I would like to use AUC instead of accuracy as primary metric in the future.
- I have taken a stab at running AutoML with both clean and preprocessed dataset and furthermore raw and uncleaned dataset to see whether it can do the cleaning and pre-preparing without anyone else and it gave great outcomes in the two of them so I dont know how AutoML took care of it itself and it spared my difficulty of information cleaning. So I need to know whether it can do this information cleaning for a wide range of ML issues or not.

## Proof of cluster clean up

- Here is the snapshot of deleting the compute cluster i took when the cluster was getting deleted

![alt text](11.PNG)


```python

```
