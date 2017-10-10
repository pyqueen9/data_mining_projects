## Naive Bayes in Python on the "adult-big" dataset

Source for data: https://archive.ics.uci.edu/ml/datasets/adult

Prepare the dataset:

#1) Parse/clean up the dataset input file and store data

Data preprocessing steps:

#1) Calculate numeric and nominal attributes means
#2) Replace missing values in the data set (with mean for numeric and mode for nominal attributes)
#3) Entropy-based discretization: 
-	  calculate entropy and information gain to find best splitting values for nominal/ categorical attributes 
#4) Update the nominal attribute values with discretized values
 
Training and testing Naive Bayes classifier steps:

#1) Perform K fold cross validation to divide data into Train Set and Test Sets
#2) Training
-	  calculate Prior probabilities
-	  calculate Likelihoods

#3) NB classification steps:  
-	  calculate Posterior probabilities for each test sample and determine predictions of the class labels
-	  store and output evaluation metrics: accuracy, micro precision, micro recall, macro precision, macro recall, F1)

Parameters to tune: 
    k = number of folds for cross validation
    pos = position of categorical attribute to discretize
