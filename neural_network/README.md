
## Neural Network classifier for the "adult-big" dataset in Python

Parsing of the file and preprocessing methods are implemented for the "adult-big" dataset including:
replacing missing values by mean/mode of class, 
discretization of the age attribute, 
and Z-score normalization for numerical data.
The program also implements neural network training and testing.
Parameters learning rate, # of hidden layer nodes, and # of epochs can be changed to try different models.
The output of the script is the first model in the results table for k=5 folds CV.

Command to run file: python neural_net.py -f adult-big.arff
