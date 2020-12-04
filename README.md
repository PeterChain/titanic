# titanic
Script for the Titanic dataset for evaluating which passengers survived


## Dependencies

Requires the following python3 libraries

- Pandas
- Scikit Learn


## How to run

Just run the python script, the script needs 2 .csv files in directory, a train and a test dataset of the titanic dataset types.
The script will output several model evaluations and create a results.csv, in Kaggle's output format.


## What does it do

Fix some missing values from both train and test sets, normalize a few values (scale values to stay within a 0 and 1 range) and categorize others (Age into integer).
We run several CART models (AdaBoost and GradientBoost) with several estimators to try to find the best match.
