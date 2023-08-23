import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tensorflow import feature_column as fc
import tensorflow as tf
import tqdm
import wandb
wandb.init(project='Test', config={})
tf.get_logger().setLevel('ERROR') # Remove unnecessary warnings


dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # Training dataset
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # Testing dataset
y_train = dftrain.pop('survived') # Just retain the survived column
y_eval = dfeval.pop('survived') # Same
# print(y_train) # Print training data
# print(dftrain.loc[0], y_train.loc[0])
# print(dftrain.describe()) # Print info on the table
# print(dftrain.shape)

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone'] # All categorical columns
NUMERIC_COLUMNS = ['age', 'fare'] # All numeric columns

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique() # Gets all of the unique instances and puts into a list. Sex would be ['female', 'male']
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)) # Appends the info about the vocabulary onto the feature column (which is the entire training dataset). Describes the key ('sex'), vocab list ('male', 'female'), the dtype and default value.

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32)) # Appends numeric data to feature

#print(feature_columns)

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=128): # data_df: Pandas dataframe, label_df: y_train or y_eval, epochs: Num, shuffle: Are we going to shuffle data, batch_size: How much data given to model at once
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df)) # Creates a tf.data.Dataset object with data and label inside of the object.
        if shuffle:
            ds = ds.shuffle(1000) # Shuffles data
        ds = ds.batch(batch_size).repeat(num_epochs) # Splits dataset into batches of 32 and does this for the number of epochs. So it will have a batch for each epoch.
        return ds # Returns a batch of the dataset
    return input_function

train_input_fn = make_input_fn(dftrain, y_train) # Calls input function to get a dataset which can be used to train a model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False) # Creates a dataset for evaluation. Not training so 1 epoch no shuffle.

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns) # Creates an estimator with the feature columns defined earlier. Parses the "legend" of the data into the train object.

linear_est.train(train_input_fn, hooks=[wandb.tensorflow.WandbHook(steps_per_log=1)]) # Trains the model based on the given dataset.
result = linear_est.evaluate(eval_input_fn) # Evaluates the model based on the given dataset. Stores the results, since this is what we are evaluating.

clear_output() # Clears console
print(result['accuracy']) # Prints the accuracy based on the eval data
print(result)

# result = list(linear_est.predict(eval_input_fn))
# print(dfeval.loc[0])
# print(y_eval.loc[0])
# print(result[0]['probabilities'][1]) # Prints probably of survival based on x set in eval