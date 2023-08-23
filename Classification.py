import tensorflow as tf
import pandas as pd

tf.get_logger().setLevel('ERROR') # Remove unnecessary warnings

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# Uses keras to grab datasets and put them into a pandas dataframe
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
)
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
)

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

train_y = train.pop('Species') # Pops species from train and puts the species into the train_y. Transfers species from train to train_y
test_y = test.pop('Species')

def input_fn(features, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels)) # Convert features and labels to a dataset object

    if training: # If we are training, we should shuffle our dataset
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size) # Returns the dataset in batches

feature_columns = []
for key in train.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key)) # Creates a numeric feature column based on the keys of the training dataset

# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively
    hidden_units = [30, 10],
    # The model must choose between 3 classes. Three outputs
    n_classes = 3
)

# Train classifier
classifier.train(
    input_fn = lambda: input_fn(train, train_y, training=True), # Creates a callable function. x = lambda: print("Hi"); x() works because you are essentially defining a function in a single line
    steps=5000 # Steps
)

eval_result = classifier.evaluate(
    input_fn = lambda: input_fn(test, test_y, training=False) # Do same but for evaluating
)

print(f"{eval_result}")

# Predictor. Predicts based on values
def input_fn(features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size) # Returns a dataset object from the features provided. Does not provide a y value because we don't know it, and that is what we are trying to predict.

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted")
# Retrieves a value for each prompt. Breaks out of loop if number is provided, since we are looking for length
for feature in features:
    valid = True
    while valid:
        val = input(feature + ": ")
        if not val.isdigit(): valid = False

    predict[feature] = [float(val)] # Must be in a list, because it's expecting that we would have more than 1 value to predict for

predictions = classifier.predict(input_fn=lambda: input_fn(predict)) # Predicts using the input function (lambda as before)
for pred_dict in predictions: # Because predictions will be a list we must account for that
    class_id = pred_dict['class_ids'][0] # Gets class ids
    probability = pred_dict['probabilities'][class_id]

    print(f"Prediction is: {SPECIES[class_id]} {(100*probability)}")