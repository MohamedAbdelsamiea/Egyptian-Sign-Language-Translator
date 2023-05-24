# Import necessary libraries
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the hand landmark data and corresponding labels from a pickled data file
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize a Random Forest classifier model and fit it to the training data
rf_model = RandomForestClassifier()
rf_model.fit(train_data, train_labels)

# Use the trained model to predict labels for the testing data and calculate the accuracy of the predictions
predicted_labels = rf_model.predict(test_data)
accuracy = accuracy_score(predicted_labels, test_labels)

# Print the accuracy of the model's predictions
print('{}% of samples were classified correctly!'.format(accuracy * 100))

# Save the trained model to a pickled data file
model_file = open('model.p', 'wb')
pickle.dump({'model': rf_model}, model_file)
model_file.close()
