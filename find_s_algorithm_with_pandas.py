import pandas as pd

def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data

def find_s_algorithm(training_data):
    # Get the number of attributes
    num_attributes = len(training_data.columns) - 1

    # Initialize the most specific hypothesis
    hypothesis = ['0'] * num_attributes

    # Iterate through each example in the training data
    for i in range(len(training_data)):
        # Check if the example is a positive example
        if training_data.iloc[i, -1].strip().lower() == 'yes':
            # For the first positive example, initialize the hypothesis
            if hypothesis == ['0'] * num_attributes:
                hypothesis = training_data.iloc[i, :-1].tolist()
            else:
                # Update the hypothesis for each attribute
                for j in range(num_attributes):
                    if hypothesis[j] != training_data.iloc[i, j]:
                        hypothesis[j] = '?'

    return hypothesis

# Read the training data from CSV
file_path = '../Sample Datasets for project/training_data.csv'
training_data = read_csv(file_path)

# Apply the FIND-S algorithm
most_specific_hypothesis = find_s_algorithm(training_data)

print("The most specific hypothesis is:", most_specific_hypothesis)
