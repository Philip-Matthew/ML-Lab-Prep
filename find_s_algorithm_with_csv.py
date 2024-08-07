import csv

def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

def find_s_algorithm(training_data):
    # Initialize the most specific hypothesis
    hypothesis = ['?' for _ in range(len(training_data[0]) - 1)]
    
    # Update hypothesis based on positive training examples
    for instance in training_data:
        if instance[-1].strip().lower() == 'yes':  # Check if the instance is positive
            # print(f"Positive example found: {instance[:-1]}")
            for i in range(len(hypothesis)):
                if hypothesis[i] == '?':
                    hypothesis[i] = instance[i]
                elif hypothesis[i] != instance[i]:
                    hypothesis[i] = '?'
            # print(f"Updated hypothesis: {hypothesis}")
    
    return hypothesis

# Main function
# if __name__ == "__main__":
file_path = '../Sample Datasets for project/training_data.csv'
training_data = read_csv(file_path)
# print("Training data:")
# for row in training_data:
#     print(row)
hypothesis = find_s_algorithm(training_data)
print("The most specific hypothesis found by FIND-S algorithm is:", hypothesis)
