import pandas as pd


# Function to calculate precision, recall, and F1 score
def calculate_metrics(tp, fp, fn):
    precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
    f1_score = round(2 * (precision * recall) / (precision + recall), 3) if (precision + recall) > 0 else 0.0
    return precision, recall, f1_score


# Read the CSV file
input_file = '../../dataset/PSE/2021_init-v/exploration.csv'  # Replace with your input file name
output_file = '../../dataset/PSE/2021_init-v/exploration.csv'  # Replace with your desired output file name

df = pd.read_csv(input_file)

# Iterate through the DataFrame and calculate the metrics
for index, row in df.iterrows():
    tp = row['Total tp']
    fp = row['Total fp']
    fn = row['Total fn']

    precision, recall, f1_score = calculate_metrics(tp, fp, fn)

    df.at[index, 'Recall'] = recall
    df.at[index, 'Precision'] = precision
    df.at[index, 'F1 score'] = f1_score

# Save the updated DataFrame back to a CSV file
df.to_csv(output_file, index=False)
