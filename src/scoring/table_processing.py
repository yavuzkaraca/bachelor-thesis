import pandas as pd


# Function to calculate precision, recall, and F1 score
def calculate_metrics(tp, fp, fn):
    precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
    f1_score = round(2 * (precision * recall) / (precision + recall), 3) if (precision + recall) > 0 else 0.0
    return precision, recall, f1_score


# Read the CSV file
file_path = '../../out/advanced/advanced_score.csv'  # Replace with your input file name

df = pd.read_csv(file_path)

"""
for index, row in df.iterrows():
    tp1 = row['L1 tp']
    tp2 = row['L2 tp']
    tp3 = row['L3 tp']

    total_tp = tp1 + tp2 + tp3

    df.at[index, 'Total tp'] = total_tp
"""


for index, row in df.iterrows():
    tp = row['Total tp']
    fp = row['Total fp']
    fn = row['Total fn']

    precision, recall, f1_score = calculate_metrics(tp, fp, fn)

    df.at[index, 'Total Recall'] = recall
    df.at[index, 'Total Precision'] = precision
    df.at[index, 'Total F1 score'] = f1_score


# Save the updated DataFrame back to a CSV file
df.to_csv(file_path, index=False)
