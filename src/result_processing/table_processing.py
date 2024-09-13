import sys

import pandas as pd


# Function to calculate precision, recall, and F1 score
def calculate_metrics(tp, fp, fn):
    precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
    f1_score = round(2 * (precision * recall) / (precision + recall), 3) if (precision + recall) > 0 else 0.0
    return precision, recall, f1_score


# Read the CSV file
file_path = '../../out/delimiter_difference/comma_delimiter_score.csv'  # Replace with your input file name

df = pd.read_csv(file_path)


for index, row in df.iterrows():
    tp1 = row['L1 tp']
    tp2 = row['L2 tp']
    tp3 = row['L3 tp']

    total_tp = tp1 + tp2 + tp3

    df.at[index, 'Total tp'] = total_tp

    # ===============================================

    fp1 = row['L1 fp']
    fp2 = row['L2 fp']
    fp3 = row['L3 fp']

    total_fp = fp1 + fp2 + fp3

    df.at[index, 'Total fp'] = total_fp

    # ===============================================

    fn1 = row['L1 fn']
    fn2 = row['L2 fn']
    fn3 = row['L3 fn']

    total_fn = fn1 + fn2 + fn3

    df.at[index, 'Total fn'] = total_fn


for index, row in df.iterrows():
    tp = row['L1 tp']
    fp = row['L1 fp']
    fn = row['L1 fn']

    precision, recall, f1_score = calculate_metrics(tp, fp, fn)

    df.at[index, 'L1 Recall'] = recall
    df.at[index, 'L1 Precision'] = precision
    df.at[index, 'L1 F1 Score'] = f1_score

    #===============================================

    tp = row['L2 tp']
    fp = row['L2 fp']
    fn = row['L2 fn']

    precision, recall, f1_score = calculate_metrics(tp, fp, fn)

    df.at[index, 'L2 Recall'] = recall
    df.at[index, 'L2 Precision'] = precision
    df.at[index, 'L2 F1 Score'] = f1_score

    # ===============================================

    tp = row['L3 tp']
    fp = row['L3 fp']
    fn = row['L3 fn']

    precision, recall, f1_score = calculate_metrics(tp, fp, fn)

    df.at[index, 'L3 Recall'] = recall
    df.at[index, 'L3 Precision'] = precision
    df.at[index, 'L3 F1 Score'] = f1_score

    # ===============================================

    tp = row['Artificial tp']
    fp = row['Artificial fp']
    fn = row['Artificial fn']

    precision, recall, f1_score = calculate_metrics(tp, fp, fn)

    df.at[index, 'Artificial Recall'] = recall
    df.at[index, 'Artificial Precision'] = precision
    df.at[index, 'Artificial F1 Score'] = f1_score

    # ===============================================

    tp = row['Total tp']
    fp = row['Total fp']
    fn = row['Total fn']

    precision, recall, f1_score = calculate_metrics(tp, fp, fn)

    df.at[index, 'Total Recall'] = recall
    df.at[index, 'Total Precision'] = precision
    df.at[index, 'Total F1 Score'] = f1_score


# Save the updated DataFrame back to a CSV file
df.to_csv(file_path, index=False)

if __name__ == '__main__':
    main()
    sys.argv
