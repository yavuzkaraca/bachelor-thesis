import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the CSV file
file_path = "../../out/advanced/advanced_score.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Extract the relevant columns
extracted_df = df[['Id', 'Total Recall', 'Total Precision', 'Total F1 Score']]

# Set the Id column as the index
extracted_df.set_index('Id', inplace=True)

# Plot the data with specific colors
plt.figure(figsize=(12, 6))
ax = extracted_df.plot(kind='bar', figsize=(12, 6), width=0.8, color=['#FFD700', '#FFA500', '#FF4500'])  # Yellow, Orange, Red
plt.title('Total Recall, Precision, and F1 Score by ID')
plt.ylabel('Scores')
plt.xticks(rotation=45, ha='right')
plt.legend(loc='best')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()

# Rename the columns
df = df.rename(columns={
    'L1 tp': 'Level 1 tp',
    'L1 fp': 'Level 1 fp',
    'L1 fn': 'Level 1 fn',
    'L1 Recall': 'Level 1 Recall',
    'L1 Precision': 'Level 1 Precision',
    'L1 F1 Score': 'Level 1 F1 Score',
    'L2 tp': 'Level 2 tp',
    'L2 fp': 'Level 2 fp',
    'L2 fn': 'Level 2 fn',
    'L2 Recall': 'Level 2 Recall',
    'L2 Precision': 'Level 2 Precision',
    'L2 F1 Score': 'Level 2 F1 Score',
    'L3 tp': 'Level 3 tp',
    'L3 fp': 'Level 3 fp',
    'L3 fn': 'Level 3 fn',
    'L3 Recall': 'Level 3 Recall',
    'L3 Precision': 'Level 3 Precision',
    'L3 F1 Score': 'Level 3 F1 Score'
})

# Exclude the 'Id' column and any column that contains 'tp', 'fp', or 'fn'
df_filtered = df.drop(columns=['Id']).filter(regex='^(?!.*(tp|fp|fn)).*$')

# Convert all other columns to numeric, forcing non-numeric to NaN, then fill NaN with 0
df_filtered = df_filtered.apply(pd.to_numeric, errors='coerce').fillna(0)

# Sum all the rows for each numeric column
column_sums = df_filtered.sum()
column_average = column_sums.div(8)

# Create the bar plot
plt.figure(figsize=(12, 6))
bar_plot = column_average.plot(kind='bar', color='plum', edgecolor='black')
# Set y-axis to logarithmic scale
plt.yscale('log')

# Add titles and labels
plt.title('Average of the Dataset for each Level of Incompleteness')
plt.xlabel('Columns')
plt.ylabel('Average (Log Scale)')

# Add gridlines for better readability
plt.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust layout to prevent clipping
plt.tight_layout()

# Show the plot
plt.show()