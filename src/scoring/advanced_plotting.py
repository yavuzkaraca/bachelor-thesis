import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the CSV file
file_path = "../../out/advanced/advanced_score.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Rename the columns
columns_rename = {
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
}

"""
================================ PLOT ONE ================================
"""

# Extract the relevant columns
extracted_df = df[['Id', 'Total Recall', 'Total Precision', 'Total F1 Score']]

# Set the Id column as the index
extracted_df.set_index('Id', inplace=True)

# Plot the data with specific colors
ax = extracted_df.plot(kind='bar', figsize=(8, 6), width=0.4, color=['#333333', '#B0C4DE', 'teal'], edgecolor='black')  # Yellow, Orange, Red
plt.title('Total Recall, Precision, and F1 Score by ID')
plt.ylabel('Scores')
plt.xticks(rotation=45, ha='right')
plt.legend(loc='best')
plt.grid(axis='y', linestyle='--', linewidth=0.3, alpha=0.5)
plt.tight_layout()

# Show the plot
plt.show()

"""
================================ PLOT TWO ================================
"""

df = df.rename(columns=columns_rename)

# Exclude the 'Id' column and any column that contains 'tp', 'fp', or 'fn'
df_filtered = df.drop(columns=['Id']).filter(regex='^(?!.*(tp|fp|fn)).*$')

# Convert all other columns to numeric, forcing non-numeric to NaN, then fill NaN with 0
df_filtered = df_filtered.apply(pd.to_numeric, errors='coerce').fillna(0)

# Sum all the rows for each numeric column
column_sums = df_filtered.sum()
column_average = column_sums.div(len(df))

# Create the bar plot
plt.figure(figsize=(12, 6))
bar_plot = column_average.plot(kind='bar', color='#F39C12', edgecolor='black')
# Set y-axis to logarithmic scale
plt.yscale('log')

# Add titles and labels
plt.title('Average of the Dataset for each Level of Incompleteness')
plt.xlabel('Columns')
plt.ylabel('Average (Log Scale)')

# Add gridlines for better readability
plt.grid(True, which="both", linestyle='--', linewidth=0.3, alpha=0.5)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust layout to prevent clipping
plt.tight_layout()

# Show the plot
plt.show()

"""
================================ PLOT THREE ================================
"""

file_path1 = "../../out/delimiter_difference/comma_delimiter_score.csv"
file_path2 = "../../out/advanced/advanced_score.csv"

df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

df1 = df1.rename(columns=columns_rename)
df2 = df2.rename(columns=columns_rename)

# Filter the rows corresponding to '2005_nenios' and '2024_topo_sim'
entries_to_compare = ['2005_nenios', '2024_topo_sim']
df1_filtered = df1[df1['Id'].isin(entries_to_compare)]
df2_filtered = df2[df2['Id'].isin(entries_to_compare)]

# Remove the 'Id' column and extract relevant metrics
extracted_df1 = df1_filtered[['Total Recall', 'Total Precision', 'Total F1 Score']]
extracted_df2 = df2_filtered[['Total Recall', 'Total Precision', 'Total F1 Score']]

# Calculate the average for both datasets
avg_df1 = extracted_df1.mean()
avg_df2 = extracted_df2.mean()

# Combine the averages into one DataFrame for comparison
comparison_df = pd.DataFrame({
    'Comma Delimiter': avg_df1,
    'Semicolon Delimiter': avg_df2
})

# Plot the comparison
plt.figure(figsize=(12, 6))
comparison_df.plot(kind='bar', width=0.35, color=['#2C3E50', '#F39C12'], edgecolor='black')
plt.title('Comma vs Semicolon Delimiter')
plt.ylabel('Average Scores')
plt.xticks(rotation=0, ha='center')
plt.legend(loc='best')
plt.grid(axis='y', linestyle='--', linewidth=0.3, alpha=0.5)
plt.tight_layout()
plt.show()

"""
================================ PLOT FOUR AND FIVE ================================
"""

# Filter for the '2005_nenios' entry in both datasets
df1_filtered = df1[df1['Id'] == '2005_nenios']
df2_filtered = df2[df2['Id'] == '2005_nenios']

# Exclude the 'Id' column and any column that contains 'tp', 'fp', or 'fn'
df1_filtered = df1_filtered.drop(columns=['Id']).filter(regex='^(?!.*(tp|fp|fn)).*$')
df2_filtered = df2_filtered.drop(columns=['Id']).filter(regex='^(?!.*(tp|fp|fn)).*$')

# Convert all other columns to numeric, forcing non-numeric to NaN, then fill NaN with 0
df1_filtered = df1_filtered.apply(pd.to_numeric, errors='coerce').fillna(0)
df2_filtered = df2_filtered.apply(pd.to_numeric, errors='coerce').fillna(0)

# Calculate the average for each level of incompleteness
df1_average = df1_filtered.mean()
df2_average = df2_filtered.mean()

# Combine the averages into one DataFrame for comparison
comparison_df = pd.DataFrame({
    'Comma Delimiter': df1_average,
    'Semicolon Delimiter': df2_average
})

# Create the bar plot
plt.figure(figsize=(12, 6))
comparison_df.plot(kind='bar', color=['plum', 'teal'], edgecolor='black')

# Set y-axis to logarithmic scale
plt.yscale('log')

# Add titles and labels
plt.title('Delimiter Comparsion: 2005_nenios Levels of Incompleteness)')
plt.xlabel('Columns')
plt.ylabel('Score (Log Scale)')

# Add gridlines for better readability
plt.grid(True, which="both", linestyle='--', linewidth=0.3, alpha=0.5)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust layout to prevent clipping
plt.tight_layout()

# Show the plot
plt.show()

"""
-------------------------
"""
# Filter for the '2024_topo_sim' entry in both datasets
df1_filtered = df1[df1['Id'] == '2024_topo_sim']
df2_filtered = df2[df2['Id'] == '2024_topo_sim']

# Exclude the 'Id' column and any column that contains 'tp', 'fp', or 'fn'
df1_filtered = df1_filtered.drop(columns=['Id']).filter(regex='^(?!.*(tp|fp|fn)).*$')
df2_filtered = df2_filtered.drop(columns=['Id']).filter(regex='^(?!.*(tp|fp|fn)).*$')

# Convert all other columns to numeric, forcing non-numeric to NaN, then fill NaN with 0
df1_filtered = df1_filtered.apply(pd.to_numeric, errors='coerce').fillna(0)
df2_filtered = df2_filtered.apply(pd.to_numeric, errors='coerce').fillna(0)

# Calculate the average for each level of incompleteness
df1_average = df1_filtered.mean()
df2_average = df2_filtered.mean()

# Combine the averages into one DataFrame for comparison
comparison_df = pd.DataFrame({
    'Comma Delimiter': df1_average,
    'Semicolon Delimiter': df2_average
})

# Create the bar plot
plt.figure(figsize=(12, 6))
comparison_df.plot(kind='bar', color=['plum', 'teal'], edgecolor='black')

# Set y-axis to logarithmic scale
plt.yscale('log')

# Add titles and labels
plt.title('Delimiter Comparsion: 2024_topo_sim Levels of Incompleteness')
plt.xlabel('Columns')
plt.ylabel('Score (Log Scale)')

# Add gridlines for better readability
plt.grid(True, which="both", linestyle='--', linewidth=0.3, alpha=0.5)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust layout to prevent clipping
plt.tight_layout()

# Show the plot
plt.show()

