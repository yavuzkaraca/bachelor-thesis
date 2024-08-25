import pandas as pd
import matplotlib.pyplot as plt

# File paths to CSV files
init = '../../out/exploration/2021_init-v_exploration_score.csv'
neutero = '../../out/exploration/2022_neutero_exploration_score.csv'
octo = '../../out/exploration/2023_octo_exploration_score.csv'

# Load the data
df1 = pd.read_csv(init)
df2 = pd.read_csv(neutero)
df3 = pd.read_csv(octo)

# Add a column to each DataFrame to identify the source
df1['Project'] = '2021 init-v'
df2['Project'] = '2022 neutero'
df3['Project'] = '2023 octo'

# Combine the DataFrames
combined_df = pd.concat([df1, df2, df3])

# Define the custom order for 'Id', placing combined IDs at the top
custom_order = [
    'Combined GK + CT',
    'Combined CoT + RI',
    'Combined All',
] + [id for id in combined_df['Id'].unique() if id not in ['Combined GK + CT', 'Combined CoT + RI', 'Combined All']]

# Reindex the DataFrame according to this custom order
grouped_df = combined_df.pivot_table(index='Id', columns='Project', values='F1 score')
grouped_df = grouped_df.reindex(custom_order)

# Plot the data as a bar chart
ax = grouped_df.plot(kind='bar', figsize=(10, 6), color=['#333333', '#F39C12', '#8198A0'], edgecolor='black')

# Add a vertical line after the combined bars
n_combined = len(['Combined GK + CT', 'Combined CoT + RI', 'Combined All'])
plt.axvline(x=n_combined - 0.5, color='black', linestyle='--', linewidth=1)

# Customize the plot
plt.title('F1 Scores Grouped by Prompts', fontsize=14, color='#2C3E50')
plt.xlabel('Prompt Technique', fontsize=12, color='#2C3E50')
plt.ylabel('F1 Score', fontsize=12, color='#2C3E50')
plt.xticks(rotation=45, ha='right', color='#2C3E50')
plt.yticks(color='#2C3E50')
plt.legend(title='Document', title_fontsize='13', fontsize='11')

# Set the background color for the plot area and figure
ax.set_facecolor('#F8F8F8')  # Lighter gray for the plot area
plt.gcf().set_facecolor('#FFFFFF')  # White background for the figure

# Show the plot
plt.tight_layout()
plt.show()
