import pandas as pd
import matplotlib.pyplot as plt


init = '../../dataset/PSE/2021_init-v/exploration.csv'
neutero = '../../dataset/PSE/2022_neutero/exploration.csv'
octo = '../../dataset/PSE/2023_octo/exploration.csv'

df1 = pd.read_csv(init)
df2 = pd.read_csv(neutero)
df3 = pd.read_csv(octo)

# Add a column to each DataFrame to identify the source
df1['Project'] = '2021 init-v'  # Replace with actual project name or identifier
df2['Project'] = '2022 neutero'
df3['Project'] = '2023 octo'

# Combine the DataFrames
combined_df = pd.concat([df1, df2, df3])

# Filter the DataFrame by 'Id'
allowed_ids = ['Combined GK + CT', 'Combined CoT + RI', 'Combined All']
filtered_df = combined_df[combined_df['Id'].isin(allowed_ids)]

# Step 2: Group data by 'Id' and 'Project'
grouped_df = filtered_df.pivot_table(index='Id', columns='Project', values='F1 score')

# Step 3: Plot the data as a bar chart
grouped_df.plot(kind='bar', figsize=(10, 6))

# Customize the plot
plt.title('F1 Scores Grouped by Prompts')
plt.xlabel('Prompt Technique')
plt.ylabel('F1 Score')
plt.xticks(rotation=0, ha='right')
plt.legend(title='Document')

# Show the plot
plt.tight_layout()
plt.show()
