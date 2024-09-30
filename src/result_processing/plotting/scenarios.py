import pandas as pd

from result_processing.plotting.abstract_plot import plot_grouped_bar, plot_bar, bar_colors, columns_rename, \
    rows_rename, color_palette


def plot_exploration(dataframe):
    custom_order = [
                       'Combined GK + CT',
                       'Combined CoT + RI',
                       'Combined All',
                   ] + [id for id in dataframe['Id'].unique() if
                        id not in ['Combined GK + CT', 'Combined CoT + RI', 'Combined All']]

    plot_grouped_bar(dataframe, custom_order, 'F1 Scores Grouped by Prompts', 'Prompt Technique',
                     'F1 Score')


def plot_scores_for_documents(dataframe):
    # Document Total Scores Dataset-wide
    extracted_df = dataframe[['Id', 'Total Precision', 'Total Recall', 'Total F1 Score']].set_index(
        'Id')  # Precision first
    plot_bar(
        extracted_df,
        'Total Precision, Recall, and F1 Score by ID',  # Update title to reflect Precision first
        'Document',
        'Scores',
        xtick_rotation=0,
        xtick_ha='center',
        colors=bar_colors,
        axv_line=4,
    )


def plot_dataset_wide_level_scores(dataframe):
    df_filtered = dataframe.drop(columns=['Id']).filter(regex='^(?!.*(tp|fp|fn)).*$').apply(pd.to_numeric,
                                                                                            errors='coerce').fillna(0)

    # Select and reorder columns to ensure Precision comes first, then Recall, and finally F1 Score
    column_order = [col for col in df_filtered.columns if 'Precision' in col] + \
                   [col for col in df_filtered.columns if 'Recall' in col] + \
                   [col for col in df_filtered.columns if 'F1 Score' in col]

    df_filtered = df_filtered[column_order]
    column_average = df_filtered.sum().div(len(df_filtered))

    # Reshape data to group by Level, including "Artificial" and "Total"
    grouped_data = pd.DataFrame({
        'Level 1': [column_average['Level 1 Precision'], column_average['Level 1 Recall'],
                    column_average['Level 1 F1 Score']],
        'Level 2': [column_average['Level 2 Precision'], column_average['Level 2 Recall'],
                    column_average['Level 2 F1 Score']],
        'Level 3': [column_average['Level 3 Precision'], column_average['Level 3 Recall'],
                    column_average['Level 3 F1 Score']],
        'Artificial': [column_average['Artificial Precision'], column_average['Artificial Recall'],
                       column_average['Artificial F1 Score']],
        'Total': [column_average['Total Precision'], column_average['Total Recall'], column_average['Total F1 Score']]
    }, index=['Precision', 'Recall', 'F1 Score']).T

    # Plot grouped bars with correct order
    plot_bar(
        grouped_data,
        'Average of the Dataset for each Level of Incompleteness',
        'Level Specific Metric',
        'Average (Log Scale)',
        yscale='log',
        colors=bar_colors,
        axv_line=12
    )


def plot_comparison(data1, data2, entries_to_compare, title):
    df1_filtered = data1[data1['Id'].isin(entries_to_compare)]
    df2_filtered = data2[data2['Id'].isin(entries_to_compare)]

    extracted_df1 = df1_filtered[['Total Recall', 'Total Precision', 'Total F1 Score']]
    extracted_df2 = df2_filtered[['Total Recall', 'Total Precision', 'Total F1 Score']]

    avg_df1 = extracted_df1.mean()
    avg_df2 = extracted_df2.mean()

    comparison_df = pd.DataFrame({
        'Comma Delimiter': avg_df1,
        'Semicolon Delimiter': avg_df2
    })

    plot_bar(
        comparison_df,
        title,
        'Metrics',
        'Average F1 Scores',
        xtick_rotation=0,
        xtick_ha='center',
        colors=[color_palette['dark_blue'], color_palette['orange']],
        width=0.25,
        figsize=(9, 6)
    )