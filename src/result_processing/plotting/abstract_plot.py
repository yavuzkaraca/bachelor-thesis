"""
This module contains abstract plotting methods that are used by concrete scenarios.
"""

import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

# Color palette definition
color_palette = {
    'gray': '#333333',
    'orange': '#F39C12',
    'blue_gray': '#8198A0',
    'light_blue': '#B0C4DE',
    'teal': 'teal',
    'dark_blue': '#2C3E50',
    'plum': 'plum',
    'background': '#F6F6F6',
    'dark_orange': '#542D13',
}

# Standard Bar Colors
bar_colors = [color_palette['teal'], color_palette['light_blue'], color_palette['gray']]

# Common column rename mapping
columns_rename = {
    'L1 tp': 'Level 1 tp',
    'L1 fp': 'Level 1 fp',
    'L1 fn': 'Level 1 fn',
    'L1 Precision': 'Level 1 Precision',
    'L1 Recall': 'Level 1 Recall',
    'L1 F1 Score': 'Level 1 F1 Score',
    'L2 tp': 'Level 2 tp',
    'L2 fp': 'Level 2 fp',
    'L2 fn': 'Level 2 fn',
    'L2 Precision': 'Level 2 Precision',
    'L2 Recall': 'Level 2 Recall',
    'L2 F1 Score': 'Level 2 F1 Score',
    'L3 tp': 'Level 3 tp',
    'L3 fp': 'Level 3 fp',
    'L3 fn': 'Level 3 fn',
    'L3 Precision': 'Level 3 Precision',
    'L3 Recall': 'Level 3 Recall',
    'L3 F1 Score': 'Level 3 F1 Score'
}

# Common row rename mapping
rows_rename = {
    '2001_esa': '2001 - esa',
    '2005_nenios': '2005 - nenios',
    '2009_video_search': '2009 - video search',
    '2010_home_1.3': '2010 - home 1.3',
    '2021_init-v': '2021 - init-v',
    '2022_neutero': '2022 - neutero',
    '2023_octo': '2023 - octo',
    '2024_topo_sim': '2024 - topo sim',
}


def load_and_prepare_data(file_path, document_name=None) -> DataFrame:
    """
    Loads dataframe from filepath and adds the document name as an entry if specified (useful for merging CSV files
    by name)
    """
    df = pd.read_csv(file_path)
    df.rename(columns=columns_rename, inplace=True)
    df['Id'] = df['Id'].replace(rows_rename)

    if document_name:
        df['Document'] = document_name

    return df


def plot_bar(data, title, xlabel, ylabel, xtick_rotation=45, xtick_ha='right', yscale=None, colors=None,
             axv_line=0, width=0.4, figsize=(12, 6)):
    """
    The default abstract plotting method, which can be customized by providing labels and more
    """
    ax = data.plot(kind='bar', figsize=figsize, color=colors, edgecolor='black', width=width)
    plt.title(title, fontsize=12, color=color_palette['dark_blue'])
    plt.xlabel(xlabel, fontsize=10, color=color_palette['dark_blue'])
    plt.ylabel(ylabel, fontsize=10, color=color_palette['dark_blue'])
    plt.xticks(rotation=xtick_rotation, ha=xtick_ha, color=color_palette['dark_blue'])
    plt.yticks(color=color_palette['dark_blue'])

    if yscale:
        plt.yscale(yscale)

    if not axv_line == 0:
        plt.axvline(x=axv_line - 0.5, color='black', linestyle='--', linewidth=1)

    plt.grid(True, which="both", axis='y', linestyle='--', linewidth=0.3, alpha=0.5)

    ax.set_facecolor(color_palette['background'])  # Lighter gray for the plot area
    plt.gcf().set_facecolor('#FFFFFF')  # Ensures the figure has a white background

    plt.tight_layout()
    plt.show()


def plot_grouped_bar(data, custom_order, title, xlabel, ylabel):
    """
    Groups and reorders bars before plotting
    """
    grouped_df = data.pivot_table(index='Id', columns='Document', values='F1 score')
    grouped_df = grouped_df.reindex(custom_order)

    plot_bar(
        grouped_df,
        title,
        xlabel,
        ylabel,
        xtick_rotation=45,
        xtick_ha='right',
        colors=[color_palette['gray'], color_palette['orange'], color_palette['blue_gray']],
        width=0.5,
        axv_line=3
    )
    plt.show()


def plot_comparison(data1, data2, entries_to_compare, title):
    """
    Compares and plots two dataframes,each holding scores for either comma delimiter or a semicolon delimiter
    """
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
