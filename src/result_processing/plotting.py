import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame

# File paths to CSV files
csv_files = {
    '2021_init-v': '../../out/exploration/2021_init-v_exploration_score.csv',
    '2022_neutero': '../../out/exploration/2022_neutero_exploration_score.csv',
    '2023_octo': '../../out/exploration/2023_octo_exploration_score.csv',
    'advanced': '../../out/advanced/advanced_score.csv',
    'comma_delimiter': '../../out/delimiter_difference/comma_delimiter_score.csv',
}

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

# Define the new color palette based on the provided image
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
    df = pd.read_csv(file_path)
    if document_name:
        df['Document'] = document_name
    return df


def combine_data(dfs):
    return pd.concat(dfs)


def plot_bar(data, title, xlabel, ylabel, xtick_rotation=45, xtick_ha='right', yscale=None, colors=None,
             axv_line=0, width=0.4, figsize=(12, 6)):
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


def main():
    # Load data
    df1 = load_and_prepare_data(csv_files['2021_init-v'], '2021 - init-v')
    df2 = load_and_prepare_data(csv_files['2022_neutero'], '2022 - neutero')
    df3 = load_and_prepare_data(csv_files['2023_octo'], '2023 - octo')
    combined_df = combine_data([df1, df2, df3])
    df_adv = load_and_prepare_data(csv_files['advanced'])
    df_adv = df_adv.rename(columns=columns_rename)
    df_adv['Id'] = df_adv['Id'].replace(rows_rename)

    # Exploration Plot
    custom_order = [
                       'Combined GK + CT',
                       'Combined CoT + RI',
                       'Combined All',
                   ] + [id for id in combined_df['Id'].unique() if
                        id not in ['Combined GK + CT', 'Combined CoT + RI', 'Combined All']]

    plot_grouped_bar(combined_df, custom_order, 'F1 Scores Grouped by Prompts', 'Prompt Technique', 'F1 Score')

    # Document Total Scores Dataset-wide
    extracted_df = df_adv[['Id', 'Total Precision', 'Total Recall', 'Total F1 Score']].set_index(
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

    # Level Detailed average Dataset-wide
    df_filtered = df_adv.drop(columns=['Id']).filter(regex='^(?!.*(tp|fp|fn)).*$').apply(pd.to_numeric,
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


if __name__ == "__main__":
    main()
