import pandas as pd
import matplotlib.pyplot as plt

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

# Common column rename mapping
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


def load_and_prepare_data(file_path, document_name=None):
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
    extracted_df = df_adv[['Id', 'Total Recall', 'Total Precision', 'Total F1 Score']].set_index('Id')
    plot_bar(
        extracted_df,
        'Total Recall, Precision, and F1 Score by ID',
        'Document',
        'Scores',
        xtick_rotation=0,
        xtick_ha='center',
        colors=[color_palette['gray'], color_palette['light_blue'], color_palette['teal']]
    )

    # Level Detailed average Dataset-wide
    df_filtered = df_adv.drop(columns=['Id']).filter(regex='^(?!.*(tp|fp|fn)).*$').apply(pd.to_numeric,
                                                                                         errors='coerce').fillna(0)
    column_average = df_filtered.sum().div(len(df_filtered))

    bar_colors = []
    for column in df_filtered.columns:
        if 'F1 Score' in column:
            bar_colors.append(color_palette['dark_blue'])  # Use the desired color for F1 Score
        else:
            bar_colors.append(color_palette['light_blue'])  # Default color or provided colors

    plot_bar(
        column_average,
        'Average of the Dataset for each Level of Incompleteness',
        'Level Specific Metric',
        'Average (Log Scale)',
        yscale='log',
        colors=bar_colors,
        axv_line=12
    )

    # Comma vs Semicolon
    df_comma = load_and_prepare_data(csv_files['comma_delimiter']).rename(columns=columns_rename)
    df_semi = load_and_prepare_data(csv_files['advanced']).rename(columns=columns_rename)

    plot_comparison(
        df_comma,
        df_semi,
        entries_to_compare=['2005_nenios', '2024_topo_sim'],
        title='Comma vs Semicolon Delimiter'
    )

    # Document wise delimiter comparison plots
    for entry in ['2005_nenios', '2024_topo_sim']:
        df1_filtered = df_comma[df_comma['Id'] == entry].drop(columns=['Id']).filter(regex='^(?!.*(tp|fp|fn)).*$')
        df2_filtered = df_semi[df_semi['Id'] == entry].drop(columns=['Id']).filter(regex='^(?!.*(tp|fp|fn)).*$')

        df1_average = df1_filtered.apply(pd.to_numeric, errors='coerce').fillna(0).mean()
        df2_average = df2_filtered.apply(pd.to_numeric, errors='coerce').fillna(0).mean()

        comparison_df = pd.DataFrame({
            'Comma Delimiter': df1_average,
            'Semicolon Delimiter': df2_average
        })

        plot_bar(
            comparison_df,
            f'Delimiter Comparison: {entry} Levels of Incompleteness',
            'Level Specific Metric',
            'Score (Log Scale)',
            yscale='log',
            colors=[color_palette['orange'], color_palette['teal']]
        )

    # 2 Plots for comma delimiter
    df_comma = load_and_prepare_data(csv_files['comma_delimiter']).rename(columns=columns_rename)
    df_comma['Id'] = df_comma['Id'].replace(rows_rename)

    extracted_df = df_comma[['Id', 'Total Recall', 'Total Precision', 'Total F1 Score']].set_index('Id')
    plot_bar(
        extracted_df,
        'Total Scores for Comma Delimiter Tested Documents',
        'Document',
        'Scores',
        xtick_rotation=0,
        xtick_ha='center',
        colors=[color_palette['gray'], color_palette['light_blue'], color_palette['teal']],
        width=0.25,
        figsize=(9, 6)
    )

    # Level Detailed average Dataset-wide
    df_filtered = df_comma.drop(columns=['Id']).filter(regex='^(?!.*(tp|fp|fn)).*$').apply(pd.to_numeric,
                                                                                         errors='coerce').fillna(0)
    column_average = df_filtered.sum().div(len(df_filtered))

    bar_colors = []
    for column in df_filtered.columns:
        if 'F1 Score' in column:
            bar_colors.append(color_palette['dark_blue'])  # Use the desired color for F1 Score
        else:
            bar_colors.append(color_palette['light_blue'])  # Default color or provided colors

    plot_bar(
        column_average,
        'Average Scores for Comma Delimiter Tested Documents',
        'Level Specific Metric',
        'Average (Log Scale)',
        yscale='log',
        colors=bar_colors,
        axv_line=12
    )


if __name__ == "__main__":
    main()
