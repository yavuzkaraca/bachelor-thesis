import pandas as pd

from result_processing.plotting.plotting import plot_bar, load_and_prepare_data, color_palette, columns_rename, rows_rename

csv_files = {
    '2021_init-v': '../../out/exploration/2021_init-v_exploration_score.csv',
    '2022_neutero': '../../out/exploration/2022_neutero_exploration_score.csv',
    '2023_octo': '../../out/exploration/2023_octo_exploration_score.csv',
    'advanced': '../../out/advanced/advanced_score.csv',
    'comma_delimiter': '../../out/delimiter_difference/comma_delimiter_score.csv',
}




def testttt():

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
