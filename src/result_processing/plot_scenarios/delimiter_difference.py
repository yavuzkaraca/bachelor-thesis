from result_processing.plotting import plot_bar, load_and_prepare_data, color_palette, columns_rename, rows_rename

csv_files = {
    '2021_init-v': '../../out/exploration/2021_init-v_exploration_score.csv',
    '2022_neutero': '../../out/exploration/2022_neutero_exploration_score.csv',
    '2023_octo': '../../out/exploration/2023_octo_exploration_score.csv',
    'advanced': '../../out/advanced/advanced_score.csv',
    'comma_delimiter': '../../out/delimiter_difference/comma_delimiter_score.csv',
}

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