import pandas as pd

from result_processing.plotting.abstract_plot import load_and_prepare_data
from result_processing.plotting.scenarios import plot_exploration, plot_scores_for_documents, \
    plot_dataset_wide_level_scores, plot_comparison

# File paths to CSV files
csv_files = {
    '2021_init-v': '../../../out/exploration/2021_init-v_exploration_score.csv',
    '2022_neutero': '../../../out/exploration/2022_neutero_exploration_score.csv',
    '2023_octo': '../../../out/exploration/2023_octo_exploration_score.csv',
    'full_evaluation': '../../../out/full_evaluation/full_evaluation_score.csv',
    'comma_delimiter': '../../../out/delimiter_difference/comma_delimiter_score.csv',
}


def main():
    # Exploration
    df_init = load_and_prepare_data(csv_files['2021_init-v'], '2021 - init-v')
    df_neutero = load_and_prepare_data(csv_files['2022_neutero'], '2022 - neutero')
    df_octo = load_and_prepare_data(csv_files['2023_octo'], '2023 - octo')
    combined_df = pd.concat([df_init, df_neutero, df_octo])
    plot_exploration(combined_df)

    # Full Evaluation
    df_full = load_and_prepare_data(csv_files['full_evaluation'])
    plot_scores_for_documents(df_full)
    plot_dataset_wide_level_scores(df_full)

    # Delimiter Difference
    df_comma = load_and_prepare_data(csv_files['comma_delimiter'])
    df_semi = load_and_prepare_data(csv_files['full_evaluation'])

    print(df_comma)
    print(df_semi)

    plot_comparison(
        df_comma,
        df_semi,
        entries_to_compare=['2005_nenios', '2024_topo_sim'],
        title='Comma vs Semicolon Delimiter'
    )


if __name__ == "__main__":
    main()
