import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import stats
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

def dim_region_productivity(df, levels, category, kpi_name, kpi_column):
    all_subsets = []  

    for level in levels:
        st.write(f"[INFO] Starting process for {category} {kpi_name} {level}-based analysis...")

        df_level = df[df['level'] == level]

        for loc in df_level['location'].unique():
            st.write(f"[INFO] Processing for {level} - {loc}...")

            df_level_subset = df_level[df_level['location'] == loc].copy()
            df_level_subset = df_level_subset[['yearweek', 'kpi', 'level', 'location', kpi_column]]
        
            df_level_subset = df_level_subset.sort_values(by='yearweek').reset_index(drop=True)

            h_columns = [f'H-{h}' for h in [1, 2, 3, 7, 15]]
            df_level_subset['H-1'] = df_level_subset[kpi_column].diff(1)

            for h in [2, 3, 7, 15]:
                df_level_subset[f'H-{h}'] = df_level_subset[kpi_column] - df_level_subset[kpi_column].shift(1).rolling(window=h).mean()

            df_level_subset['anomaly_naik'] = (df_level_subset[h_columns] > 2).sum(axis=1)
            df_level_subset['anomaly_turun'] = (df_level_subset[h_columns] < -2).sum(axis=1)

            df_level_subset['anomaly_detection'] = np.where(
                df_level_subset['anomaly_naik'] > 0, 'anomaly naik',
                np.where(df_level_subset['anomaly_turun'] > 0, 'anomaly turun', 'normal')
            )

            df_level_subset['anomali_action'] = np.where(
                (df_level_subset['anomaly_naik'] > 3) | (df_level_subset['anomaly_turun'] > 3), 'replace',
                np.where((df_level_subset['H-15'] > 2) & (df_level_subset['H-1'] < 2), 'shifting naik',
                         np.where((df_level_subset['H-15'] < -2) & (df_level_subset['H-1'] < -2), 'shifting turun', ''))
            )

            df_level_subset['valid_value'] = np.where(df_level_subset['anomali_action'] == 'replace', np.nan, df_level_subset[kpi_column])
            df_level_subset['profile'] = np.nan
            df_level_subset.loc[1:4, 'profile'] = df_level_subset.loc[1:4, 'valid_value']

            for i in range(5, len(df_level_subset)):
                previous_week_index = i - 4
                if previous_week_index >= 1:
                    if pd.isna(df_level_subset.loc[i, 'valid_value']):
                        df_level_subset.loc[i, 'profile'] = df_level_subset.loc[previous_week_index, 'profile']
                        df_level_subset.loc[i, 'valid_value'] = df_level_subset.loc[previous_week_index, 'profile']
                    else:
                        df_level_subset.loc[i, 'profile'] = (0.75 * df_level_subset.loc[previous_week_index, 'profile'] +
                                                     0.25 * df_level_subset.loc[i, 'valid_value'])

            df_level_subset.loc[1:, 'valid_value'] = np.where(
                df_level_subset.loc[1:, 'anomali_action'] == 'replace',
                df_level_subset.loc[1:, 'profile'],
                df_level_subset.loc[1:, 'valid_value']
            )

            df_level_subset['std_dev'] = np.nan
            df_level_subset['margin'] = np.nan

            if len(df_level_subset) >= 28:
                for i in range(28, len(df_level_subset)):
                        previous_28_days = df_level_subset.loc[i-28:i-1, 'valid_value']
                        mean_28_days = previous_28_days.mean()
                        std_28_days = previous_28_days.std()

                        margin = 3.71 * (std_28_days / np.sqrt(28))
                        df_level_subset.loc[i, 'ub'] = mean_28_days + margin
                        df_level_subset.loc[i, 'lb'] = mean_28_days - margin
                        df_level_subset.loc[i, 'std_dev'] = std_28_days
                        df_level_subset.loc[i, 'margin'] = margin
            else:
                    for i in range(4, len(df_level_subset)):
                        previous_4_days = df_level_subset.loc[i-4:i-1, 'valid_value']
                        mean_4_days = previous_4_days.mean()
                        std_4_days = previous_4_days.std()

                        margin = 3.71 * (std_4_days / np.sqrt(4))
                        df_level_subset.loc[i, 'ub'] = mean_4_days + margin
                        df_level_subset.loc[i, 'lb'] = mean_4_days - margin
                        df_level_subset.loc[i, 'std_dev'] = std_4_days
                        df_level_subset.loc[i, 'margin'] = margin

            df_level_subset['alert_flag'] = np.nan
            df_level_subset.loc[df_level_subset['valid_value'] < df_level_subset['lb'], 'alert_flag'] = 1
            df_level_subset.loc[df_level_subset['valid_value'] > df_level_subset['ub'], 'alert_flag'] = 2

            df_level_subset["significant_result"] = np.nan
            
            if len(df_level_subset) >= 28:
                for i in range(28, len(df_level_subset)):
                    previous_28_days = df_level_subset.loc[i-28:i-1, "valid_value"].dropna()
                    if len(previous_28_days) < 2:
                        continue

                    t_statistic, p_value = stats.ttest_1samp(previous_28_days, df_level_subset.loc[i, "valid_value"])
                    alert_flag = df_level_subset.loc[i, "alert_flag"]
                    alpha = 0.00001

                    if p_value < alpha:
                        if alert_flag == 1:
                            df_level_subset.loc[i, "significant_result"] = "turun"
                        elif alert_flag == 2:
                            df_level_subset.loc[i, "significant_result"] = "naik"
            else:
                for i in range(4, len(df_level_subset)):
                    previous_4_days = df_level_subset.loc[i-4:i-1, "valid_value"].dropna()
                    if len(previous_4_days) < 2:
                        continue

                    t_statistic, p_value = stats.ttest_1samp(previous_4_days, df_level_subset.loc[i, "valid_value"])
                    alert_flag = df_level_subset.loc[i, "alert_flag"]
                    alpha = 0.09

                    if p_value < alpha:
                        if alert_flag == 1:
                            df_level_subset.loc[i, "significant_result"] = "turun"
                        elif alert_flag == 2:
                            df_level_subset.loc[i, "significant_result"] = "naik"

            df_level_subset["significant_result"].fillna("maintain", inplace=True)
            df_level_subset.loc[0, "significant_result"] = np.nan

            # Tambahkan kolom trend_indicator
            df_level_subset['trend_indicator'] = df_level_subset['significant_result'].apply(
                lambda x: f'<span style="color: red;">▲</span>' if x == "naik" 
                else f'<span style="color: red;">▼</span>' if x == "turun" 
                else f'<span style="color: green;">─</span>'
            )

            df_level_subset.rename(columns={kpi_column: "actual_value"}, inplace=True)

            df_level_subset['category'] = category
            df_level_subset['kpi_name'] = kpi_name

            df_level_subset['date_process'] = datetime.today().strftime('%Y-%m-%d')
            df_level_subset['yearweek_process'] = datetime.today().strftime('%Y%W')

            cols = ['yearweek_process'] + [col for col in df_level_subset.columns if col not in ['yearweek_process']]
            df_level_subset = df_level_subset[cols]
            df_level_subset = df_level_subset.drop_duplicates()
            all_subsets.append(df_level_subset)

    result_df = pd.concat(all_subsets, ignore_index=True)
    result_df = result_df[['yearweek_process', 
                                    'yearweek', 
                                    'level', 
                                    'kpi_name', 
                                    'category', 
                                    'location', 
                                    'valid_value', 
                                    'significant_result',
                                    'trend_indicator']]
    result_df['valid_value'] = result_df['valid_value'].round(2)
    result_df = result_df.drop_duplicates()
    return result_df


def main():
    st.title("KPI Analisis")

    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data yang diunggah:")
        st.write(df)

        if 'category' not in df.columns:
            category = st.text_input("Masukkan nilai untuk kolom 'category' (nilai tunggal):")
            if category:
                df['category'] = category  
        else:
            category = st.selectbox("Pilih kategori", df['category'].unique())

        if 'kpi_name' not in df.columns:
            kpi_name = st.text_input("Masukkan nilai untuk kolom 'kpi_name' (nilai tunggal):")
            if kpi_name:
                df['kpi_name'] = kpi_name  
        else:
            kpi_name = st.selectbox("Pilih KPI Name", df['kpi_name'].unique())

        levels = st.multiselect("Pilih level", df['level'].unique())
        kpi_column = st.selectbox("Pilih Kolom KPI", df.columns)

        if st.button("Proses"):
            result_df = dim_region_productivity(df, levels, category, kpi_name, kpi_column)
            st.write("Hasil Analisis:")
            
            yearweek_options = ["All"] + sorted(result_df['yearweek'].unique().tolist())
            selected_yearweek = st.selectbox(
                "Pilih Yearweek untuk melihat detail:",
                options=yearweek_options
            )


            if selected_yearweek == "All":
                filtered_df = result_df  
            else:
                filtered_df = result_df[result_df['yearweek'] == selected_yearweek]  

            st.write(filtered_df.to_html(escape=False, index=False), unsafe_allow_html=True)

if __name__ == "__main__":
    main()