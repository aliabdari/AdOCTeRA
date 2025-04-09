import pandas as pd

dataframe = pd.read_csv('./generated_results/results_table_2_tmp.csv')

data = {
    "m1": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    "m2": [dataframe.iloc[0]['m2'], dataframe.iloc[5]['m2'], dataframe.iloc[10]['m2'], dataframe.iloc[15]['m2'], dataframe.iloc[20]['m2'], dataframe.iloc[25]['m2'], dataframe.iloc[30]['m2']],
    "m3": [dataframe.iloc[0]['m3'], dataframe.iloc[5]['m3'], dataframe.iloc[10]['m3'], dataframe.iloc[15]['m3'], dataframe.iloc[20]['m3'], dataframe.iloc[25]['m3'], dataframe.iloc[30]['m3']],
    "25%_75% thresh": [dataframe.iloc[0]['MedR'], dataframe.iloc[5]['MedR'], dataframe.iloc[10]['MedR'], dataframe.iloc[15]['MedR'], dataframe.iloc[20]['MedR'], dataframe.iloc[25]['MedR'], dataframe.iloc[30]['MedR']],
    "35%_75% thresh": [dataframe.iloc[1]['MedR'], dataframe.iloc[6]['MedR'], dataframe.iloc[11]['MedR'], dataframe.iloc[16]['MedR'], dataframe.iloc[21]['MedR'], dataframe.iloc[26]['MedR'], dataframe.iloc[31]['MedR']],
    "45%_75% thresh": [dataframe.iloc[2]['MedR'], dataframe.iloc[7]['MedR'], dataframe.iloc[12]['MedR'], dataframe.iloc[17]['MedR'], dataframe.iloc[22]['MedR'], dataframe.iloc[27]['MedR'], dataframe.iloc[32]['MedR']],
    "35%_50% thresh": [dataframe.iloc[3]['MedR'], dataframe.iloc[8]['MedR'], dataframe.iloc[13]['MedR'], dataframe.iloc[18]['MedR'], dataframe.iloc[23]['MedR'], dataframe.iloc[28]['MedR'], dataframe.iloc[33]['MedR']],
    "35%_90% thresh": [dataframe.iloc[4]['MedR'], dataframe.iloc[9]['MedR'], dataframe.iloc[14]['MedR'], dataframe.iloc[19]['MedR'], dataframe.iloc[24]['MedR'], dataframe.iloc[29]['MedR'], dataframe.iloc[34]['MedR']]
}

df = pd.DataFrame(data)

csv_filename = "./generated_results/results_table_2.csv"
df.to_csv(csv_filename, index=False)
