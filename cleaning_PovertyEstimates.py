import pandas as pd 

df = pd.read_csv('datasets/PovertyEstimates_2020.csv', usecols=['FIPS_code', 'Stabr', 'Area_name','POVALL_2020', 'PCTPOVALL_2020', 'POV017_2020', 'PCTPOV017_2020','POV517_2020', 'PCTPOV517_2020', 'MEDHHINC_2020'])

# FIPS 15005 (Kalawao County) was all NULL so they were filled with 0
df = df.fillna(0)

