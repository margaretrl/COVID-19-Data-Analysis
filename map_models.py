import pandas as pd
import plotly.express as px
import json
from urllib.request import urlopen

#############################
### UNEMPLOYMENT MAP PLOT ###
#############################

# Load unemployment
df = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
    dtype={"fips": str},
)

with urlopen(
    "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
) as response:
    geo_json_data = json.load(response)

fig = px.choropleth(
    df,
    geojson=geo_json_data,
    locations="fips",
    color="unemp",
    color_continuous_scale="Inferno_r",
    range_color=(0, 12),
    scope="usa",
    labels={"unemp": "Unemployment Rate"},
)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show()

#############################
###### CASES MAP PLOT ######
#############################

df = pd.read_csv(
    "/Users/margaretrivas/Desktop/IDC4140/finalProject/datasets/confirmed_cases_by_year_updated.csv",
    dtype={"countyFIPS": str},
)

fig = px.choropleth(
    df,
    geojson=geo_json_data,
    locations="countyFIPS",
    color="2020_total",
    color_continuous_scale="Inferno_r",
    range_color=(0, 200000),
    scope="usa",
    labels={"2020_total": "COVID-19 Cases 2020"},
)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show()


#############################
###### DEATH MAP PLOT ######
#############################

df = pd.read_csv(
    "/Users/margaretrivas/Desktop/IDC4140/finalProject/datasets/death_percentage.csv",
    dtype={"countyFIPS": str},
)

fig = px.choropleth(
    df,
    geojson=geo_json_data,
    locations="countyFIPS",
    color="deaths_pct_2020",
    color_continuous_scale="Inferno_r",
    range_color=(0, 15),
    scope="usa",
    labels={"deaths_pct_2020": "COVID-19 Deaths 2020"},
)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show()


#####################################
# Cleaning Poverty Estimate Dataset #
#####################################

dfPovertyEstimate = pd.read_csv(
    "datasets/PovertyEstimates_2020.csv",
    usecols=[
        "FIPS_code",
        "Stabr",
        "Area_name",
        "POVALL_2020",
        "PCTPOVALL_2020",
        "POV017_2020",
        "PCTPOV017_2020",
        "POV517_2020",
        "PCTPOV517_2020",
        "MEDHHINC_2020",
    ],
)

dfPovertyEstimate = dfPovertyEstimate.replace(",", "", regex=True)
dfPovertyEstimate["POVALL_2020"] = pd.to_numeric(dfPovertyEstimate["POVALL_2020"])
dfPovertyEstimate["MEDHHINC_2020"] = pd.to_numeric(dfPovertyEstimate["MEDHHINC_2020"])


fig = px.choropleth(
    dfPovertyEstimate,
    geojson=geo_json_data,
    locations="FIPS_code",
    color="PCTPOVALL_2020",
    color_continuous_scale="Inferno_r",
    range_color=(0, 30),
    scope="usa",
    labels={
        "PCTPOVALL_2020": "Estimated percent of people of all ages in poverty 2020"
    },
)

fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show()
