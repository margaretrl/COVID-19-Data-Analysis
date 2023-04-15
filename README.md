# COVID-19-Data-Analysis

### Members: Marija Travoric, Jacquelyn Nogueras, Margaret Rivas, Hannah Housand

### Project Description:
Exploring how COVID-19 cases or deaths are related to other variables, such as mask mandates, socioeconomic status, or healthcare access. This can help identify factors that contribute to the spread of COVID-19. We utilized linear regression to model relationships. 

### Data

Infection Data:

- [covid_deaths_usafacts.csv](https://usafacts.org/visualizations/coronavirus-covid-19-spread-map/) USAFacts Death by county

- [covid_county_population_usafacts.csv](https://usafacts.org/visualizations/coronavirus-covid-19-spread-map/) USAFacts Population by county 

- [covid_confirmed_usafacts.csv](https://usafacts.org/visualizations/coronavirus-covid-19-spread-map/) USAFacts Confirmed cases by county

Variables Data:

- [mask-use-by-county.csv](https://github.com/nytimes/covid-19-data/blob/master/mask-use/mask-use-by-county.csv) NYT Survey data for mask use

- [PovertyEstimates.csv](https://www.ers.usda.gov/data-products/county-level-data-sets/county-level-data-sets-download-data/) USDA Poverty estimate per county and state 2020

- [Unenployment.csv](https://www.ers.usda.gov/data-products/county-level-data-sets/county-level-data-sets-download-data/)USA Unenployment and median household incomde for USA states 2000-2021

- [interventions.csv](https://github.com/JieYingWu/COVID-19_US_County-level_Summaries/blob/master/data/interventions.csv) Restrictions for gatherings by date

      import datetime
      
      date = datetime.date.fromordinal(ordinal_date)
      
      print(date.month, date.day, date.year)
      
### Checklist

- [ ] Linear regression masks
- [ ] Linear regression lockdowns
- [ ] Clustering Poverty levels - Finding death hotspots
- [ ] Linear regression Poverty levels - Deaths correlation
- [ ] Clustering Poverty levels - Finding cases hotspots
- [ ] Linear regression Poverty levels - Cases correlation
- [ ] Clustering Median Household Income - Death Hotspots
- [ ] Linear regression Household Income