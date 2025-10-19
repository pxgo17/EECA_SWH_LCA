# Supporting Data for EECA SWH LCA project 

This folder contains processed and raw datasets supporting the analysis of embodied emissions, power dispatch, and thermal operation.


## How to Use

Most raw datasets in this repository are used as inputs for the power dispatch model and are provided in CSV format. These files are processed by the Python scripts located in scripts/energy_model.
The majority of the datasets are derived from Electricity Market Information (EMI) datasets and supporting spreadsheets used in the EDGS scenario analysis. This folder also includes time series data generated using the pvlib and windpowerlib libraries to estimate site-specific intermittent generation.

Processed datasets are organised into three subfolders:

energy/ – Hourly time series of annual power dispatch results for the base year, reference, and environmental cases across different hydrological years.

thermal/ – Aggregated operational results from thermal modelling, including total annual electricity generation and emissions for each technology, household, EDGS scenario, and hydro condition.

lca/ – Outputs from the integration of embodied energy and emissions with operational analysis results.


## Links to original sources

EDGS scenario data: 

  https://www.mbie.govt.nz/building-and-energy/energy-and-natural-resources/energy-statistics-and-modelling/energy-modelling/electricity-de   mand-and-generation-scenarios

EMI temporal power dispatch series: 

  https://www.emi.ea.govt.nz/Wholesale/Datasets/Generation/Generation_MD
  https://www.emi.ea.govt.nz/Retail/Dashboards
  
Weather files for NZ: 
  
  https://www.building.govt.nz/getting-started/climate-change-work-programme/resources/weather-files-aotearoa-new-zealand
  https://niwa.co.nz/climate-and-weather/generating-synthetic-wind-data#:~:text=NIWA%20has%20created%20synthetic%2C%20multi,farms%20on%20th   e%20national%20grid.

Hydro-power information: 
  
  https://www.emi.ea.govt.nz/Environment/Reports/3UN1KD?_si=v|3
  https://energy.nzx.com/
  
  