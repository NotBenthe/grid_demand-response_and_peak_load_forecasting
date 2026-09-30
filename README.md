# grid_demand-response_and_peak_load_forecasting


The data is the 'Smart meters in London' dataset, downloaded from [Kaggle](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london?resource=download). 

This is a subset of a dataset that measures the energy consumption in homes in England, Wales and Schotland. The dataset is a refactorised version of the data from from London data store, that contains the energy consumption records of 5567 London households that took part in the UK Power Networks led Low Carbon London project between November 2011 and February 2014. The data from the smart meters seems associated only to the electrical consumption. The details related at the acorn group are provided by the CACI, and the weather data are from darksky.


## Requirements
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
pytest>=7.3.0
flake8>=6.0.0
seaborn
```

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Project Structure
```
├── data
	└── informations_households.csv	# contains all the information on the households in the panel (their acorn group, their tariff) and in which block.csv.gz file their data are stored
	└── halfhourly_dataset          # contains the block files with the half-hourly smart meter measurement
        ├── block_0.csv
        └── ...
    └── daily_dataset               # contains the block files with the daily information like the number of measures, minimum, maximum, mean, median, sum and std.
        ├── block_0.csv
        └── ...
    └── acorn_details.csv           # Details on the acorn groups and their profile of the people in the group, it's come from this xlsx spreadsheet.The first three columns are the attributes studied, the ACORN-X is the index of the attribute. At a national scale, the index is 100 if for one column the value is 150 it means that there are 1.5 times more people with this attribute in the ACORN group than at the national scale. You can find an explanation on the CACI website
    └── weather_daily_darksky.csv   # contains the daily data from darksky api. You can find more details about the parameters in the documentation of the api
    └── weather_hourly_darksky.csv  # contains the hourly data from darksky api. You can find more details about the parameters in the documentation of the api

└── README.md         			# This file
└── requirements.txt 			# required modules for the code
```

There is 19 files in this dataset :

     : 

    

   

    

    


For me some ideas to analyze the data:

    Segmentation of the consumption daily pattern
    Disaggregation of the electricity load curve
    Cross the consumption result and the acorn information
    Forecast the electricity consumption of a household, I wrote an article on this subject
    What if I add electrical heating system ? an EV battery system ?
    Forecast at a global scale (London consumption)
