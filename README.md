# The Impact of Forecast Characteristics on the Forecast Value for the Dispatchable Feeder

This repository contains code and data to replicate the results in "The Impact of Forecast Characteristics on the Forecast Value for
the Dispatchable Feeder".

> Dorina Werling, Maximilian Beichter, Benedikt Heidrich, Kaleb Phipps, Ralf
> Mikut, and Veit Hagenmeyer. 2023. The Impact of Forecast Characteristics
> on the Forecast Value for the Dispatchable Feeder. In The 14th ACM International Conference on Future Energy Systems (e-Energy ’23 Companion), June 20–23, 2023, Orlando, FL, USA. ACM, New York, NY, USA, 13 pages.
> https://doi.org/10.1145/3599733.3600251

## Acknowledgements and Funding

This project is funded by the Helmholtz Association’s Initiative and
Networking Fund through Helmholtz AI, the Helmholtz Association under the Program “Energy System Design”, and the German
Research Foundation (DFG) as part of the Research Training Group
2153 “Energy Status Data: Informatics Methods for its Collection,
Analysis and Exploitation”.


## Environment

Use the given requirements.txt to create a python environment with the python version 3.9.7.

## Quick Start:

To start the experiment for a specific building (bldg_id) and a specific dataset with potentially manipulated loads (load_factor), you need to call

```python
python pipeline.py --id bldg_id --factor load_factor
```

Note that you have to create the dataset with the corresponding load_factor in advance.

## Data
 The [solar home electricity dataset](https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data) was used for the paper. This consists of the years 2010 - 2013, which are divided into individual files. To run the forecasting pipeline on it, the data must be downloaded and prepared so that all 3 years are concatenated and within the columns the prosumption data of a single house are located. Additionally the factors have to be applied.
Further the data has to be placed at "data/solar_home_all_data_2010-2013{args.factor}.csv" , where args.factor is for example "_ldiv2".
## License

This code is licensed under the [MIT License](LICENSE).
