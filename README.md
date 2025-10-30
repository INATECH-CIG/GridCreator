# GridCreator
Automatically generate distribution networks based on open data

The aim is to build a tool for automated generation of synthetic low-voltage networks based on open data such as
- open street map (osm)
- Zensus 2022
- Marktstammdatenregister

The tool builds up on existing projects and databases and combines them in a single tool to cover the whole pipeline from choosing an area and timerange to having a fully parametrised distribution grid.
The output distribution grid will contain
- lines, buses, substations
- households and installed technologies
- timeseries for renewable generation
- timeseries for household electricity demand (loads, electric vehicles, heat-pumps)
- industrial and commercial users and installed technologies
- timeseries for industrial heat demand, electricity demand and generation

## Documentation
tbd

## Installation
1. Clone the repository and navigate to its directory:
```bash
git clone https://github.com/INATECH-CIG/GridCreator.git
cd assume
```
2. Set up an environment
```bash
conda create -n GridCreator python=3.12.11
```
3. Install required packages
```bash
conda activate GridCreator
pip install -r requirements.txt
```

## Necessary input data
for step 1:
- download input.zip containing weather data, zensus data and ding0 grids from bwsyncandshare cloud
- 

## Release Status
This Repo is in work in progress.
