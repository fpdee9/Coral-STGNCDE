# Code and data: Spatial variation in upper limits of coral cover on the Great Barrier Reef

[https://doi.org/10.5061/dryad.ngf1vhj44](https://doi.org/10.5061/dryad.ngf1vhj44)

## Description of the data and file structure

### Upper limits of coral communities

This repository contains the data and code to reproduce the results of the manuscript "Spatial variation in upper limits of coral cover on the Great Barrier Reef".

The scripts were run in R version 4.3.2 using the following packages:

* brms (version 2.20.4)
* patchwork (version 1.2.0)
* viridis (version 0.6.4)
* dplyr (version 1.1.4)
* ggplot2 (version 3.5.1)

To run the code, the working directory must have three folders: *Data* (which should include *coral_cover.csv* and *environmental_variables.csv*), *Code* (which should contain *analyses.R* and *functions.R*), and *Outputs* (where models and plots will be saved).

### Files and variables

#### Data files

**coral_cover.csv**: contains coral cover (Scleractinian corals) estimates from the manta tow surveys. Each row corresponds to one tow.

* YEAR_CODE: financial year when the survey was conducted (e.g., 201415 is for 2014-2015)
* REEF_NAME: name of the reef surveyed
* ZONE: wave exposure zone (categorical- *flank1* and *flank2* have intermediate wave exposure, *front* has high wave exposure, and *back* is sheltered)
* PATH: vector of coordinates of the trajectory covered by each manta tow
* HC_MIN: hard coral cover was scored as categorical intervals (0, >0-5%, >5-10%, >10-20%, >20-30%, >30-40%, >40-50%, >50-62.5%, >62.5-75%, >75-87.5%, >87.5-100%), HC_MIN is the lower end of the percentage hard coral cover interval
* HC_MAX: upper end of the percentage hard coral cover interval
* Central_LAT: latitude at the center of the tow
* Central_LON: longitude at the center of the tow
* HARD_COVER: midpoint of the coral cover interval (i.e., midpoint of HC_MIN and HC_MAX)

**environmental_variables.csv**: contains the environmental variables for each site (wave exposure zone within reef)

* Reef: name of reef
* Zone: wave exposure zone (*flank1* and *flank2* have intermediate wave exposure, *front* has high wave exposure, and *back* is sheltered)
* Latitude: site latitude
* Longitude: site longitude
* median_ubed90: 90<sup>th</sup>  quantile of horizontal water velocity at bed (m/s) (summarised as the median value for each site)
* per_suitable: proportion of hards substrate available
* temp_median: median temperature (degrees C) 
* Secchi_median: median Secchi depth (m)

## Code/software (Zenodo)

### Code

* analyses.R: estimates upper limits from time series coral cover data, fits models and generates figures shown in the manuscript
* functions.R: includes the required functions to run the analyses

## Access information

Data was derived from the following sources:

* Callaghan, D.P., Leon, J.X. & Saunders, M.I. (2015) Wave modelling as a proxy for seagrass ecological modelling: Comparing fetch and process-based predictions for a bay and reef lagoon. *Estuarine, Coastal and Shelf Science*, **153**, 108–120.
* Emslie, M.J., Bray, P., Cheal, A.J., Johns, K.A., Osborne, K., Sinclair-Taylor, T. & Thompson, C.A. (2020) Decades of monitoring have informed the stewardship and ecological understanding of Australia’s Great Barrier Reef. *Biological conservation*, **252**, 108854.
* Lyons, M.B., Roelfsema, C.M., Kennedy, E. V, Kovacs, E.M., Borrego‐Acevedo, R., Markey, K., Roe, M., Yuwono, D.M., Harris, D.L. & Phinn, S.R. (2020) Mapping the world’s coral reefs using a global multiscale earth observation framework. *Remote Sensing in Ecology and Conservation*, **6**, 557–568.
* Steven, A.D.L., Baird, M.E., Brinkman, R., Car, N.J., Cox, S.J., Herzfeld, M., Hodge, J., Jones, E., King, E. & Margvelashvili, N. (2019) eReefs: An operational information system for managing the Great Barrier Reef. *Journal of Operational Oceanography*, **12**, S12–S28.

