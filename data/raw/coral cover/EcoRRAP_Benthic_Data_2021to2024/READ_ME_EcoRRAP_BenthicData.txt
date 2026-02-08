This README explains how the EcoRRAP benthic composition data was generated, cleaned, and structured for use in further analysis. The data originates from the EcoRRAP-photoquadrat project and was processed using ReefCloud's AI-assisted annotation pipeline.

- 42 images per EcoRRAP plot were assessed in ReefCloud with 50 points per image. The ReefCloud model was trained on a separate set of images (12 images per plot) in 2021, 2022, and 2023 and from various plots (P1-P4). The ReefCloud model was then applied to the surveys containing 42 images. Both sets of images (12 per plot and 42 per plot) were derived from the same 3D models, however, it should be noted that no human annotations were performed on the dataset described in this document. The 12 image dataset is available upon request. Please contact m.toor@aims.gov.au for access.
- The label set used for annotation can be found here with brief explanations of each label, which follow the CATAMI classification scheme: https://aimsgovau.sharepoint.com/:x:/s/RRAPEcoRRAP/ERbZizKxXLpBoN0cd1AFQZ8BYPIwT6UY3C2266qNNE1Nlw?e=MuaU3v
- 17% of images were annotated manually while the remaining images were annotated by the ReefCloud model. For further information about model training please read the report "Report_Benthic_Annotation_20230905" that is accessible via the metadata record in the AIMS Data Centre: https://tsv-apps.aims.gov.au/metadata/view/48de2e3c-52a9-42f0-84ac-eb13baab0b07

The dataset includes 352 plots from 2021, 336 plots from 2022, 327 plots from 2023, and 192 plots from 2024.
The following plots are not included because they were not imaged due to field constraints:
2022				2023					2024				
KEMD_BA1_S All plots 		KEML_BA1_S All plots 			CBHE All plots			
ONLI_BA1_D All plots 		ONMO_BA1D_P1 				CBLM All plots
ONLI_LA1_S All plots 		TSDU All plots (except BA1S)		Keppel Islands All plots
ONMO_FR2_D All plots 							ONMO All plots
									ONLI All plots

* Note that only the Central and Torres Strait clusters were imaged in 2024 with all plots imaged.
---

The file 

Finalised_EcoRRAP_ReefCloud_BenthicData_2021to2024_42.csv is the RAW data as it comes exported from ReefCloud for all surveys containing 42 images. This excludes any superseded surveys that may still be present in the Reefcloud Project. In this file, each row represents the cover estimate per single image. Each survey will contains ~42 images and thus ~42 rows.
Due to the image extraction process, some surveys may have slightly less than 42 images.

Fields included in above files:
 - Date(UTC): date (year and month) that the plot was imaged. Day is always the 1st of the month of imaging. Time of imaging listed as 10:00:00am for all plots
 - Year: Year that the plot was imaged
 - Project: ReefCloud project name (EcoRRAP-photoquadrat)
 - Site: abbreviation for EcoRRAP reef, including cluster and reef name. See bottom of this document for full list of abbreviations
 - Depth_m: 5m average depth for shallow sites, 12m average depth for deep sites
 - Transect: EcoRRAP plot number (1-4)
 - Total: number of points annotated per image
 - Org_name: ReefCloud project name (EcoRRAP-photoquadrat)
 - Survey_ID: ReefCloud specific ID
 - Site_ID: ReefCloud specific ID


This file then went through the first data cleaning steps using Python script: https://colab.research.google.com/drive/1EqjGsaVuVBRsFh6gUcwSkv6O5zEI7EBb?usp=sharing 
This script maps the classes in ReefCloud to the higher-level subgroups and functional groups. 



The outputs of this were then run through the next cleaning process using R to remove extraneous columns and rows and pool sites into habitat (ie. BA1 to BA and FR1 and FR2 into just FR, etc.) with the following script: EcoRRAP_reefcloud_cleanup.Rmd
This generated the following cleaned files:
 - reefcloudcover42: all of the classes used in the ReefCloud label set
 - reefcloudcover42_functionalgroup42: broad taxonomic groups (e.g. hard coral, soft coral, etc.)
 - reefcloudcover42_subgroup42: intermediate groupings (e.g. Massive, foliose, etc.)

These three files are analysis ready and can easily be filtered by cluster, reef, site, zone, year and plot (see below for explanation of each field). 
Note: This will have to be repeated and paths changed in the script if additional taxonomic levels are required. 

Fields included in the above files:
 - Date(UTC): date (year and month) that the plot was imaged. Day is always the 1st of the month of imaging. Time of imaging listed as 10:00:00am for all plots
 - Year: Year that the plot was imaged
 - Cluster: EcoRRAP reef grouping. See bottom of this document for list of clusters
 - Reef: Reefs surveyed. See bottom of this document for list of reefs
 - Reef_pooled: Same as above reefs, with the exception of reefs in Keppel Islands combined into one group (KE) and reefs in the Palm Islands combined into one group (PA)
 - Site: EcoRRAP site designations. See bottom of this document for list of site abbreviations
 - Habitat: habitat type. See bottom of document for list of habitats
 - Zone: indicates depth of survey, shallow (5m) or deep (12m)
 - Plot: EcoRRAP plot number (1-4)
 - Survey_ID: ReefCloud specific ID
 - Site_ID: ReefCloud specific ID
 - Date: date of imaging (YearMonth)

---

Important notes on the data:

- The deep and shallow flank sites at Moore Reef (ONMO_FL1D & ONMO_FL1S) were not imaged on the 2022 field trip due to logistical problems. 
	- Therefore they were imaged on a fieldtrip in February 2023 as well as the planned 2023 time point in October 2023. 
	- In this dataset, the 02/2023 imaging is considered as part of the 2022 dataset and both the survey title and year say 2022.

- The majority of images used were selected with a python script that maximised spatial coverage based on the plot's 3D model. 
	- Where less than 12 images were extracted, the remaining images were extracted manually. 
	- A record of plots and the number of images per plot that were manually extracted is available, contact Maren Toor (m.toor@aims.gov.au) for more information if required. 

---

EcoRRAP Dictionary

Clusters:
 - CB: Capricorn Bunkers
 - KE: Keppel Islands
 - OC: Offshore Central
 - PA: Palm Islands
 - ON: Offshore Northern
 - TS: Torres Strait

Reefs: 
 - LM: Lady Musgrave
 - HE: Heron Island
 - HW: Halfway Island (Keppel Island group)
 - GK: Greak Keppel Island
 - MD: Middle Island (Keppel Island group)
 - ML: Miall Island (Keppel Island group)
 - NK: North Keppel Island
 - DA: Davies Reef
 - LB: Little Broadhurst Reef
 - CH: Chicken Reef
 - OR: Orpheus Island
 - PE: Pelorus Island
 - MO: Moore Reef
 - LI: Lizard Island
 - AU: Aukane Island
 - DU: Dungeness Reef
 - MA: Masig Island

Habitats (numbers associated with habitat abbreviation indicate multiple sites of a habitat type within a reef, e.g. BA1 is 'back 1' while BA2 is 'back 2'):
 - BA: back
 - FL: flank
 - FR: front
 - LA: lagoon