# Corn Prediction using Satellite Images       
As part of 50.038 Computational Data Science module from SUTD, our group embarked on a data science project aimed at ____. One portion of this project involves the use of satellite images to predict corn yield and export data to predict whether the price of corn would increase or decrease.      

## Running the project     
Install the required dependencies          

```
pip install -r requirements.txt
```

Retrieve and preprocess the satellite data needed by running the notebook (put notebook directory here)       

Models are defined in models.py file, and they are trained and evaluated in the train_and_eval.py file.

## Data Sources        
Satellite images were taken from Google Earth Engine. The specific satellite datasets used in this project are listed below:       
• [USDA NASS Cropland Data Layers] (https://developers.google.com/earth-engine/datasets/catalog/USDA_NASS_CDL#bands)
• [MYD11A2.061 Aqua Land Surface Temperature and Emissivity 8-Day Global 1km] (https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MYD11A2)
• [MOD09A1.061 Terra Surface Reflectance 8-Day Global 500m] (https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09A1)

data taken from here: https://www.nass.usda.gov/Quick_Stats/Lite/index.php      

reference project: https://github.com/tpjoe/SpringBoard
