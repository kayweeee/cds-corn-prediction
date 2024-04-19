# Corn Prediction using Satellite Images       
As part of 50.038 Computational Data Science module from SUTD, our group embarked on a data science project aimed at creating a multi-modal machine learning-based framework for corn price prediction to assist corn farmers, businesses and investors with managing price fluctuation risks and maximising their profits. One portion of this project involves the use of satellite images to predict corn yield and export data to predict whether the price of corn would increase or decrease.      

## Running the Satellite Yield Prediciton Project    
### Install the required dependencies          

```
pip install -r requirements.txt
```
            
All files from this point will take the Satellite Yield Prediction folder as the root. 
1. Retrieve and preprocess the satellite data needed by running the ./data_retrieval_and_processing notebook.
2. To get some data visualisation, run the ./data_viz notebook.      
3. Models are defined in ./src/models.py file, and they are trained and evaluated in the ./train_and_evaluation notebook.    
4. Our pretrained models can be found in ./models foler and can be loaded into the train_and_evaluation notebook to be evaluated.        

## Data Sources        
Satellite images were taken from Google Earth Engine. The specific satellite datasets used in this project are listed below:       
• [USDA NASS Cropland Data Layers](https://developers.google.com/earth-engine/datasets/catalog/USDA_NASS_CDL#bands)            
• [MYD11A2.061 Aqua Land Surface Temperature and Emissivity 8-Day Global 1km](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MYD11A2)            
• [MOD09A1.061 Terra Surface Reflectance 8-Day Global 500m](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09A1)            
            
Corn yield and export data was taken from [USDA NASS](https://www.nass.usda.gov/Quick_Stats/Lite/index.php)     

Credits to [Thanaphong Phongpreecha](https://github.com/tpjoe/SpringBoard) for code for preprocessing satellite images from Google Earth Enginer. 
