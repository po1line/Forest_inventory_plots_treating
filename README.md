# Forest inventory plots treating: spatial data preparation for test sample size increasing


Forest inventory is an essential source of information that can be used for ecosystem productivity monitoring and spatial carbon accounting. One of the common ways to provide such inventory is to describe target characteristics (tree type, height, age, carbon stocks) on the level of ``forest inventory plot``. Specifics and limitations of inventory plots are that they are relatively small, which allows making precise estimations at chosen locations, but makes it difficult to combine such data with open-source satellite imagery because of the small radius of inventory plot (e.g. 9 meters, while the size of Sentinel image pixel equals to 10 m).
To cope with this limitation, the radius of the inventory plot could be manually increased, which in turn increases the number of pixels with information. This could be crucial for image classification tasks, where the amount of labelled data directly influences the quality of predictions. At such a manual radius increase, a question remains: how representative is the data compared to the initial inventory plot data?

Thus, to obtain as much meaningful information after manual radius increase as possible, a ``clustering`` could be applied. 

## Clustering step description

From the dataset obtained from satellite and radar imagery (spectral and terrain characteristics and their derivatives), we want to obtain as much representative data as possible. A buffering step is provided [here](https://github.com/mishagrol/ForestMapping). We use a simple K-means cluster algorithm. The pipeline is short and consists of two main steps:
1. for each plot, we provide clustering using ``get_cluster_pixels`` function
2. from each plot, we select only those records which correspond to the largest cluster ``get_selection``

Supporting information:
* clustering procedure can be performed in two modifications: on the set of non-correlated features ``get_cluster_pixels`` and after feature selection from PCA  ``get_cluster_pixels_PCA``
* number of clusters is selected automatically according to the "elbow rule”, but its maximum amount should be set by the user according to the number of target features, e.g., expected tree types could be seen in the plot
* For PCA modification of clustering, it should be noted that the number of components of also flexible and depends on the number of features and samples.
  
## Test data description
The test data was made using the fusion of data from inventory plots and resampled Sentinel and SRTM data. The code is provided [here](https://github.com/mishagrol/ForestMapping).

|      |Column     |  Non-Null Count |Dtype | 
|--- | ------       | --------------    | -----  |
| 0   |B01            | 5820 non-null   |float64|
| 1   |B02           |5820 non-null   |float64|
| 2   |B03           |5820 non-null   |float64|
| 3   |B04           |5820 non-null   |float64|
| 4   |B05           |5820 non-null   |float64|
| 5   |B06           |5820 non-null   |float64|
| 6   |B07           |5820 non-null   |float64|
| 7   |B08           |5820 non-null   |float64|
| 8   |B8A           |5820 non-null   |float64|
| 9   |B09           |5820 non-null   |float64|
| 10  |B11           |5820 non-null   |float64|
| 11  |B12           |5820 non-null   |float64|
| 12  |NDVI         |5820 non-null   |float64|
| 13  |EVI           |5820 non-null   |float64|
| 14  |MSAVI      |   5820 non-null   |float64|
| 15  |NDRE        |  5820 non-null   |float64|
| 16  |aspect       | 5820 non-null   |float64|
| 17  |slope         |5820 non-null   |float64|
| 18  |wetnessindex|  5820 non-null   |float64|
| 19  |sink          |5820 non-null   |float64|
| 20  |key           |5820 non-null   |int64  |
| 21  |class         |5820 non-null   |int64  |

* Columns 0-11 – values of spectral bands from Sentinel data
* Columns 12-15 – common spectral indices useful for vegetation characterisation
* Columns 16-19 – terrain characteristics obtained from SRTM mission
* ``key`` – inventory plot number
* ``class`` – dominant land cover or tree type

## Functions description:

1. ``get_cluster_pixels gets`` function requires the following arguments:   
* ``data`` – pandas DataFrame;
*  ``key`` – inventory plot number; 
*  ``correlation_threshold`` – a value for correlation between features to exclude highly correlated ones, takes values from 0 to 1. By default, it is 0.7;
* `` max_number_of_clusters`` – a value of expected types of trees or land cover types on the plot. By default, it is 5 for the territory of the study corresponding to test data.

2. ``get_selection`` function requires the following arguments:  
* ``data`` – pandas DataFrame.

