#libraries

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

from shapely import affinity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import make_scorer

from yellowbrick.cluster import KElbowVisualizer

#for PCA option
from sklearn.decomposition import PCA

###functions


def get_cluster_pixels(data:pd.DataFrame, key: int = 1, correlation_threshold:float=0.7)->pd.DataFrame: 
    attmpt = data[data.key == key]
    attmpt_c = attmpt.drop(columns = ['key', 'class']).corr().abs() #'index',
    #attmpt.corr().style.background_gradient(cmap="Blues")

    # Select upper triangle of correlation matrix
    upper = attmpt_c.where(np.triu(np.ones(attmpt_c.shape), k=1).astype(bool))

    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]

    # Drop features 
    attmpt_ = attmpt.drop(to_drop, axis=1)#, inplace=True)

    #preprocessing of the data
    #from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(attmpt_.drop(columns = ['key', 'class']))#'index',
    scaled_data = scaler.transform(attmpt_.drop(columns = ['key', 'class']))#'index',
    #from yellowbrick.cluster import KElbowVisualizer
    model = KMeans(n_init=10)
    # k is range of number of clusters.
    visualizer = KElbowVisualizer(model, k=(1,8), timings= True)
    visualizer.fit(scaled_data)        # Fit data to visualizer
    plt.close()
    elbow_value = visualizer.elbow_value_
    if elbow_value == None:
        elbow_value = 2
    kmeans_model = KMeans(n_clusters = elbow_value, random_state=100) # elbow_value_ == number of clusters
    kmeans_model.fit(scaled_data)

    attmpt["clusters"] = kmeans_model.labels_
    attmpt.clusters.value_counts().reset_index()#.duplicated(subset=['clusters'])#.iloc[0,0]
    
    return attmpt

##selection of rows related to most abundant clusters

def get_selection(attmpt:pd.DataFrame)->pd.DataFrame:
    
    cluster_stat = attmpt.clusters.value_counts().to_dict()
    cluster_count = list(cluster_stat.values())
    cluster_non_equal = cluster_count[0]>cluster_count[1]
    if cluster_non_equal:
        target_cluster = list(cluster_stat.keys())[0]
        mask = attmpt.clusters == target_cluster
        data_grol = attmpt.loc[mask]
    else: 
        print('equal cluster')
        data_grol = pd.DataFrame()
    return data_grol

### clustering on the PCA option: before clustering we select only those features which contirbution to first two compontents is above the third quantile of the values distribution ###

def get_cluster_pixels_PCA(data:pd.DataFrame, key: int = 1)->pd.DataFrame: 
    #subsetting
    attmpt_pca_step = data[data.key == key]
    #abjusting columns
    cols_to_drop = ['key', 'class'] #'index',
    attempt_pca = attmpt_pca_step.drop(columns = cols_to_drop)

    #getting values and scalling them
    X = attempt_pca.values 
    scaler_pca = StandardScaler()
    scaler_pca.fit(X)
    X_scaled = scaler_pca.transform(X)

    #performing of principal component analysis
    pca = PCA(n_components=20, random_state=2020)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    #selecting columns for clustering

    #first component
    stat_eigen = pd.DataFrame(pca.components_, 
                              columns=attempt_pca.columns).T.loc[:, 0].abs().quantile(0.75)#.mean()#
    pca_results = pd.DataFrame(pca.components_,columns=attempt_pca.columns).T#.loc[mean_eqigen]
    mask_pca = pca_results.loc[:, 0].abs() > stat_eigen #selecting components with loads above third quartile 
    columns_for_clustering = pca_results.loc[mask_pca].index.tolist()

    #second component
    stat_eigen = pd.DataFrame(pca.components_,
                              columns=attempt_pca.columns).T.loc[:, 1].abs().quantile(0.75)#.mean()#
    pca_results = pd.DataFrame(pca.components_,columns=attempt_pca.columns).T#.loc[mean_eqigen]
    mask_pca = pca_results.loc[:, 1].abs() > stat_eigen #selecting components with loads above third quartile
    second_component = pca_results.loc[mask_pca].index.tolist()

    #final list of columns for clustering
    columns_for_clustering = columns_for_clustering+second_component

    # filtering subset by columns from previous step
    data_for_clustering = attempt_pca.loc[:, columns_for_clustering]

    #BACK TO K-MEANS

    #preprocessing of the data
    #from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(data_for_clustering)#'index',
    scaled_data = scaler.transform(data_for_clustering)#'index',

    #from yellowbrick.cluster import KElbowVisualizer
    model = KMeans()
    # k is range of number of clusters.
    visualizer = KElbowVisualizer(model, k=(1,8), timings= True)
    visualizer.fit(scaled_data)        # Fit data to visualizer
    plt.close()
    elbow_value = visualizer.elbow_value_
    if elbow_value == None:
        elbow_value = 2
    kmeans_model = KMeans(n_clusters = visualizer.elbow_value_, random_state=100) # elbow_value_ == number of clusters
    kmeans_model.fit(scaled_data)

    attmpt_pca_step["clusters"] = kmeans_model.labels_
    #attmpt.clusters.value_counts().reset_index()#.duplicated(subset=['clusters'])#.iloc[0,0]
   
    return attmpt_pca_step        

##selection of rows related to most abundant clusters (PCA option)

def get_selection_PCA(attmpt_pca_step:pd.DataFrame)->pd.DataFrame:
    
    cluster_stat = attmpt_pca_step.clusters.value_counts().to_dict()
    cluster_count = list(cluster_stat.values())
    cluster_non_equal = cluster_count[0]>cluster_count[1]
    if cluster_non_equal:
        target_cluster = list(cluster_stat.keys())[0]
        mask = attmpt_pca_step.clusters == target_cluster
        data_grol = attmpt_pca_step.loc[mask]
    else: 
        print('equal cluster')
        data_grol = pd.DataFrame()
    return data_grol 
