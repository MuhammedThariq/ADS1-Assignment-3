# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 12:31:00 2023

@author: HP
"""

#importing required modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import itertools as iter
from sklearn import cluster


def data_clean(dataframe):
    """This function cleans the dataframe by converting it into a numerical 
    type and fill NaN values with median and returns the cleaned data"""
    #Storing columns, country name and country code into dataframe data1
    data1 = dataframe[dataframe.columns[dataframe.columns.isin(['Country Name', 
                                                       'Country Code'])]]
    #Dropping the columns that are not required for cleaning
    dataframe = dataframe.drop(['Country Name', 'Country Code'], axis=1)
    #Using a for loop to get each attribute of the dataframe for cleaning
    for value in dataframe.columns.values:
        dataframe[value] = pd.to_numeric(dataframe[value])
        dataframe[value] = dataframe[value].fillna(dataframe[value].median())
    #Concatenating the cleaned dataframe with data1
    result_data = pd.concat([data1, dataframe], axis=1)
    return result_data


def outlier(df):
    """This function takes dataframe as an argument and then removes the 
    outliers from the data and returns the dataframe"""
    #Using a forloop to get each attribute of the dataframe 
    for value in df.columns[2:]:
        q1 = df[value].quantile(0.25) #Getting first quantile
        q3 = df[value].quantile(0.75) #Getting the third quantile
        iqr = q3-q1 #Finding the inner quantile range
        whisker_width = 1.5
        lower_whisker = q1 - (whisker_width*iqr) #finding the lower whisker
        upper_whisker = q3 + (whisker_width*iqr) #Finding the upper whisker
        # Finding the index values of the data that are greater than upper
        #whisker and lower than lower whisker 
        for values in df[value]:
            if (values < lower_whisker) | (values > upper_whisker):
                df[value] = df[value].replace(values, df[value].median())
    return df


def normalise(X):
    """ This function takes dataframe as an argument and then normalise values
    of each column to fall between 0 and 1"""
    #Using for loop to get each columns from the dataframe and preforming
    #min max normalisation on the column values.
    for col in X.columns.values:
        X[col] = pd.to_numeric(X[col])
        for value in X[col]:
            X[col] = X[col].replace(value, (value - min(X[col]))/ 
                                    (max(X[col]) - min(X[col])))
    return X #returns the normalised data


def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f


def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper 


#reading the csv file into a dataframe
df_worldbank = pd.read_csv("API_19_DS2_en_csv_v2_4700503.csv")
print(df_worldbank)
print(df_worldbank.columns)

# creating a new dataframe with the required attributes, from 1990 to 2019
df_worldbank = df_worldbank[['Country Name', 'Country Code', 
                                  'Indicator Name', 'Indicator Code', '1990', 
                                  '1991', '1992', '1993', '1994', '1995',
                                  '1996', '1997', '1998', '1999', '2000', 
                                  '2001', '2002', '2003', '2004',
                                  '2005', '2006', '2007', '2008', '2009', 
                                  '2010', '2011', '2012', '2013',
                                  '2014', '2015', '2016', '2017', '2018', 
                                  '2019']]
# Dropping the rows of the attribute Country name equals to World
print(df_worldbank.loc[df_worldbank['Country Name']=='World'])
df_worldbank = df_worldbank.drop(df_worldbank.index[19684:19760])

indicator_code = ['EN.ATM.NOXE.KT.CE', 'AG.YLD.CREL.KG', 'EG.ELC.RNWX.KH', 
                  'EG.FEC.RNEW.ZS']

#creating a data frame which contains values only for the nitrous oxide 
#emission indicator.
df_no2 = df_worldbank.groupby('Indicator Code').get_group ('EN.ATM.NOXE.KT.CE'
                                                           ).reset_index()
df_no2 = df_no2.drop(['Indicator Code','Indicator Name', 'index'], axis=1)
#calling data clean function to replace NaN values with median
df_no2 = data_clean(df_no2)
#calling the outlier function to replace outliers with median
df_no2 = outlier(df_no2)
#creating a new data frame to store country  name and country code
df_wb = df_no2.iloc[:, 0:2].copy()
#finding the average of the nitrous oxide emission from the year 1990
# to 2019 and creating a new column with the indicator code as column name
# in df_wb dataframe 
df_wb['EN.ATM.NOXE.KT.CE'] = df_no2.iloc[:, 2:].mean(axis=1)

#creating a data frame which contains values only for the  
#cereal yield.
df_yield = df_worldbank.groupby('Indicator Code').get_group('AG.YLD.CREL.KG'
                                                            ).reset_index()
df_yield = df_yield.drop(['Indicator Code','Indicator Name', 'index'], axis=1)
#calling data clean function to replace NaN values with median
df_yield = data_clean(df_yield)
#calling the outlier function to replace outliers with median
df_yield = outlier(df_yield)
#finding the average of the cereal yield from the year 1990
# to 2019 and creating a new column with the indicator code as column name
# in df_wb dataframe
df_wb['AG.YLD.CREL.KG'] = df_yield.iloc[:, 2:].mean(axis=1)

#creating a data frame which contains values only for the  renewable
#electricity production
df_elec = df_worldbank.groupby('Indicator Code').get_group('EG.ELC.RNWX.KH'
                                                           ).reset_index()
df_elec = df_elec.drop(['Indicator Code','Indicator Name', 'index'], axis=1)
#calling data clean function to replace NaN values with median
df_elec  = data_clean(df_elec)
#calling the outlier function to replace outliers with median
df_elec  = outlier(df_elec)
#finding the average of the renewable electricity production from the year 1990
# to 2019 and creating a new column with the indicator code as column name
# in df_wb dataframe
df_wb['EG.ELC.RNWX.KH'] = df_elec.iloc[:, 2:].mean(axis=1)


#creating a data frame which contains values only for the  renewable energy
#consumption
df_renew = df_worldbank.groupby('Indicator Code').get_group('EG.FEC.RNEW.ZS'
                                                            ).reset_index()
df_renew = df_renew.drop(['Indicator Code','Indicator Name', 'index'], axis=1)
#calling data clean function to replace NaN values with median
df_renew  = data_clean(df_renew)
#calling the outlier function to replace outliers with median
df_renew  = outlier(df_renew)
#finding the average of the renewable energy consumption from the year 1990
# to 2019 and creating a new column with the indicator code as column name
# in df_wb dataframe
df_wb['EG.FEC.RNEW.ZS'] = df_renew.iloc[:, 2:].mean(axis=1)

print(df_wb)

#creating a new dataframe after dropping the nominal attribute for clustering
df_wb_norm = df_wb.drop(['Country Name','Country Code'],axis=1)
#calling the normalise function to normalise the attribute values
df_wb_norm = normalise(df_wb_norm)
print(df_wb_norm)

#performing an elbow test to find optimal number of clusters
n = [1,2,3,4,5,6,7,8,9]
SSE = []
for value in n:
    kmeans = cluster.KMeans(n_clusters=value)
    kmeans.fit(df_wb_norm)
    SSE.append(kmeans.inertia_)
plt.figure(dpi=144)
plt.plot(n, SSE)  
plt.show()  

#fitting the data into a kmeans algorithm with number of clusters
#equal to 3
kmeans = cluster.KMeans(n_clusters=3, max_iter=100, random_state=0)
labels = kmeans.fit_predict(df_wb_norm)
labels = labels.reshape(-1,1)
#creating a dataframe labels to store the cluster ID
labels = pd.DataFrame(data=labels, columns=['Cluster ID'])
#concatenating the normalised dataframe with the labels
df_result = pd.concat((df_wb_norm, labels), axis=1)
#concatenating the actual data frame with labels
df_wb = pd.concat((df_wb, labels), axis=1)
print(df_result)
print(df_wb)

#Creating a new dataframe to store the average values of each cluster
cluster_info = []
for value in df_result['Cluster ID'].unique():
    cluster_info.append(df_result.loc[df_result['Cluster ID']==value
                                      ].mean(axis=0))
cluster_info = pd.DataFrame(data=cluster_info)    
cluster_info = cluster_info.sort_values(by=['Cluster ID']
                                        ).reset_index().drop('index', axis=1)
print(cluster_info)

print(df_wb.loc[df_wb['Cluster ID']==0, ['Country Name']])
print(df_wb.loc[df_wb['Cluster ID']==1, ['Country Name']].values)
print()
print(df_wb.loc[df_wb['Cluster ID']==2, ['Country Name']].values)

#Creating a dataframe to perform the curve fitting
df_no2_fit = df_no2.loc[(df_no2['Country Name']=='Bangladesh')|
                        (df_no2['Country Name']=='India')| 
                        (df_no2['Country Name']=='China')]
df_no2_fit = df_no2_fit.transpose()
df_no2_fit = df_no2_fit.drop(df_no2_fit.index[0:3]).reset_index().to_numpy()
df_no2_fit = pd.DataFrame(data = df_no2_fit, columns=['Year','Bangladesh', 
                                                   'China', 'India'])
#setting the year attribute to numeric type for fitting
df_no2_fit['Year'] = pd.to_numeric(df_no2_fit['Year'])
param, covar = opt.curve_fit(logistic, df_no2_fit["Year"], 
                             df_no2_fit["Bangladesh"], p0=(3e12, 0.03, 2000.0))
sigma = np.sqrt(np.diag(covar))
print("parameters:", param)
print("std. dev.", sigma)
df_no2_fit["fit"] = logistic(df_no2_fit["Year"], *param)
df_no2_fit.plot("Year", ["Bangladesh", "fit"])
plt.show()

#Plotting a bar blot to show the average nitrous oxide emission 
#of each cluster
plt.figure(dpi=144)
x_axis = np.arange(len(cluster_info['Cluster ID']))
cluster = ['Worst', 'Intermediate', 'Least']
plt.bar(x_axis, cluster_info['EN.ATM.NOXE.KT.CE'])
plt.xticks(x_axis, cluster)
plt.title("Average NO2 emission", fontsize=15)
plt.ylabel("NO2 emission")
plt.show()

#Plotting the scatter plot to show the partitioning of the data points
#into 3 clusters
plt.figure(dpi=144, figsize=(8,5))
plt.scatter(df_result.loc[df_result['Cluster ID']==0, ['EN.ATM.NOXE.KT.CE']], 
                          df_result.loc[df_result['Cluster ID']==0, 
                                        ['EG.FEC.RNEW.ZS']], label="Worst")
plt.scatter(df_result.loc[df_result['Cluster ID']==1, ['EN.ATM.NOXE.KT.CE']], 
                          df_result.loc[df_result['Cluster ID']==1, 
                                        ['EG.FEC.RNEW.ZS']], 
                          label="Intermediate")
plt.scatter(df_result.loc[df_result['Cluster ID']==2, ['EN.ATM.NOXE.KT.CE']], 
                          df_result.loc[df_result['Cluster ID']==2, 
                                        ['EG.FEC.RNEW.ZS']], label="Least")
plt.xlabel("NO2 emission", fontsize=12)
plt.ylabel('Renewable energy consumption', fontsize=12)
plt.title("NO2 emission vs renewable energy consumption", fontsize=15)
plt.legend(loc='upper right')
plt.show()

#Plotting a Bar Plot to show the NO2 emmission of each country over the years
plt.figure(dpi=144,figsize=(10,7))
#filtering the dataframe df_no2 to get the data of specific years
df_no2_new = df_no2.loc[(df_no2['Country Name']=='Bangladesh')|
                        (df_no2['Country Name']=='India')| 
                        (df_no2['Country Name']=='China')]
x_axis = np.arange(len(df_no2_new['Country Name']))
plt.bar(x_axis - 0.3,df_no2_new['1990'], width=0.1, label='1990')
plt.bar(x_axis - 0.2,df_no2_new['1995'], width=0.1, label='1995')
plt.bar(x_axis - 0.1,df_no2_new['2000'], width=0.1, label='2000')
plt.bar(x_axis - 0.0,df_no2_new['2005'], width=0.1, label='2005')
plt.bar(x_axis + 0.1,df_no2_new['2010'], width=0.1, 
        label='2010')
plt.bar(x_axis + 0.2,df_no2_new['2015'], width=0.1, label='2015')
plt.bar(x_axis + 0.3,df_no2_new['2019'], width=0.1, 
        label='2019')
plt.xticks(x_axis, df_no2_new['Country Name'], fontsize=13)
plt.legend()
plt.xlabel("Countries" ,fontsize=13)
plt.ylabel('thousand metric tons of CO2 equivalent', fontsize=13)
plt.title("Nitrous Oxide Emmission", fontsize=15)
plt.show()

#Plotting a line plot to show the trend in the cereal yield
plt.figure(dpi=144, figsize=(8, 5))
#creating a new dataframe for cereal yield with selected countries
df_yield_new = df_yield.loc[(df_no2['Country Name']=='Bangladesh')|
                        (df_no2['Country Name']=='India')| 
                        (df_no2['Country Name']=='China')]
#transposing the data frame to get the years as a single column
df_yield_new = df_yield_new.transpose()
df_yield_new = df_yield_new.drop(df_yield_new.index[0:3]
                                 ).reset_index().to_numpy()
df_yield_new = pd.DataFrame(data = df_yield_new, columns=['Year','Bangladesh', 
                                                         'China', 'India'])
df_yield_rolling = df_yield_new.rolling(window=2).mean()
print(df_yield_new)
for value in df_yield_new.columns[1:].values:
    plt.plot(df_yield_rolling['Year'], df_yield_rolling[value], label = value)
    plt.legend(loc='best')
    plt.xlim(1990,2020)
    plt.xlabel("Year", fontsize=13)
    plt.ylabel("kg/hectare", fontsize=13)
    plt.title("Cereal Yield", fontsize=15)
plt.show()

#Plotting a scatter plot to show the relation between nitrous oxide emission
#and cereal yield
plt.figure(dpi=144, figsize=(8,5))
#Transposing the dataframe to get the Years into a single column
df_no2_new = df_no2_new.transpose()
df_no2_new = df_no2_new.drop(df_no2_new.index[0:3]).reset_index().to_numpy()
df_no2_new = pd.DataFrame(data = df_no2_new, columns=['Year','Bangladesh', 
                                                   'China', 'India'])
plt.scatter(x=df_yield_new['Bangladesh'], y=df_no2_new['Bangladesh'], 
            label="Bangladesh")
plt.legend()
plt.xlabel("Cereal Yield", fontsize=13)
plt.ylabel("Nitrous Oxide emmission",  fontsize=13)
plt.title("Relation between cereal yield and NO2 emission",  fontsize=15)
plt.show()

#Creating a scatter plot to show the relation of nitrous oxide emission
#and renewable energy consumption
plt.figure(dpi=144, figsize=(8,5))
#creating a new dataframe for renewable energy consumption with
#the selected countries
df_renew_new = df_renew.loc[(df_no2['Country Name']=='Pakistan')|
                        (df_no2['Country Name']=='India')| 
                        (df_no2['Country Name']=='China')]
#transposing the dataframe to get the years into a single column
df_renew_new = df_renew_new.transpose()
df_renew_new = df_renew_new.drop(df_renew_new.index[0:3]
                                 ).reset_index().to_numpy()
df_renew_new = pd.DataFrame(data = df_renew_new, columns=['Year','Bangladesh',
                                                          'China', 'India'])
plt.scatter(x=df_renew_new['Bangladesh'], 
            y=df_no2_new['Bangladesh'], label="Bangladesh")
plt.legend()
plt.xlabel("Renewable energy consumption", fontsize=13)
plt.ylabel("Nitrous Oxide emmission", fontsize=13)
plt.title("Relation between Renewable energy consumption and NO2 emission", 
          fontsize=15)
plt.show()

#Creating a line plot to show the trend in renewable energy consumption
#over the years
plt.figure(dpi=144, figsize=(8,5))
df_renew_rolling = df_renew_new.rolling(window=2).mean()
for value in df_renew_new.columns[1:].values:
    plt.plot(df_renew_rolling['Year'], df_renew_rolling[value], label = value)
    plt.legend(loc='best')
    plt.xlim(1990,2020)
    plt.xlabel("Year", fontsize=13)
    plt.ylabel("Percentage of consumption", fontsize=13)
    plt.title("Renewable energy consumption", fontsize=15)
plt.show()

#Creating a line plot to show the trend in nitrous oxide emission
#in Bangladesh over the years
plt.figure(dpi=144)
df_no2_new['Year'] = pd.to_datetime(df_no2_new['Year'])
plt.plot(df_no2_new['Year'], df_no2_new['Bangladesh'])
plt.xlabel('Years')
plt.ylabel('thousand metric tons of CO2 equivalent')
plt.title('NO2 emission')
plt.show()

#Creating a line plot to show the forecasting of the nitrous oxide emission
# in Bangladesh upto the year 2031
plt.figure(dpi=144, figsize=(8,5))
year = np.arange(1990, 2031)
forecast = logistic(year, *param)
plt.plot(df_no2_fit["Year"], df_no2_fit["Bangladesh"], label="Bangladesh")
plt.plot(year, forecast, label="forecast")
plt.xlabel("year", fontsize=13)
plt.ylabel("thousand metric tons of CO2 equivalent", fontsize=13)
plt.title('NO2 emission forecasting', fontsize=15)
plt.legend()
plt.show()

#Creating a line plot to show the confidence ranges in forecasting
plt.figure(dpi=144)
#calling the err_ranges function to get the lower and upper limit of 
#the confidence range
low, up = err_ranges(year, logistic, param, sigma)
plt.plot(df_no2_fit["Year"], df_no2_fit["Bangladesh"], label="Bangladesh")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("No2 emission")
plt.legend()
plt.show()
