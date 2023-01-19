# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 12:31:00 2023

@author: HP
"""

#importing required modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sb
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
    index_list = []
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
    #     # Storing the index values into an index list
    #     index_list.extend(ls)
    # #Using sorted and set function to remove the repeated index values and 
    # #sort the index values
    # index_list = sorted(set(index_list))
    # #Dropping the data points from the dataframe whose index values are in the 
    # #index list
    # df = df.drop(index_list)
    return df


def normalise(X):
    """ This function takes dataframe as an argument and then normalise values
    of each column to fall between 0 and 1"""
    #Using for loop to get each columns from the dataframe and preforming
    #min max normalisation on the column values.
    for col in X.columns.values:
        for value in X[col]:
            X[col] = X[col].replace(value, (value - min(X[col]))/ 
                                    (max(X[col]) - min(X[col])))
    return X #returns the normalised data


#reading the csv file into a dataframe
df_worldbank = pd.read_csv("API_19_DS2_en_csv_v2_4700503.csv")
print(df_worldbank)
print(df_worldbank.columns)

# creating a new dataframe with the required 
df_worldbank = df_worldbank[['Country Name', 'Country Code', 
                                  'Indicator Name', 'Indicator Code', '1990', '1991', '1992', '1993', '1994', '1995',
                                  '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
                                  '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                                  '2014', '2015', '2016', '2017', '2018', '2019']]


indicator_code = ['EN.ATM.NOXE.KT.CE', 'AG.YLD.CREL.KG', 'EG.ELC.RNWX.KH', 
                  'EG.FEC.RNEW.ZS']
df_no2 = df_worldbank.groupby('Indicator Code').get_group('EN.ATM.NOXE.KT.CE').reset_index()
df_no2 = df_no2.drop(['Indicator Code','Indicator Name', 'index'], axis=1)
df_no2 = data_clean(df_no2)
df_no2 = outlier(df_no2)
df_wb = df_no2.iloc[:, 0:2].copy()
df_wb['EN.ATM.NOXE.KT.CE'] = df_no2.iloc[:, 2:].mean(axis=1)

df_yield = df_worldbank.groupby('Indicator Code').get_group('AG.YLD.CREL.KG').reset_index()
df_yield = df_yield.drop(['Indicator Code','Indicator Name', 'index'], axis=1)
df_yield = data_clean(df_yield)
df_yield = outlier(df_yield)
df_wb['AG.YLD.CREL.KG'] = df_yield.iloc[:, 2:].mean(axis=1)


df_elec = df_worldbank.groupby('Indicator Code').get_group('EG.ELC.RNWX.KH').reset_index()
df_elec = df_elec.drop(['Indicator Code','Indicator Name', 'index'], axis=1)
df_elec  = data_clean(df_elec)
df_elec  = outlier(df_elec)
df_wb['EG.ELC.RNWX.KH'] = df_elec.iloc[:, 2:].mean(axis=1)



df_renew = df_worldbank.groupby('Indicator Code').get_group('EG.FEC.RNEW.ZS').reset_index()
df_renew = df_renew.drop(['Indicator Code','Indicator Name', 'index'], axis=1)
df_renew  = data_clean(df_renew)
df_renew  = outlier(df_renew)
df_wb['EG.FEC.RNEW.ZS'] = df_renew.iloc[:, 2:].mean(axis=1)

print(df_wb)


# df_wb_ = df_wb.drop(['Indicator Code','Indicator Name'], axis=1).reset_index()
# df_wb = df_wb.drop('index', axis=1)

# for value in indicator_code[1:]:
#     df = df_worldbank_2015.groupby('Indicator Code').get_group(value)
#     df_2015 = pd.concat([df_2015.reset_index(), df['2015'].reset_index()], 
#                         axis=1)
#     df_2015 = df_2015.drop('index', axis=1)    
 
# df_2015 = df_2015.to_numpy()
# df_wb_2015 = pd.DataFrame(data = df_2015, columns=['Country Name', 
#                                                    'Country Code', 
#                                                    'EN.ATM.NOXE.KT.CE', 
#                                                    'AG.YLD.CREL.KG', 
#                                                    'EG.ELC.RNWX.KH', 
#                                                    'EG.FEC.RNEW.ZS'])
# df_wb_2015 = data_clean(df_wb_2015)
# df_wb_2015_ = outlier(df_wb_2015)
# print(df_wb_2015['Country Name'].values)
# # country =['United Kingdom', 'India', 'Bangladesh', 'Brazil', 'China', 'France', 
# #           'United States', 'Switzerland', 'Chile']
# print(df_wb_2015)
df_wb_norm = df_wb.drop(['Country Name','Country Code'],axis=1)
df_wb_norm = normalise(df_wb_norm)
# dataplot = sb.heatmap(df_wb_2015_norm.corr(), annot=True)
print(df_wb_norm)

n = [1,2,3,4,5,6,7,8,9]
SSE = []
for value in n:
    kmeans = cluster.KMeans(n_clusters=value)
    kmeans.fit(df_wb_norm)
    SSE.append(kmeans.inertia_)

plt.figure(dpi=144)
plt.plot(n, SSE)  
plt.show()  

kmeans = cluster.KMeans(n_clusters=3, max_iter=100, random_state=0)
labels = kmeans.fit_predict(df_wb_norm)
labels = labels.reshape(-1,1)
labels = pd.DataFrame(data=labels, columns=['Cluster ID'])
df_result = pd.concat((df_wb_norm, labels), axis=1)
df_wb = pd.concat((df_wb, labels), axis=1)
print(df_result)
print(df_wb.loc[df_wb['Country Name']=='China'])
print(df_wb.loc[df_wb['Cluster ID']==2])

cluster_info = []
for value in df_result['Cluster ID'].unique():
    cluster_info.append(df_result.loc[df_result['Cluster ID']==value].mean(axis=0))
cluster_info = pd.DataFrame(data=cluster_info)    
cluster_info = cluster_info.sort_values(by=['Cluster ID']).reset_index().drop('index', axis=1)
print(cluster_info)



# # df_worldbank_1995 = df_worldbank[['Country Name', 'Country Code', 
# #                                   'Indicator Name', 'Indicator Code', '1995']]


# # indicator_code = ['EN.ATM.NOXE.KT.CE', 'AG.YLD.CREL.KG', 'EG.ELC.RNWX.KH', 
# #                   'EG.FEC.RNEW.ZS']
# # df_1995 = df_worldbank_1995.groupby('Indicator Code').get_group('EN.ATM.NOXE.KT.CE')
# # df_1995 = df_1995.drop(['Indicator Code','Indicator Name'], axis=1).reset_index()
# # df_1995 = df_1995.drop('index', axis=1)
# # for value in indicator_code[1:]:
# #     df = df_worldbank_1995.groupby('Indicator Code').get_group(value)
# #     df_1995 = pd.concat([df_1995.reset_index(), df['1995'].reset_index()], 
# #                         axis=1)
# #     df_1995 = df_1995.drop('index', axis=1)    
 
# # df_1995 = df_1995.to_numpy()
# # df_wb_1995 = pd.DataFrame(data = df_1995, columns=['Country Name', 
# #                                                    'Country Code', 
# #                                                    'EN.ATM.NOXE.KT.CE', 
# #                                                    'AG.YLD.CREL.KG', 
# #                                                    'EG.ELC.RNWX.KH', 
# #                                                    'EG.FEC.RNEW.ZS'])
# # df_wb_1995 = data_clean(df_wb_1995)
# # df_wb_1995_ = outlier(df_wb_1995)
# # print(df_wb_1995['Country Name'].values)
# # # country =['United Kingdom', 'India', 'Bangladesh', 'Brazil', 'China', 'France', 
# # #           'United States', 'Switzerland', 'Chile']
# # print(df_wb_1995)
# # df_wb_1995_norm = df_wb_1995.drop(['Country Name','Country Code'],axis=1)
# # df_wb_1995_norm = normalise(df_wb_1995_norm)
# # print(df_wb_1995_norm)

# # kmeans = cluster.KMeans(n_clusters=2, max_iter=100, random_state=0)
# # labels = kmeans.fit_predict(df_wb_1995_norm)
# # labels = labels.reshape(-1,1)
# # labels = pd.DataFrame(data=labels, columns=['Cluster ID'])
# # df_result_1995 = pd.concat((df_wb_1995_norm, labels), axis=1)
# # df_wb_1995 = pd.concat((df_wb_1995, labels), axis=1)
# # print(df_result_1995)
# # print(df_wb_1995.loc[df_wb_1995['Country Name']=='Pakistan'])

# # cluster_info_1995 = []
# # for value in df_result_1995['Cluster ID'].unique():
# #     cluster_info_1995.append(df_result_1995.loc[df_result_1995['Cluster ID']==value].mean(axis=0))
# # cluster_info_1995 = pd.DataFrame(data=cluster_info_1995)    
# # cluster_info_1995 = cluster_info_1995.sort_values(by=['Cluster ID']).reset_index().drop('index', axis=1)
# # print(cluster_info_1995)



plt.figure(dpi=144)
x_axis = np.arange(len(cluster_info['Cluster ID']))
cluster = ['Cluster 0', 'Cluster 1', 'Cluster 3']
plt.bar(x_axis, cluster_info['EN.ATM.NOXE.KT.CE'])
plt.xticks(x_axis, cluster)
plt.title("Average NO2 emission", fontsize=15)
plt.ylabel("NO2 emission")
plt.show()


# plt.figure(dpi=144)
# x_axis = np.arange(len(cluster_info_1995['Cluster ID']))
# cluster = ['Cluster 0', 'Cluster 1']
# plt.bar(x_axis, cluster_info_1995['EN.ATM.NOXE.KT.CE'])
# plt.xticks(x_axis, cluster)
# plt.title("Average NO2 emission 1995", fontsize=15)
# plt.ylabel("NO2 emission")
# plt.show()


plt.figure(dpi=144, figsize=(8,5))
plt.scatter(df_result.loc[df_result['Cluster ID']==0, ['EN.ATM.NOXE.KT.CE']], 
                          df_result.loc[df_result['Cluster ID']==0, 
                                        ['EG.FEC.RNEW.ZS']], label="Cluster 0")
plt.scatter(df_result.loc[df_result['Cluster ID']==1, ['EN.ATM.NOXE.KT.CE']], 
                          df_result.loc[df_result['Cluster ID']==1, 
                                        ['EG.FEC.RNEW.ZS']], label="Cluster 1")
plt.scatter(df_result.loc[df_result['Cluster ID']==2, ['EN.ATM.NOXE.KT.CE']], 
                          df_result.loc[df_result['Cluster ID']==2, 
                                        ['EG.FEC.RNEW.ZS']], label="Cluster 2")
plt.xlabel("NO2 emission", fontsize=12)
plt.ylabel('Renewable energy consumption', fontsize=12)
plt.title("NO2 emission vs renewable energy consumption", fontsize=15)
plt.legend(loc='upper right')
plt.show()

# plt.figure(dpi=144, figsize=(8,5))
# plt.scatter(df_result_1995.loc[df_result['Cluster ID']==0, ['EN.ATM.NOXE.KT.CE']], 
#                           df_result_1995.loc[df_result_1995['Cluster ID']==0, 
#                                         ['EG.FEC.RNEW.ZS']], label="Cluster 0")
# plt.scatter(df_result_1995.loc[df_result_1995['Cluster ID']==1, ['EN.ATM.NOXE.KT.CE']], 
#                           df_result_1995.loc[df_result_1995['Cluster ID']==1, 
#                                         ['EG.FEC.RNEW.ZS']], label="Cluster 1")
# plt.xlabel("NO2 emission", fontsize=12)
# plt.ylabel('Renewable energy consumption', fontsize=12)
# plt.title("NO2 emission vs renewable energy consumption 1995", fontsize=15)
# plt.legend(loc='upper right')
# plt.show()

