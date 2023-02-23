import sys
import pickle
import numpy as np
import math
import time
import pandas as pd
import dask as dd
from dask.dataframe import from_pandas
import vaex as vaex



# write list to binary file
def write_pickle(file, filename, epoch_num):
    # store list in binary file so 'wb' mode
    with open(filename + str(epoch_num), 'wb') as fp:
        pickle.dump(file, fp)
        print('Done writing list into a binary file')

# Read list to memory
def read_pickle(filename, epoch_num):
    # for reading also binary mode is important
    with open(filename + str(epoch_num), 'rb') as fp:
        file = pickle.load(fp)
        return file

def create_checkpoint(dictionary):
    return 0



def compute_buckets(dictionary, list1to1, list2to10, list11to100, list101to1000, list1001):

    #print("Computing buckets df (seconds): " + str(time.perf_counter()))
    df = pd.DataFrame(list(dictionary.items()), columns=['Cache Lines', 'Appearances'])
    #ddf = from_pandas(df, npartitions=2)
    #vdf = vaex.from_pandas(df)


    #vdf1to1 = vdf[vdf.Appearances == 1]
    #ddf1to1 = ddf.loc[(ddf['Appearances'] == 1)].compute(scheduler='threads')
    df1to1 = df.loc[df['Appearances'] == 1]
    #bucket1to1 = df1to1.set_index('Cache Lines')['Appearances'].to_dict()
    #bucket1to1 = dict((k, v) for k, v in dictionary.items() if v == 1)
    #avg = np.mean(list(bucket1to1.values()))
    avg = df1to1['Appearances'].mean()
    if math.isnan(avg):
        avg = 0
    list1to1.append((avg, len(df1to1.index)))
   #print("df1to1 calc" + str(time.perf_counter()))

    #vdf2to10 = vdf[(vdf.Appearances > 1) & (vdf.Appearances < 11)]
    #ddf2to10 = ddf.loc[(ddf['Appearances'] > 1) & (ddf['Appearances'] < 11)].compute(scheduler='threads')
    df2to10 = df.loc[(df['Appearances'] > 1) & (df['Appearances'] < 11)]
    #bucket2to10 = df2to10.set_index('Cache Lines')['Appearances'].to_dict()
    #bucket2to10 = dict((k, v) for k, v in dictionary.items() if 1 < v < 11)
    #avg = np.mean(list(bucket2to10.values()))
    avg = df2to10['Appearances'].mean()
    if math.isnan(avg):
        avg = 0
    list2to10.append((avg, len(df2to10.index)))
    #print("df2to10 calc" + str(time.perf_counter()))

    #vdf11to100 = vdf[(vdf.Appearances > 10) & (vdf.Appearances < 101)]
    #ddf11to100 = ddf.loc[(ddf['Appearances'] > 10) & (ddf['Appearances'] < 101)].compute(scheduler='threads')
    df11to100 = df.loc[(df['Appearances'] > 10) & (df['Appearances'] < 101)]
    #bucket11to100 = df11to100.set_index('Cache Lines')['Appearances'].to_dict()
    #bucket11to100 = dict((k, v) for k, v in dictionary.items() if 10 < v < 101)
    #avg = np.mean(list(bucket11to100.values()))
    avg = df11to100['Appearances'].mean()
    if math.isnan(avg):
        avg = 0
    list11to100.append((avg, len(df11to100.index)))
    #print("df11to100 calc" + str(time.perf_counter()))

    #vdf101to1000 = vdf[(vdf.Appearances > 100) & (vdf.Appearances < 1001)]
    #ddf101to1000 = ddf.loc[(ddf['Appearances'] > 100) & (ddf['Appearances'] < 1001)].compute(scheduler='threads')
    df101to1000 = df.loc[(df['Appearances'] > 100) & (df['Appearances'] < 1001)]
    #bucket101to1000 = df101to1000.set_index('Cache Lines')['Appearances'].to_dict()
    #bucket101to1000 = dict((k, v) for k, v in dictionary.items() if 100 < v < 1001)
    #avg = np.mean(list(bucket101to1000.values()))
    avg = df101to1000['Appearances'].mean()
    if math.isnan(avg):
        avg = 0
    list101to1000.append((avg, len(df101to1000.index)))
    #print("df101to1000 calc" + str(time.perf_counter()))


    #vdf1001 = vdf[(vdf.Appearances > 1000)]
    #ddf1001 = ddf.loc[ddf['Appearances'] > 1000].compute(scheduler='threads')
    df1001 = df.loc[df['Appearances'] > 1000]
    #bucket1001 = df1001.set_index('Cache Lines')['Appearances'].to_dict()
    #bucket1001 = dict((k, v) for k, v in dictionary.items() if v > 1000)
    #avg = np.mean(list(bucket1001.values()))
    avg = df1001['Appearances'].mean()
    if math.isnan(avg):
        avg = 0
    list1001.append((avg, len(df1001.index)))
    #print("df1001 calc" + str(time.perf_counter()))



def compute_df(list2to10, list11to100, list101to1000, list1001, cls_processed, compression_ratio_array):
    df2to10 = pd.DataFrame.from_records(list2to10, columns=['Average', 'CL Appearances'])
    df2to10['Buffer Type'] = '2to10 Buffer'
    df2to10 = df2to10.set_index('Buffer Type', append=True).unstack('Buffer Type')

    df11to100 = pd.DataFrame.from_records(list11to100, columns=['Average', 'CL Appearances'])
    df11to100['Buffer Type'] = '11to100 Buffer'
    df11to100 = df11to100.set_index('Buffer Type', append=True).unstack('Buffer Type')

    df101to1000 = pd.DataFrame.from_records(list101to1000, columns=['Average', 'CL Appearances'])
    df101to1000['Buffer Type'] = '101to1000 Buffer'
    df101to1000 = df101to1000.set_index('Buffer Type', append=True).unstack('Buffer Type')

    df1001 = pd.DataFrame.from_records(list1001, columns=['Average', 'CL Appearances'])
    df1001['Buffer Type'] = '1001+ Buffer'
    df1001 = df1001.set_index('Buffer Type', append=True).unstack('Buffer Type')

    df = pd.concat([df2to10, df11to100, df101to1000, df1001], axis=1)
    df['CLs Processed'] = np.array(cls_processed).tolist()
    df['Compression Ratio'] = np.array(compression_ratio_array).tolist()
    return df



def calculate_sanity(list1to1, list2to10, list11to100, list101to1000, list1001, stop_index, cl_sum):
    ##calculate sanity
    sanity = list2to10[stop_index][1] * list2to10[stop_index][0] \
             + list11to100[stop_index][1] * list11to100[stop_index][0] \
             + list101to1000[stop_index][1] * list101to1000[stop_index][0] + \
             list1001[stop_index][1] * list1001[stop_index][0]
    singleton = list1to1[stop_index][1] * list1to1[stop_index][0]



    if math.floor(sanity + singleton) == cl_sum or math.ceil(sanity + singleton) == cl_sum:
        print("sanity check")
        return sanity
    else:
        print("bad sum")
    print("epoch number:" + str(stop_index))
    print("epoch time calculation (seconds):" + str(time.perf_counter()))


