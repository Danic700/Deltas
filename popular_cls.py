#!/usr/bin/env python
# coding: utf-8

# In[24]:


# !/usr/bin/python3.9
import sys
import pickle
import os.path
import math
from fileinput import filename

import numpy as np
import copy
import pandas as pd
import csv
from collections import Counter

from numpy import NaN


class Epoch:
    '''
    Holds deltas (kern and user) for a given epoch
    '''

    def __init__(self, raw_epoch_data=None):
        # epoch raw data format (diff-count, [(offset, old-CL-bytes, new-CL-bytes)]
        if raw_epoch_data != None:
            self.number = raw_epoch_data[0]
            self.total_kern_changes = raw_epoch_data[3]
            self.user_cl_changes = raw_epoch_data[2]
            self.total_user_cl_changes = raw_epoch_data[1]
            self.kern_cl_changes = raw_epoch_data[4]

            assert (len(raw_epoch_data[2]) == raw_epoch_data[1])
            assert (len(raw_epoch_data[4]) == raw_epoch_data[3])


def load(filename):
    '''
    Load series of epochs from pickled file (output from ex.py)
    '''
    epochs = []
    with open(filename, 'rb') as infile:
        while True:
            try:
                x = pickle.load(infile)
                # print('({}, {},{})'.format(x[0],x[1],x[3]))
                epochs.append(Epoch(x))
            except EOFError:
                break
    print('Loaded {} base epochs'.format(len(epochs)))
    return epochs


def count_cls(dictionary, cl_changes):
    for cl in cl_changes:
        key = cl[2]
        if key in dictionary.keys():
            dictionary[key] += 1
        else:
            dictionary[key] = 1


def split_deltas(filename, symbol_size):
    epochs = load(filename)
    data = epochs[0].kern_cl_changes[0][2]
    info = [data[i:i + symbol_size] for i in range(0, len(data), symbol_size)]
    t = epochs[0].kern_cl_changes[0]
    epochs[0].kern_cl_changes[0] = t + (info,)
    print(info)



def plot_popular_cls(filename):
    epochs = load(filename)
    kern_cls_processed = []
    prev_kern_cls_processed = 0
    list1to1 = []
    list2to10 = []
    list11to100 = []
    list101to1000 = []
    list1001 = []
    kernel_dictionary = {}
    kernel_counter = Counter
    stop_index = 0
    kern_cl_sum = 0
    kern_cls = []
    compression_ratio = 0
    for epoch in epochs:

        kern_cl_changes = epoch.kern_cl_changes
        kern_cl_sum = kern_cl_sum + epoch.total_kern_changes

        count_cls(kernel_dictionary, kern_cl_changes)

        bucket1to1 = dict((k, v) for k, v in kernel_dictionary.items() if v == 1)
        avg = np.mean(list(bucket1to1.values()))
        if math.isnan(avg):
            avg = 0
        list1to1.append((avg, len(bucket1to1)))

        bucket2to10 = dict((k, v) for k, v in kernel_dictionary.items() if 1 < v < 11)
        avg = np.mean(list(bucket2to10.values()))
        if math.isnan(avg):
            avg = 0
        compression_ratio = (np.sum(list(bucket2to10.values())) * avg) / kern_cl_sum
        list2to10.append((avg, len(bucket2to10), compression_ratio))

        bucket11to100 = dict((k, v) for k, v in kernel_dictionary.items() if 10 < v < 101)
        avg = np.mean(list(bucket11to100.values()))
        if math.isnan(avg):
            avg = 0
        compression_ratio = (np.sum(list(bucket11to100.values())) * avg) / kern_cl_sum
        list11to100.append((avg, len(bucket11to100), compression_ratio))

        bucket101to1000 = dict((k, v) for k, v in kernel_dictionary.items() if 100 < v < 1001)
        avg = np.mean(list(bucket101to1000.values()))
        if math.isnan(avg):
            avg = 0
        compression_ratio = (np.sum(list(bucket101to1000.values())) * avg) / kern_cl_sum
        list101to1000.append((avg, len(bucket101to1000), compression_ratio))

        bucket1001 = dict((k, v) for k, v in kernel_dictionary.items() if v > 1000)
        avg = np.mean(list(bucket1001.values()))
        if math.isnan(avg):
            avg = 0
        compression_ratio = (np.sum(list(bucket1001.values())) * avg) / kern_cl_sum
        list1001.append((avg, len(bucket1001), compression_ratio))

        kern_cls_processed.append(len(kern_cl_changes) + prev_kern_cls_processed)
        prev_kern_cls_processed = kern_cls_processed[-1]

        if stop_index == 600:
            break
        else:
            sanity = list2to10[stop_index][1]*list2to10[stop_index][0]\
                     + list11to100[stop_index][1]*list11to100[stop_index][0] \
                     + list101to1000[stop_index][1]*list101to1000[stop_index][0] + \
                     list1001[stop_index][1]*list1001[stop_index][0]
            singleton = list1to1[stop_index][1]*list1to1[stop_index][0]
            if singleton + sanity == kern_cl_sum:
                print("sanity check")
            else:
                print("bad sum")
            stop_index = stop_index + 1


    df2to10 = pd.DataFrame.from_records(list2to10, columns=['Average', 'CL Appearances', 'Compression Ratio'])
    df2to10['Buffer Type'] = '2to10 Buffer'
    df2to10 = df2to10.set_index('Buffer Type', append=True).unstack('Buffer Type')

    df11to100 = pd.DataFrame.from_records(list11to100, columns=['Average', 'CL Appearances', 'Compression Ratio'])
    df11to100['Buffer Type'] = '11to100 Buffer'
    df11to100 = df11to100.set_index('Buffer Type', append=True).unstack('Buffer Type')

    df101to1000 = pd.DataFrame.from_records(list101to1000, columns=['Average', 'CL Appearances', 'Compression Ratio'])
    df101to1000['Buffer Type'] = '101to1000 Buffer'
    df101to1000 = df101to1000.set_index('Buffer Type', append=True).unstack('Buffer Type')

    df1001 = pd.DataFrame.from_records(list1001, columns=['Average', 'CL Appearances', 'Compression Ratio'])
    df1001['Buffer Type'] = '1001+ Buffer'
    df1001 = df1001.set_index('Buffer Type', append=True).unstack('Buffer Type')

    df = pd.concat([df2to10, df11to100, df101to1000, df1001], axis=1)
    df['CLs Processed'] = np.array(kern_cls_processed).tolist()
    df.to_csv('out.csv', index=False)

    # kernel_dictionary = dict((k, v) for k, v in kernel_dictionary.items())
    # data_items = kernel_dictionary.items()
    # data_list = list(data_items)
    # kernel_df = pd.DataFrame(data_list,  columns=["CL", "COUNT"])
    #
    #
    #
    # user_dictionary = dict((k, v) for k, v in user_dictionary.items() if v >= 100)
    # data_items = user_dictionary.items()
    # data_list = list(data_items)
    # user_df = pd.DataFrame(data_list, columns=["CL", "COUNT"])
    #
    # kernel_df.to_csv("kernel_cls", sep='\t')
    # user_df.to_csv("user_cls", sep='\t')
    #
    # df = pd.concat([user_df, kernel_df], axis=0, ignore_index=True)
    # grouped_df = df.groupby(['CL'])['COUNT'].sum().reset_index()
    #
    # return grouped_df;


##split_deltas('./phoronix-ebizzy.deltas', 32)
plot_popular_cls('./phoronix-ebizzy.deltas')
