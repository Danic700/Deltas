#!/usr/bin/env python
# coding: utf-8

# In[24]:


# !/usr/bin/python3.9
import sys
import pickle
import os
import math
import time
import multiprocessing
import utils
from itertools import islice
import zlib

import zstd

import numpy as np
import pandas as pd
from collections import Counter

from numpy import NaN

MAX_SYMBOL_SIZE = 64
KERN_CL_CHANGES = 4
USER_CL_CHANGES = 2

class Epoch:
    '''
    Holds deltas (kern and user) for a given epoch
    '''

    def __init__(self, raw_epoch_data=None):
        # epoch raw data format (diff-count, [(offset, old-CL-bytes, new-CL-bytes)]
        if raw_epoch_data is not None:
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
                epochs.append(x)
            except EOFError:
                break
    outfile = 'test.deltas'
    compressed_epochs = []
    with open(outfile, 'wb') as out:
        for epoch in epochs:
            compressed_kern_cl_changes = []
            for kern_cl in epoch[KERN_CL_CHANGES]:
                compressed_kern_cl = (kern_cl[0], zstd.compress(kern_cl[1], 9), zstd.compress(kern_cl[2], 9))
                compressed_kern_cl_changes.append(compressed_kern_cl)
            compressed_user_cl_changes = []
            for user_cl in epoch[USER_CL_CHANGES]:
                compressed_user_cl = (user_cl[0], zstd.compress(user_cl[1], 9), zstd.compress(user_cl[2], 9))
                compressed_user_cl_changes.append(compressed_user_cl)

            compressed_epoch = (epoch[0],epoch[1], compressed_user_cl_changes, epoch[3], compressed_kern_cl_changes)

            pickle.dump(compressed_epoch, out, pickle.HIGHEST_PROTOCOL)

    epochs2 = []
    with open(outfile, 'rb') as infile:
        while True:
            try:
                x = pickle.load(infile)
                # print('({}, {},{})'.format(x[0],x[1],x[3]))
                epochs2.append(x)
            except EOFError:
                break

    print('Loaded {} base epochs'.format(len(epochs)))
    return epochs



def count_cls(dictionary, cl_changes):
    for cl in cl_changes:
        split_cl_array = cl[3]
        for split_cl in split_cl_array:
            key = split_cl                # take cl from split symbol array
            if key in dictionary.keys():
                dictionary[key] += 1
            else:
                dictionary[key] = 1


def split_deltas(epochs, symbol_size):
    for epoch in epochs:
        kernel_index = 0
        user_index = 0
        kern_cl_changes = epoch.kern_cl_changes
        epoch.total_kern_changes = epoch.total_kern_changes * (MAX_SYMBOL_SIZE // symbol_size) # number of kern_cl_change grows when working with different symbol size
        for cl_tuple in kern_cl_changes:
            kern_bytes = cl_tuple[2]
            if kern_bytes == b'':       # this means the last epoch
                 break
            split_bytes = [kern_bytes[i:i + symbol_size] for i in range(0, len(kern_bytes), symbol_size)]
            epoch.kern_cl_changes[kernel_index] = cl_tuple + (split_bytes,)  # add another member to the tuple
            kernel_index += 1


def split_deltas1(epoch, symbol_size):

    epoch.total_kern_changes = epoch.total_kern_changes * (MAX_SYMBOL_SIZE // symbol_size) # number of cl_change grows when working with different symbol size
    epoch.total_user_cl_changes = epoch.total_user_cl_changes * (MAX_SYMBOL_SIZE // symbol_size)
    split(epoch.kern_cl_changes, symbol_size)
    split(epoch.user_cl_changes, symbol_size)



def split(cl_changes, symbol_size):
    index = 0
    for cl_tuple in cl_changes:
        cl_bytes = cl_tuple[2] # this is the modified cl
        if bytes == b'':     # this means the last epoch
            break
        split_bytes = [cl_bytes[i:i + symbol_size] for i in range(0, len(cl_bytes), symbol_size)]
        cl_changes[index] = cl_tuple + (split_bytes,)
        index += 1


def plot_popular_cls(epochs, symbol_size, start_index):
    kern_cls_processed = []
    prev_kern_cls_processed = 0
    user_cls_processed = []
    prev_user_cls_processed = 0
    kernel_list1to1 = []
    kernel_list2to10 = []
    kernel_list11to100 = []
    kernel_list101to1000 = []
    kernel_list1001 = []
    kernel_dictionary = {}
    user_list1to1 = []
    user_list2to10 = []
    user_list11to100 = []
    user_list101to1000 = []
    user_list1001 = []
    user_dictionary = {}
    stop_index = 0
    kern_cl_sum = 0
    user_cl_sum = 0
    kernel_compression_ratio_array = []
    user_compression_ratio_array = []

    if start_index != 0:
        # and to load the session again:
        1+1

    for epoch in islice(epochs, start_index, 600):

        split_deltas1(epoch, symbol_size)

        print("Processing Epoch#: " + str(epoch.number))

        kern_cl_changes = epoch.kern_cl_changes
        kern_cl_sum = kern_cl_sum + epoch.total_kern_changes   # total
        count_cls(kernel_dictionary, kern_cl_changes)
        print("Counted Kernel CLS  (seconds): " + str(time.perf_counter()))


        user_cl_changes = epoch.user_cl_changes
        user_cl_sum = user_cl_sum + epoch.total_user_cl_changes  # total
        count_cls(user_dictionary, user_cl_changes)
        print("Counted User CLS  (seconds): " + str(time.perf_counter()))


        utils.compute_buckets(kernel_dictionary,  kernel_list1to1, kernel_list2to10, kernel_list11to100, kernel_list101to1000, kernel_list1001)
        print("Computed Kernel Buckets  (seconds): " + str(time.perf_counter()))

        utils.compute_buckets(user_dictionary,  user_list1to1, user_list2to10, user_list11to100, user_list101to1000, user_list1001)
        print("Computed User Buckets  (seconds): " + str(time.perf_counter()))


        kern_cls_processed.append(len(kern_cl_changes) * (MAX_SYMBOL_SIZE // symbol_size) + prev_kern_cls_processed)
        prev_kern_cls_processed = kern_cls_processed[-1]  ##last member of array is the newest previous cls processed


        user_cls_processed.append(len(user_cl_changes) * (MAX_SYMBOL_SIZE // symbol_size) + prev_user_cls_processed)
        prev_user_cls_processed = user_cls_processed[-1]  ##last member of array is the newest previous cls processed

        kern_sanity = utils.calculate_sanity(kernel_list1to1, kernel_list2to10, kernel_list11to100,kernel_list101to1000, kernel_list1001, stop_index, kern_cl_sum)
        kernel_compression_ratio = 1 - kern_sanity / kern_cl_sum
        kernel_compression_ratio_array.append(kernel_compression_ratio)
        user_sanity = utils.calculate_sanity(user_list1to1, user_list2to10, user_list11to100, user_list101to1000, user_list1001, stop_index, user_cl_sum)
        user_compression_ratio = 1 - user_sanity / user_cl_sum
        user_compression_ratio_array.append(user_compression_ratio)



        stop_index = stop_index + 1


    kern_df = utils.compute_df(kernel_list2to10, kernel_list11to100, kernel_list101to1000, kernel_list1001, kern_cls_processed, kernel_compression_ratio_array)
    kern_df.to_csv(f'out{"kernel" + str(symbol_size)}.csv', index=False)
    user_df = utils.compute_df(user_list2to10, user_list11to100, user_list101to1000, user_list1001, user_cls_processed, user_compression_ratio_array)
    user_df.to_csv(f'out{"user" + str(symbol_size)}.csv', index=False)




if __name__ == "__main__":

    start_time = time.perf_counter()
    filename = sys.argv[1] ## './phoronix-nettle-aes.deltas'
    symbol_size = int(sys.argv[2])  ###sys.argv[1]   -- need to get the argument
    start_index = 0
    print("filename is " + filename)
    print("symbol_size is " + str(symbol_size))

    epochs = load(filename)

    plot_popular_cls(epochs, symbol_size, start_index)   ## One process for debug


    # #Creates two processes
    # n = 300
    # split_epochs = [epochs[i * n:(i + 1) * n] for i in range((len(epochs) + n - 1) // n)]
    # p1 = multiprocessing.Process(target=plot_popular_cls, args=(split_epochs[0], symbol_size, ))
    # p2 = multiprocessing.Process(target=plot_popular_cls, args=(split_epochs[1], symbol_size, ))
    #
    # # Start processes
    # p1.start()
    # p2.start()
    #
    # # Wait for processes
    # p1.join()
    # p2.join()


    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time - start_time} seconds")
