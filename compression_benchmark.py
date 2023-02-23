#!/usr/bin/env python
# coding: utf-8

# In[24]:


# !/usr/bin/python3.9
import sys
import pickle
import time

import compression_util
import plot


import numpy as np

import pandas as pd


MAX_SYMBOL_SIZE = 64
KERN_CL_CHANGES = 4
USER_CL_CHANGES = 2
running_serial_number = 0
cl_dictionary = {}


class Epoch:
    '''
    Holds deltas (kern and user) for a given epoch
    '''

    def __init__(self, raw_epoch_data=None):
        # epoch raw data format (diff-count, [(offset, old-CL-bytes, new-CL-bytes, split-CL-bytes)]
        if raw_epoch_data is not None:
            self.number = raw_epoch_data[0]
            self.total_user_cl_changes = raw_epoch_data[1]
            self.user_cl_changes = raw_epoch_data[2]
            self.total_kern_changes = raw_epoch_data[3]
            self.kern_cl_changes = raw_epoch_data[4]

            assert (len(raw_epoch_data[2]) == raw_epoch_data[
                1])  ## make sure the size of changes array is equal to number of total changes
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
                # epochs.append(x)
            except EOFError:
                break

    print('Loaded {} base epochs'.format(len(epochs)))
    return epochs



if __name__ == "__main__":
    start_time = time.perf_counter()
    filename = sys.argv[1]  ## './phoronix-nettle-aes.deltas'
    symbol_size = int(sys.argv[2])  ###sys.argv[1]   -- need to get the argument
    print("filename is " + filename)
    print("symbol_size is " + str(symbol_size))

    epochs = load(filename)

    no_compression_result = compression_util.benchmark_no_compression(epochs, filename)
    zlib_result = compression_util.benchmark_zlib(epochs, filename)
    zstd_result = compression_util.benchmark_zstd(epochs, filename)
    lz4_result = compression_util.benchmark_lz4(epochs, filename)
    gzip_result = compression_util.benchmark_gzip(epochs, filename)
    snappy_result = compression_util.benchmark_snappy(epochs, filename)
    rle_result = compression_util.benchmark_rle(epochs, filename)


    df = pd.DataFrame([no_compression_result,
                       zlib_result,
                       zstd_result,
                       lz4_result,
                       gzip_result,
                       snappy_result,
                       rle_result])
    df_tilted = df.stack().unstack(0)

    df.to_csv(f"{filename}-compression-benchmark.csv")
    df_tilted.to_csv(f"{filename}tilted-compression-benchmark.csv")


    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time - start_time} seconds")
