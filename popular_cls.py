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


class Epoch:
    '''
    Holds deltas (kern and user) for a given epoch
    '''

    def __init__(self, raw_epoch_data=None):
        # epoch raw data format (diff-count, [(offset, old-CL-bytes, new-CL-bytes)]
        if raw_epoch_data != None:
            self.number = raw_epoch_data[0]
            self.user_cl_changes = raw_epoch_data[2]
            self.kern_cl_changes = raw_epoch_data[4]

            assert (len(raw_epoch_data[2]) == raw_epoch_data[1])
            assert (len(raw_epoch_data[4]) == raw_epoch_data[3])

    def copy(self):
        '''
        Create deep copy
        '''
        return copy.deepcopy(self)

    def _change_list_to_page_writes(self, change_list, page_size=4096):
        shift = int(math.log(page_size, 2))
        page_writes = {}
        for c in change_list:
            key = c[0] >> shift;
            if not key in page_writes:
                page_writes[key] = 1
            else:
                page_writes[key] = page_writes[key] + 1
            assert page_writes[key] <= 64
        return page_writes

    def _change_list_to_write_density(self, change_list, page_size=4096):
        '''
        Take a change list and merge to derive page write densities
        '''
        assert (type(change_list) == list)

        result = {}
        # coalesce to pages
        page_writes = self._change_list_to_page_writes(change_list, page_size)

        density_list = []
        for c in page_writes.values():
            assert type(c) == int
            assert c <= (page_size / 64)
            density_list.append(c / (page_size / 64))
        result["clwr_count"] = len(change_list)
        result["pagewr_count"] = len(page_writes)
        if len(density_list) > 0:
            result["density_mean"] = np.mean(density_list)
            result["density_max"] = np.max(density_list)
            result["density_min"] = np.min(density_list)
        else:
            result["density_mean"] = 0
            result["density_max"] = 0
            result["density_min"] = 0
        return result

    def merge_later(self, later_epoch):
        '''
        Merge a later epoch (this overrides existing change entries with 
        thos in the provided change data)
        '''
        assert type(later_epoch) == Epoch

        # raw_epoch_data overrides existing
        table = {}
        for i in self.user_cl_changes:
            table[i[0]] = i

        for i in later_epoch.user_cl_changes:
            table[i[0]] = i

        self.user_cl_changes = []
        for entry in table.values():
            self.user_cl_changes.append(entry)

        table.clear()
        for i in self.kern_cl_changes:
            table[i[0]] = i

        for i in later_epoch.kern_cl_changes:
            table[i[0]] = i

        self.kern_cl_changes = []
        for entry in table.values():
            self.kern_cl_changes.append(entry)

    def k_write_density(self, page_size=4096):
        '''
        Summarize kernel page write densities
        '''
        return self._change_list_to_write_density(self.kern_cl_changes, page_size)

    def u_write_density(self, page_size=4096):
        '''
        Summarize user page write densities
        '''
        return self._change_list_to_write_density(self.user_cl_changes, page_size)


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

def plot_popular_cls(filename):
    epochs = load(filename)
    num_of_cls_processed = 0
    bucket2to10 = {}
    epoch_num = 0
    kernel_dictionary = {}
    user_dictionary = {}
    for epoch in epochs:
        epoch_num += 1
        kern_cl_changes = epoch.kern_cl_changes
        user_cl_changes = epoch.user_cl_changes
        count_cls(kernel_dictionary, kern_cl_changes)
        count_cls(user_dictionary, user_cl_changes)
        bucket2to10 = dict((k, v) for k, v in kernel_dictionary.items() if 1 < v < 11)
        with open(f'csvs/epoch{epoch_num}.csv', 'w') as f:
            for key in bucket2to10.keys():
                f.write("%s, %s\n" % (key, bucket2to10[key]))

    kernel_dictionary = dict((k, v) for k, v in kernel_dictionary.items())
    data_items = kernel_dictionary.items()
    data_list = list(data_items)
    kernel_df = pd.DataFrame(data_list,  columns=["CL", "COUNT"])



    user_dictionary = dict((k, v) for k, v in user_dictionary.items() if v >= 100)
    data_items = user_dictionary.items()
    data_list = list(data_items)
    user_df = pd.DataFrame(data_list, columns=["CL", "COUNT"])

    kernel_df.to_csv("kernel_cls", sep='\t')
    user_df.to_csv("user_cls", sep='\t')

    df = pd.concat([user_df, kernel_df], axis=0, ignore_index=True)
    grouped_df = df.groupby(['CL'])['COUNT'].sum().reset_index()

    return grouped_df;





plot_popular_cls('./phoronix-ebizzy.deltas')


#plot_density('./phoronix-ebizzy.deltas', "Cache Lines")
