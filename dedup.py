#!/usr/bin/env python
# coding: utf-8

# In[24]:


# !/usr/bin/python3.9
import sys
import pickle
import time

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

            assert (len(raw_epoch_data[2]) == raw_epoch_data[1])  ## make sure the size of changes array is equal to number of total changes
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
                #epochs.append(x)
            except EOFError:
                break

    print('Loaded {} base epochs'.format(len(epochs)))
    return epochs





1


def split_deltas1(epoch, symbol_size):

    epoch.total_kern_changes = epoch.total_kern_changes * (MAX_SYMBOL_SIZE // symbol_size) # number of cl_change grows when working with different symbol size
    epoch.total_user_cl_changes = epoch.total_user_cl_changes * (MAX_SYMBOL_SIZE // symbol_size)
    split(epoch.kern_cl_changes, symbol_size)
    split(epoch.user_cl_changes, symbol_size)



def split(cl_changes, symbol_size):
    for index, cl_tuple in enumerate(cl_changes):
        cl_bytes = cl_tuple[2] # this is the modified cl
        if bytes == b'':     # this means the last epoch
            break
        split_bytes = [cl_bytes[i:i + symbol_size] for i in range(0, len(cl_bytes), symbol_size)]
        cl_changes[index] = cl_tuple + (bytes(split_bytes[0]),)


def dump_cl_changes(cl_changes, out):
    size = 0
    for index, cl_tuple in enumerate(cl_changes):
        #print((len(cl_tuple[3])))
        #pickle.dump(cl_tuple[3], out, pickle.HIGHEST_PROTOCOL) ##dump split_cl tuple
        #data1 = msgpack.packb(cl_tuple[3])
        out.write(cl_tuple[3])
        size += len(cl_tuple[3])
    print(str(size))


def dedup(epochs):
        out1 = open(filename+'-no-dedup.bin', 'wb')
        out2 = open(filename+'-dedup.bin', 'wb')
        for epoch in epochs:
            split_deltas1(epoch, symbol_size)

            dump_cl_changes(epoch.kern_cl_changes, out1)
            #dump_cl_changes(epoch.user_cl_changes, out1)

            convert_cls_to_serial_number(epoch.kern_cl_changes)
            #convert_cls_to_serial_number(epoch.user_cl_changes)

            dump_cl_changes(epoch.kern_cl_changes, out2)
            #dump_cl_changes(epoch.user_cl_changes, out2)
            break

        out1.close()
        out2.close()



def convert_cls_to_serial_number(cl_changes):
    global running_serial_number
    global cl_dictionary
    for index, cl in enumerate(cl_changes):
        split_cl = cl[3]
        key = split_cl  # take cl from split symbol array
        if key not in cl_dictionary.keys():
            cl_dictionary[key] = running_serial_number.to_bytes(3, 'big')
            running_serial_number = running_serial_number + 1
        cl_changes[index] = (cl[0], cl[1], cl[2], cl_dictionary[key]) ##do the dedup




if __name__ == "__main__":

    start_time = time.perf_counter()
    filename = sys.argv[1] ## './phoronix-nettle-aes.deltas'
    symbol_size = int(sys.argv[2])  ###sys.argv[1]   -- need to get the argument
    print("filename is " + filename)
    print("symbol_size is " + str(symbol_size))

    epochs = load(filename)

    dedup(epochs)

    my_bytes = b'\x00\x00\x0f'

    with open('my_bytes.bin', 'wb') as f:
        f.write(my_bytes)
        f.write(my_bytes)


    print("the amount of unique cl's is " + str(running_serial_number))
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time - start_time} seconds")
