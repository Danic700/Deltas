import sys
import time

import my_rle
import lz4.frame
import zstd
import zlib
import snappy
import gzip

import RLE


def benchmark_no_compression(epochs, calculate_xor):
    all_bytes = bytearray()
    start_time_bench = time.perf_counter()
    diff_user_cl = 0
    diff_kern_cl = 0
    for epoch in epochs:
        for kern_cl in epoch.kern_cl_changes:
            cl = kern_cl[2]
            if len(kern_cl[1]) != len(kern_cl[2]):
                diff_kern_cl = diff_kern_cl + 1
            if calculate_xor:
                cl = xor(kern_cl[1], kern_cl[2])
            all_bytes.extend(cl)
        for user_cl in epoch.user_cl_changes:
            cl = user_cl[2]
            if len(user_cl[1]) != len(user_cl[2]):
                diff_user_cl = diff_user_cl + 1
            if calculate_xor:
                cl = xor(user_cl[1], user_cl[2])
            all_bytes.extend(cl)

    print(f"{diff_kern_cl} --  {diff_user_cl}")
    finish_time_bench = time.perf_counter()
    print(f"No Compression finished in {finish_time_bench - start_time_bench} seconds")

    total_size = (sys.getsizeof(all_bytes)) / (1024 * 1024)  # Get Size in MB
    if calculate_xor:
        return [total_size, finish_time_bench - start_time_bench, "NO COMPRESSION_XOR", all_bytes.count(0)], all_bytes
    else:
        return [total_size, finish_time_bench - start_time_bench, "NO COMPRESSION", all_bytes.count(0)], all_bytes



def benchmark_rle(epochs):
    kern_bytes = bytearray()
    user_bytes = bytearray()
    start_time_bench = time.perf_counter()
    for epoch in epochs:
        for kern_cl in epoch.kern_cl_changes:
            xor_value = xor(kern_cl[1], kern_cl[2])
            kern_bytes.extend(xor_value)
        for user_cl in epoch.user_cl_changes:
            xor_value = xor(user_cl[1], user_cl[2])
            user_bytes.extend(xor_value)

    compressed_kern_cl_values, compressed_kern_cl_appearances = RLE.encode(kern_bytes)
    compressed_user_cl_values, compressed_user_cl_appearances = RLE.encode(user_bytes)

    compressed_kern_cl_values = bytes(compressed_kern_cl_values)
    compressed_user_cl_values = bytes(compressed_user_cl_values)

    finish_time_bench = time.perf_counter()
    print(f"RLE Compression finished in {finish_time_bench - start_time_bench} seconds")

    total_size = (sys.getsizeof(compressed_kern_cl_values) + sys.getsizeof(compressed_user_cl_values)) / (
                1024 * 1024)  # Get Size in MB
    return [total_size, finish_time_bench - start_time_bench, "RLE"]


def benchmark(all_bytes, compression_algo):
    library = globals()[compression_algo]
    start_time_bench = time.perf_counter()
    if compression_algo == 'lz4':
        compressed_bytes = library.frame.compress(bytes(all_bytes))
    else:
        compressed_bytes = library.compress(bytes(all_bytes))
    finish_time_bench = time.perf_counter()
    print(f"{compression_algo} Compression finished in {finish_time_bench - start_time_bench} seconds")
    total_size = (sys.getsizeof(compressed_bytes)) / (1024 * 1024)  # Get Size in MB
    return [total_size, finish_time_bench - start_time_bench, compression_algo]


def xor(byte1, byte2):
    byte_array_1 = bytearray(byte1)
    byte_array_2 = bytearray(byte2)

    # Perform XOR operation on each pair of bytes
    result = bytearray()
    for byte1, byte2 in zip(byte_array_1, byte_array_2):
        result.append(byte1 ^ byte2)

    return result


