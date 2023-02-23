import sys
import time

import rle
import lz4.frame
import zstd
import zlib
import snappy
import gzip
import brotli


def benchmark_no_compression(epochs, filename, calculate_xor):
    kern_bytes = bytearray()
    user_bytes = bytearray()
    start_time_bench = time.perf_counter()
    for epoch in epochs:
        for kern_cl in epoch.kern_cl_changes:
            # xor = rle.xor(kern_cl[1], kern_cl[2])
            # compressed_kern_cl = zstd.compress(kern_cl[2], 22)
            kern_bytes.extend(kern_cl[2])
        for user_cl in epoch.user_cl_changes:
            # xor = rle.xor(user_cl[1], user_cl[2])
            # compressed_user_cl = zstd.compress(user_cl[2], 22)
            user_bytes.extend(user_cl[2])

    finish_time_bench = time.perf_counter()
    print(f"No Compression finished in {finish_time_bench - start_time_bench} seconds")

    total_size = (sys.getsizeof(kern_bytes) + sys.getsizeof(user_bytes)) / (1024 * 1024)  # Get Size in MB
    return [total_size, finish_time_bench - start_time_bench, "NO COMPRESSION"]


def benchmark_zstd(epochs, calculate_xor):
    kern_bytes = bytearray()
    user_bytes = bytearray()
    start_time_bench = time.perf_counter()
    for epoch in epochs:
        for kern_cl in epoch.kern_cl_changes:
            cl = kern_cl[2]
            if calculate_xor:
                cl = rle.xor(kern_cl[1], kern_cl[2])
            compressed_kern_cl = zstd.compress(cl, 9)
            kern_bytes.extend(compressed_kern_cl)
        for user_cl in epoch.user_cl_changes:
            cl = user_cl[2]
            if calculate_xor:
                cl = rle.xor(user_cl[1], user_cl[2])
            compressed_user_cl = zstd.compress(cl, 9)
            user_bytes.extend(compressed_user_cl)

    finish_time_bench = time.perf_counter()
    print(f"ZSTD Compression finished in {finish_time_bench - start_time_bench} seconds")

    total_size = (sys.getsizeof(kern_bytes) + sys.getsizeof(user_bytes)) / (1024 * 1024)  # Get Size in MB
    return [total_size, finish_time_bench - start_time_bench, "ZSTD"]


def benchmark_zlib(epochs, calculate_xor):
    kern_bytes = bytearray()
    user_bytes = bytearray()
    start_time_bench = time.perf_counter()
    for epoch in epochs:
        for kern_cl in epoch.kern_cl_changes:
            cl = kern_cl[2]
            if calculate_xor:
                cl = rle.xor(kern_cl[1], kern_cl[2])
            compressed_kern_cl = zlib.compress(cl, 9)
            kern_bytes.extend(compressed_kern_cl)
        for user_cl in epoch.user_cl_changes:
            cl = user_cl[2]
            if calculate_xor:
                cl = rle.xor(user_cl[1], user_cl[2])
            compressed_user_cl = zlib.compress(cl, 9)
            user_bytes.extend(compressed_user_cl)

    finish_time_bench = time.perf_counter()
    print(f"ZLIB Compression finished in {finish_time_bench - start_time_bench} seconds")

    total_size = (sys.getsizeof(kern_bytes) + sys.getsizeof(user_bytes)) / (1024 * 1024)  # Get Size in MB
    return [total_size, finish_time_bench - start_time_bench, "ZLIB"]


def benchmark_gzip(epochs, calculate_xor):
    kern_bytes = bytearray()
    user_bytes = bytearray()
    start_time_bench = time.perf_counter()
    for epoch in epochs:
        for kern_cl in epoch.kern_cl_changes:
            cl = kern_cl[2]
            if calculate_xor:
                cl = rle.xor(kern_cl[1], kern_cl[2])
            compressed_kern_cl = gzip.compress(cl, 9)
            kern_bytes.extend(compressed_kern_cl)
        for user_cl in epoch.user_cl_changes:
            cl = user_cl[2]
            if calculate_xor:
                cl = rle.xor(user_cl[1], user_cl[2])
            compressed_user_cl = gzip.compress(cl, 9)
            user_bytes.extend(compressed_user_cl)

    finish_time_bench = time.perf_counter()
    print(f"GZIP Compression finished in {finish_time_bench - start_time_bench} seconds")

    total_size = (sys.getsizeof(kern_bytes) + sys.getsizeof(user_bytes)) / (1024 * 1024)  # Get Size in MB
    return [total_size, finish_time_bench - start_time_bench, "GZIP"]


def benchmark_lz4(epochs, filename):
    kern_bytes = bytearray()
    user_bytes = bytearray()
    start_time_bench = time.perf_counter()
    for epoch in epochs:
        for kern_cl in epoch.kern_cl_changes:
            # xor = rle.xor(kern_cl[1], kern_cl[2])
            compressed_kern_cl = lz4.frame.compress(kern_cl[2], 16)
            kern_bytes.extend(compressed_kern_cl)
        for user_cl in epoch.user_cl_changes:
            # xor = rle.xor(user_cl[1], user_cl[2])
            compressed_user_cl = lz4.frame.compress(user_cl[2], 16)
            user_bytes.extend(compressed_user_cl)

    finish_time_bench = time.perf_counter()
    print(f"LZ4 Compression finished in {finish_time_bench - start_time_bench} seconds")

    total_size = (sys.getsizeof(kern_bytes) + sys.getsizeof(user_bytes)) / (1024 * 1024)  # Get Size in MB
    return [total_size, finish_time_bench - start_time_bench, "LZ4"]


def benchmark_snappy(epochs, filename):
    kern_bytes = bytearray()
    user_bytes = bytearray()
    start_time_bench = time.perf_counter()
    for epoch in epochs:
        for kern_cl in epoch.kern_cl_changes:
            # xor = rle.xor(kern_cl[1], kern_cl[2])
            compressed_kern_cl = snappy.compress(kern_cl[2])
            kern_bytes.extend(compressed_kern_cl)
        for user_cl in epoch.user_cl_changes:
            # xor = rle.xor(user_cl[1], user_cl[2])
            compressed_user_cl = snappy.compress(user_cl[2])
            user_bytes.extend(compressed_user_cl)

    finish_time_bench = time.perf_counter()
    print(f"Snappy Compression finished in {finish_time_bench - start_time_bench} seconds")

    total_size = (sys.getsizeof(kern_bytes) + sys.getsizeof(user_bytes)) / (1024 * 1024)  # Get Size in MB
    return [total_size, finish_time_bench - start_time_bench, "SNAPPY"]


def benchmark_rle(epochs, filename):
    kern_bytes = bytearray()
    user_bytes = bytearray()
    start_time_bench = time.perf_counter()
    for epoch in epochs:
        for kern_cl in epoch.kern_cl_changes:
            xor = rle.xor(kern_cl[1], kern_cl[2])
            compressed_kern_cl_values, compressed_kern_cl_appearances = rle.encode_2(xor)
            kern_bytes.extend(compressed_kern_cl_values)
            kern_bytes.extend(compressed_kern_cl_appearances)
        for user_cl in epoch.user_cl_changes:
            xor = rle.xor(user_cl[1], user_cl[2])
            compressed_user_cl_values, compressed_user_cl_appearances = rle.encode_2(xor)
            user_bytes.extend(compressed_user_cl_values)
            user_bytes.extend(compressed_user_cl_appearances)

    finish_time_bench = time.perf_counter()
    print(f"RLE Compression finished in {finish_time_bench - start_time_bench} seconds")

    total_size = (sys.getsizeof(kern_bytes) + sys.getsizeof(user_bytes)) / (1024 * 1024)  # Get Size in MB
    return [total_size, finish_time_bench - start_time_bench, "RLE"]


def benchmark(epochs, compression_algo):
    kern_bytes = bytearray()
    user_bytes = bytearray()
    start_time_bench = time.perf_counter()
    for epoch in epochs:
        for kern_cl in epoch.kern_cl_changes:
            # xor = rle.xor(kern_cl[1], kern_cl[2])
            compressed_kern_cl = operations[compression_algo](kern_cl[2])
            kern_bytes.extend(compressed_kern_cl)
        for user_cl in epoch.user_cl_changes:
            # xor = rle.xor(user_cl[1], user_cl[2])
            compressed_user_cl = operations[compression_algo](user_cl[2])
            user_bytes.extend(compressed_user_cl)

    finish_time_bench = time.perf_counter()
    print(f"{compression_algo} Compression finished in {finish_time_bench - start_time_bench} seconds")

    total_size = (sys.getsizeof(kern_bytes) + sys.getsizeof(user_bytes)) / (1024 * 1024)  # Get Size in MB
    return [total_size, finish_time_bench - start_time_bench, "LZ4"]



def no_compression(data):
    return data


def zlib_compression(data):
    return zlib.compress(data, 9)


def zstd_compression(data):
    return zstd.compress(data, 9)


def gzip_compression(data):
    return gzip.compress(data, 9)


def lz4_compression(data):
    return lz4.frame.compress(data, 16)


def snappy_compression(data):
    return snappy.compress(data)

operations = {
    'no_compression': lambda data: no_compression(data),
    'zlib': lambda data: zlib.compress(data, 9),
    'zstd': lambda data: zstd.compress(data, 9),
    'gzip': lambda data: gzip.compress(data, 9),
    'lz4': lambda data: lz4_compression(data),
    'snappy': lambda data: snappy_compression(data)

}