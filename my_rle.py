def encode_1(data):
    """Run-length encodes byte data."""
    encoded_data = bytearray()
    i = 0
    while i < len(data):
        count = 1
        while i + count < len(data) and data[i + count] == data[i]:
            count += 1
        if count > 255:
            count = 255
        encoded_data.extend([count, data[i]])
        i += count
    return bytes(encoded_data)


def decode_1(encoded_data):
    """Run-length decodes byte data."""
    decoded_data = bytearray()
    i = 0
    while i < len(encoded_data):
        count = encoded_data[i]
        value = encoded_data[i + 1]
        decoded_data.extend([value] * count)
        i += 2
    return bytes(decoded_data)


def encode_2(data):
    count = 1
    prev = data[0]
    values = bytearray()
    counts = bytearray()
    for curr in data[1:]:
        if curr != prev:
            values.append(prev)
            counts.append(count)
            count = 1
            prev = curr
        else:
            count += 1
    values.append(prev)
    counts.append(count)
    return values, counts

def decode_2(values, counts):
    result = bytearray()
    for value, count in zip(values, counts):
        result.extend(bytearray(count * [value]))
    return result



def xor(byte1, byte2):
    byte_array_1 = bytearray(byte1)
    byte_array_2 = bytearray(byte2)

    # Perform XOR operation on each pair of bytes
    result = bytearray()
    for byte1, byte2 in zip(byte_array_1, byte_array_2):
        result.append(byte1 ^ byte2)

    return result
