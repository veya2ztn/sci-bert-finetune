import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import sys 
def convert_to_float16(chunk):
    return chunk.astype(np.float16)


MAXFLOAT16 = 60000 #65500
MINFLOAT16 =-60000
def convert_large_array_to_float16(arr, num_workers=None):
    arr = arr*1000 # the accelerate in float16 from m/s^2 to mm/s^2

    arr[arr > MAXFLOAT16] = MAXFLOAT16
    arr[arr < MINFLOAT16] = MINFLOAT16
    if num_workers is None:
        num_workers = len(os.sched_getaffinity(0))

    chunk_size = arr.shape[0] // num_workers
    chunks = [arr[i * chunk_size:(i + 1) * chunk_size]
              for i in range(num_workers)]

    # If there are any remaining elements, add them to the last chunk
    if arr.shape[0] % num_workers != 0:
        chunks[-1] = np.vstack((chunks[-1], arr[num_workers * chunk_size:]))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        converted_chunks = list(
            tqdm(executor.map(convert_to_float16, chunks), total=len(chunks)))

    return np.vstack(converted_chunks)


if __name__ == "__main__":
    
    origin_path = sys.argv[1] #"TEAM/ITALY/test.stead.aligned.b2.npy"
    print(f"Loading data from {origin_path} .........")
    arr = np.load(origin_path)
    print("done..........")
    converted_arr = convert_large_array_to_float16(arr)
    print("Original array dtype:", arr.dtype)
    print("Converted array dtype:", converted_arr.dtype)
    
    save_path = origin_path.replace('.npy','.f16.npy')
    print("Save data to:", save_path)
    np.save(save_path,converted_arr)