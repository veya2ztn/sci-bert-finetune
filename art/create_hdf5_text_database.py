from multiprocessing import Pool
import h5py
import csv
from tqdm import tqdm

def process_row(row):
    return [row[1], row[2]]


def convert2hdf5_parallel(file_path, hdf5_path, full_length):
    # Create a pool of workers
    with Pool() as pool:
        with open(file_path) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            next(reader, None)  # Skip the header
            # Use the pool's imap method to apply process_row to each row
            results = list(tqdm(pool.imap(process_row, reader), total=full_length))

    with h5py.File(hdf5_path, 'w') as file:
        dataset = file.create_dataset('data', shape=(full_length, 2), dtype=h5py.special_dtype(vlen=str))
        # Write the processed data to the HDF5 file sequentially
        for i, result in enumerate(results):
            dataset[i, :] = result
                

if __name__ == "__main__":
    convert2hdf5_parallel("data/wikipedia-split/psgs_w100.tsv",
                          "data/wikipedia-split/psgs_w100.hdf5",
                          21015325)

