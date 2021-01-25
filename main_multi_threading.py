from fastTB import run_model
import multiprocessing
import pandas as pd
import math
import os

current_folder = os.path.dirname(__file__)

path_preprocess = 'result_preprocess/result_preprocess.csv'
output_result = 'output_test/output.csv'

folder_split_df_path = os.path.join(current_folder, 'split_df')
if not os.path.isdir(folder_split_df_path):
    os.mkdir(folder_split_df_path)

folder_output_final_result = os.path.join(current_folder, 'final_result')
if not os.path.isdir(folder_output_final_result):
    os.mkdir(folder_output_final_result)


def run_df_model(path_input_df_data, output_path):
    df_data = pd.read_parquet(path_input_df_data)
    df_data[['result']] = df_data.apply(
        lambda row: run_model(row['raw_only_clean'], row['raw_sentences'], row['title'], row['number_of_nouns']),
        axis=1)
    df_data.to_parquet(output_path, index=False)


def split_df(path_input, num_workers, version, start=None, end=None):
    # result = []
    if start is not None and end is not None:
        df_input = pd.read_csv(path_input, delimiter='|')[start:end]
    else:
        df_input = pd.read_csv(path_input, delimiter='|')[start:]
    len_df = len(df_input)
    num_each_worker = math.floor(len(df_input) / num_workers)
    current_cursor = 0
    for worker_i in range(num_workers):
        to_cursor = current_cursor + num_each_worker
        print(f'to_cursor_before = {to_cursor}, num_each_worker: {num_each_worker}')
        if worker_i == num_workers - 1:
            to_cursor = len_df
        df_worker = df_input[current_cursor: to_cursor]
        path_split_df_worker_i = os.path.join(folder_split_df_path, f'{version}_{worker_i}.parquet')
        print(f'writing to file at: {path_split_df_worker_i}')
        df_worker.to_parquet(path_split_df_worker_i, index=False)
        # result.append(df_worker)
        print(f'worker: {worker_i}, index_start: {current_cursor}, '
              f'index_end: {to_cursor - 1}, len_df: {len(df_worker)}, total_df: {len_df}')
        current_cursor = to_cursor


def run_multi_processing(num_workers, version):
    process_list = []
    for worker_i in range(num_workers):
        path_input_worker_i = os.path.join(folder_split_df_path, f'{version}_{worker_i}.parquet')
        path_output_worker_i = os.path.join(folder_output_final_result, f'output_{version}_{worker_i}.parquet')
        process_ = multiprocessing.Process(target=run_df_model, args=(path_input_worker_i, path_output_worker_i))
        process_.start()
        process_list.append(process_)
        print(f'start process: {process_.pid}')
        # process_.join()
    for p in process_list:
        p.join()

    print('DONE')


def do_all_in_one(num_workers, version, start, end):
    # Step 1: split file
    print('SPLIT FILE')
    split_df(path_input=path_preprocess, num_workers=num_workers, version=version, start=start, end=end)

    # Step 2: run multi processing
    print('RUN MULTI PROCESSING')
    run_multi_processing(num_workers=num_workers, version=version)

# MAINNNN CONFIG ============================
num_workers = 10

batch_size = 500
max = 11490
min = 0
num_batch = math.ceil((max - min) / batch_size)
start = min
end = 0
for i in range(num_batch):
    end = start + batch_size
    # CONFIG
    version = 'typebase191_dm_cnn_11k_top_{}_{}'.format(start, end)
    print(f'start = {start}, end = {end}')
    if end > 11200:
        end = None
    do_all_in_one(num_workers, version, start, end)
    start = end
