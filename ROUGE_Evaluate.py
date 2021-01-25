import os
from rouge import Rouge
import statistics as sta
import pandas as pd
import glob


def read_multi_df(path_folder, pattern):
    all_files = glob.glob(path_folder + f'/{pattern}')
    li = []
    for filename in all_files:
        df = pd.read_parquet(filename)
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame


def main(path_abstracts, path_folder, pattern, path_rouge):
    # hyp = 'hyp'
    # raw_ref = 'abstracts'
    # FJoin = os.path.join
    # files_hyp = [FJoin(hyp, f) for f in os.listdir(hyp)]
    # files_raw_ref = [FJoin(raw_ref, f) for f in os.listdir(hyp)]
    df_abs = pd.read_parquet(path_abstracts)
    df_final_result = read_multi_df(path_folder, pattern)

    df_join = pd.merge(df_final_result, df_abs, how='inner', on='id_file_name')
    print(f'len df_join = {len(df_join)}')

    rouge_1_tmp = []
    rouge_2_tmp = []
    rouge_L_tmp = []
    for row in df_join.itertuples():
        abstract = row.abstract
        predict = row.result
        rouge = Rouge()
        scores = rouge.get_scores(predict, abstract, avg=True)
        rouge_1 = scores["rouge-1"]["f"]
        rouge_2 = scores["rouge-2"]["f"]
        rouge_L = scores["rouge-l"]["f"]
        rouge_1_tmp.append(rouge_1)
        rouge_2_tmp.append(rouge_2)
        rouge_L_tmp.append(rouge_L)

    rouge_1_avg = sta.mean(rouge_1_tmp)
    rouge_2_avg = sta.mean(rouge_2_tmp)
    rouge_L_avg = sta.mean(rouge_L_tmp)
    print('Rouge-1')
    print(rouge_1_avg)
    print('Rouge-2')
    print(rouge_2_avg)
    print('Rouge-L')
    print(rouge_L_avg)
    print(f'writing rouge result at {path_rouge}.....')
    with open(path_rouge, 'w', encoding='utf-8') as f:
        f.write(f'Rouge-1: {rouge_1_avg}\n')
        f.write(f'Rouge-2: {rouge_2_avg}\n')
        f.write(f'Rouge-L: {rouge_L_avg}\n')
    print('DONE WRITE ROUGE RESULT')


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    path_abstracts = os.path.join(current_dir, 'abstracts', 'abstract.parquet')
    path_predict = os.path.join(current_dir, 'final_result')
    # CONFIG
    version = 'typebase191_dm_cnn_11k_top_400'

    version_regex = f'output_{version}*.parquet'
    if not os.path.isdir(os.path.join(current_dir, 'rouge_result')):
        os.mkdir(os.path.join(current_dir, 'rouge_result'))
    path_rouge = os.path.join(current_dir, 'rouge_result', f'rouge_{version}.txt')
    main(path_abstracts, path_predict, version_regex, path_rouge)
