"""
sample randomly 100 patent applications
"""
import os, json, csv, psutil, sys
import random
from glob import glob
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import re
import multiprocessing
from nltk.tokenize import word_tokenize

csv.field_size_limit(sys.maxsize)


def extract_data(filename):
    res = {}
    with open(filename) as json_file:
        data = json.load(json_file)

    res['date'] = data['filing_date']
    res['decision'] = data['decision']
    res['domain'] = data['main_ipcr_label']
    res['claims'] = data['claims']
    res['abstract'] = data['abstract']
    return res

#=======================================================================================#
# DATA_DIRS = ['../qatent_bigdir/hearst-pattern/data/2018/', '../qatent_bigdir/hearst-pattern/data/2017/', '../qatent_bigdir/hearst-pattern/data/2016/', '../qatent_bigdir/hearst-pattern/data/2015/']
# fils = []
# for dir in DATA_DIRS:
#     fils.extend(glob(os.path.join(dir, '*.json')))
# random.shuffle(fils)

# threadn = psutil.cpu_count()
# print(f'we have {threadn} precessors')

# pool = multiprocessing.Pool(threadn)
# with open('./data/data_15_18.csv', 'a', newline='') as csvfile:
#     colnames = ['date', 'decision', 'domain', 'claims', 'abstract']
#     csvwriter = csv.DictWriter(csvfile, fieldnames=colnames)
#     for row in tqdm(pool.imap_unordered(extract_data, fils), total=len(fils)):
#         csvwriter.writerow(row)
#=======================================================================================#

#=======================================================================================#
df_dict = defaultdict(list)
domain_counter = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0}
target_domains = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

# sample 500 accepted patents filed in 2017 and 2018 (40 for each domain)
with open('./data/data_15_18.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile, fieldnames=['date', 'decision', 'domain', 'claims', 'abstract'])
    for row in tqdm(reader):
        domain = row['domain'][0]
        decision = row['decision']
        if decision == 'ACCEPTED' and domain in target_domains and row['date'][:4] in ['2017', '2018'] and ('(canceled)' not in row['claims']):
            df_dict['domain'].append(domain)
            df_dict['claims'].append(row['claims'])
            df_dict['abstract'].append(row['abstract'])
            domain_counter[domain] += 1
            if domain_counter[domain] >= 50:
                target_domains.remove(domain)
            if target_domains == []:
                break

df = pd.DataFrame(df_dict)
df.to_csv('./data/eval_data.csv', index=False)
#=======================================================================================#

#=======================================================================================#
# statistics 
df = pd.read_csv('./data/eval_data.csv')
domains = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
for d in domains:
    sub_df = df[df['domain']==d]
    nb_claims = sub_df['claims'].apply(lambda c: len(re.findall('\d+. [AT]', c)))
    nb_words_claims = sub_df['claims'].apply(lambda x: len(word_tokenize(x)))
    nb_words_abstract = sub_df['abstract'].apply(lambda x: len(word_tokenize(x)))
    print('domain:', d)
    print('number of claims:', sum(nb_claims)/len(nb_claims))
    print('number of words of claims:', sum(nb_words_claims)/len(nb_words_claims))
    print('number of words of abstracts:', sum(nb_words_abstract)/len(nb_words_abstract))
#=======================================================================================#
