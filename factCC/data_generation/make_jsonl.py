import csv, json, sys
from tqdm import tqdm 
csv.field_size_limit(sys.maxsize)

jsonl = []
with open('../../../data/data_15_18.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile, fieldnames=['date', 'decision', 'domain', 'claims', 'abstract'])
    for i, row in enumerate(tqdm(reader)):
        jsonl.append({'id': i, 'text': row['claims']})


with open('../data/claims_15_18.jsonl', 'w') as outfile:
    for entry in jsonl:
        json.dump(entry, outfile)
        outfile.write('\n')