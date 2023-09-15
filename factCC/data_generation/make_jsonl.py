import csv, json, sys, os
from tqdm import tqdm 
csv.field_size_limit(sys.maxsize)

def load_data(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

def save_data(data_list, output_file):
    with open(output_file, "w", encoding="utf-8") as fd:
        for example in data_list:
            example = dict(example)
            fd.write(json.dumps(example, ensure_ascii=False) + "\n")


jsonl = []
# #==============================================================================#
# with open('../../../data/data_15_18.csv', newline='') as csvfile:
#     reader = csv.DictReader(csvfile, fieldnames=['date', 'decision', 'domain', 'claims', 'abstract'])
#     for i, row in enumerate(tqdm(reader)):
#         jsonl.append({'id': i, 'text': row['claims']})


# with open('../data/claims_15_18.jsonl', 'w') as outfile:
#     for entry in jsonl:
#         json.dump(entry, outfile)
#         outfile.write('\n')
# #==============================================================================#

# combine positive and negative examples
positive_examples = load_data("../data/claims_15_18-positive-noise.jsonl")
negative_examples = load_data("../data/claims_15_18-negative-noise.jsonl")

data = positive_examples + negative_examples
save_data(data, "../data/data-train.jsonl")
