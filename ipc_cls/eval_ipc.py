import argparse, os, csv, sys
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertForSequenceClassification

csv.field_size_limit(sys.maxsize)

parser = argparse.ArgumentParser()
parser.add_argument('--path_data', type=str, default="../data/eval_data.csv")
parser.add_argument('--path_prediction', required=True, type=str)
parser.add_argument('--path_model', required=False, type=str, default="./output/checkpoint_epoch3.pt")
parser.add_argument('--path_output', type=str, default="./evaluation")

# Parse the command-line arguments
args = parser.parse_args()


if __name__ == '__main__':
    df = pd.read_csv(args.path_data)
    claims = df['claims'].apply(lambda x: '[claims] '+x).to_list()
    abstracts = pd.read_csv(args.path_prediction)['abstract'].fillna("").apply(lambda x: '[abstract] '+x).to_list()
    actuals = df['domain'].to_list()

    path_eval = args.path_output
    if not os.path.isdir(path_eval):
        os.makedirs(path_eval)
    method = args.path_prediction.split('/')[-1].split('.')[0]
    path_eval = os.path.join(path_eval, method)

    datasets = {'train': defaultdict(list), 'test': defaultdict(list)}
    with open('../data/data_15_18.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=['date', 'decision', 'domain', 'claims', 'abstract'])
        for row in tqdm(reader):
            if row['date'][:4] in ['2016', '2017']:
                datasets['train']['labels'].append(row['domain'][:4])
                datasets['train']['labels'].append(row['domain'][:4])

    train_dataset = pd.DataFrame(datasets['train'])

    # convert label string into numbers
    mlb = MultiLabelBinarizer()
    mlb.fit_transform([[label] for label in list(set(train_dataset['labels'].to_list()))])

    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained BERT model and tokenizer
    model_name = "./bert-for-patents/"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(mlb.classes_))
    model.to(device)

    # Wrap the model with DataParallel
    model = nn.DataParallel(model)
    
    # load last checkpoint
    checkpoint_path = os.path.join(args.path_model)
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint)
    print(f"Resuming training from checkpoint: {checkpoint_path}")

    scores, predictions_claim, predictions_abstract = [], [], []
    with torch.no_grad():
        for claim, abstract in tqdm(zip(claims, abstracts), total=len(df)):
            max_length = 512  # Set the maximum number of tokens

            inputs_claim = tokenizer(claim, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
            inputs_abstract = tokenizer(abstract, return_tensors="pt", max_length=max_length, truncation=True, padding=True)

            logits_claim = model(**inputs_claim).logits
            logits_abstract= model(**inputs_abstract).logits

            predicted_class_id_claim = logits_claim.argmax().item()
            predicted_class_id_abstract = logits_abstract.argmax().item()

            # convert label id into IPC code
            predictions_claim.append(mlb.classes_[predicted_class_id_claim])
            predictions_abstract.append(mlb.classes_[predicted_class_id_abstract])

            if predicted_class_id_claim == predicted_class_id_abstract:
                scores.append(1)
            else:
                scores.append(0)

    # save predicted IPC labels into file
    df_res = pd.DataFrame({'claim_ipc': predictions_claim, 'abstract_ipc': predictions_abstract, 'actuals': actuals})
    df_res.to_csv(f'{path_eval}.csv', index=False)
    print(args.path_prediction, sum(scores)/len(scores))

