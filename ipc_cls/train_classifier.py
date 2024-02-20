import torch
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm
import csv, sys
from collections import defaultdict
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

import warnings
warnings.filterwarnings('ignore')

csv.field_size_limit(sys.maxsize)

class MultiLabelDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
    
# Prepare the dataset
datasets = {'train': defaultdict(list), 'test': defaultdict(list)}
with open('../data/data_15_18.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile, fieldnames=['date', 'decision', 'domain', 'claims', 'abstract'])
    for row in tqdm(reader):
        if row['date'][:4] in ['2016', '2017']:
            datasets['train']['labels'].append(row['domain'][:4])
            datasets['train']['text'].append('[abstract] '+row['abstract'])

            datasets['train']['labels'].append(row['domain'][:4])
            datasets['train']['text'].append('[claims] '+row['claims'])
        elif row['date'][:4] == '2018':
            datasets['test']['labels'].append(row['domain'][:4])
            datasets['test']['text'].append('[abstract] '+row['abstract'])

            datasets['test']['labels'].append(row['domain'][:4])
            datasets['test']['text'].append('[claims] '+row['claims'])

train_dataset = pd.DataFrame(datasets['train'])
test_dataset = pd.DataFrame(datasets['test'])

# ########################## create and save test datasets ##########################
test_dataset1 = test_dataset[test_dataset['text'].apply(lambda x: x.startswith('[abstract]'))]
test_dataset2 = test_dataset[test_dataset['text'].apply(lambda x: x.startswith('[claims]'))]
assert len(test_dataset1) == len(test_dataset2)

test_dataset_all = pd.concat([test_dataset1, test_dataset2])

test_dataset1.to_csv('./dataset/abstract_testset.csv')
test_dataset2.to_csv('./dataset/claims_testset.csv')
test_dataset_all.to_csv('./dataset/abstract_claims_testset.csv')

# ###################################################################################
test_dataset1 = pd.read_csv('./dataset/abstract_testset.csv')
test_dataset2 = pd.read_csv('./dataset/claims_testset.csv')

# convert label string into numbers
mlb = MultiLabelBinarizer()
mlb.fit_transform([[label] for label in list(set(train_dataset['labels'].to_list()))])

train_dataset.loc[:, 'labels'] = train_dataset['labels'].apply(lambda x: mlb.transform([[x]])[0])
test_dataset.loc[:, 'labels'] = test_dataset['labels'].apply(lambda x: mlb.transform([[x]])[0])

print("Train dataset size:", len(train_dataset)
        , "\nTest dataset size:", len(test_dataset))

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "./bert-for-patents/"
tokenizer = BertTokenizer.from_pretrained(model_name)

train_params = {'batch_size': 64,
                'shuffle': True,
                'num_workers': 4
                }

test_params = {'batch_size': 64,
                'shuffle': True,
                'num_workers': 4
                }
EPOCHS = 3
LEARNING_RATE = 1e-05

# Define the directory path where you want to save the model
output_dir = "./output"

train_data = MultiLabelDataset(train_dataset, tokenizer)
training_loader = DataLoader(train_data, **train_params)
# test_loader = DataLoader(test_data, **test_params)



# Check if a saved checkpoint exists
lst_epoch = 0
checkpoints = sorted([fname for fname in os.listdir(output_dir) if fname.startswith('checkpoint')])
if checkpoints:
    checkpoint_path = f"{output_dir}/{checkpoints[-1]}"
    model = BertForSequenceClassification.from_pretrained(checkpoint_path, num_labels=len(mlb.classes_))
    model.to(device)

    print(f"Resuming training from checkpoint: {checkpoint_path}")
    lst_epoch = int(checkpoint_path.split('_epoch')[1].split('.')[0])

else:
    # Load pre-trained BERT model and tokenizer
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(mlb.classes_))
    model.to(device)

# Wrap the model with DataParallel
model = nn.DataParallel(model)

# Set optimizer and loss function
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

# Enable mixed-precision training
scaler = GradScaler()

# Training loop
for epoch in range(lst_epoch, EPOCHS):
    model.train()
    train_loss = 0.0

    progress_bar = tqdm(training_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)
    for data in progress_bar:
    # for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)
        
        optimizer.zero_grad()

        # Use autocast to automatically cast operations to mixed precision
        with autocast():
            outputs = model(ids, attention_mask=mask)
            loss = loss_fn(outputs.logits, targets)

        # Perform backpropagation and optimization using the GradScaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        
        # Update the progress bar
        progress_bar.set_postfix({"Train Loss": loss.item()})

    # Saving Model, Configuration, and Tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of DataParallel
    model_to_save.save_pretrained(f"{output_dir}/checkpoint_epoch{epoch + 1}")


    
# Validation for dataset1
model.eval()
fin_targets=[]
fin_outputs=[]
with torch.no_grad():
    for _, data in tqdm(test_dataset1.iterrows(), total=len(test_dataset1)):
        targets = mlb.transform([[data["labels"]]]).argmax().item()

        inputs = tokenizer(data["text"], return_tensors="pt", max_length=512, truncation=True, padding=True)
        outputs = model(**inputs).logits.argmax().item()

        fin_targets.append(targets)
        fin_outputs.append(outputs)

outputs1, targets1 = fin_outputs, fin_targets

accuracy = metrics.accuracy_score(targets1, outputs1)
f1_score = metrics.f1_score(targets1, outputs1, average='weighted')
precision_score = metrics.precision_score(targets1, outputs1, average='weighted')
recall_score = metrics.recall_score(targets1, outputs1, average='weighted')
print("Evaluation results on abstracts:")
print(f"Accuracy Score = {accuracy}")
print(f"F1 Score (Weighted) = {f1_score}")
print(f"Precision Weighted) = {precision_score}")
print(f"Recall (Weighted) = {recall_score}")
# save predictions
res_df1 = pd.DataFrame({'prediction': [mlb.classes_[label] for label in outputs1], 'actuals': [mlb.classes_[label] for label in targets1]})
res_df1.to_csv('./dataset/abstract_pred.csv', index=False)
print("===============================")


# Validation for dataset2
fin_targets=[]
fin_outputs=[]
with torch.no_grad():
    for _, data in tqdm(test_dataset2.iterrows(), total=len(test_dataset2)):
        targets = mlb.transform([[data["labels"]]]).argmax().item()

        inputs = tokenizer(data["text"], return_tensors="pt", max_length=512, truncation=True, padding=True)
        outputs = model(**inputs).logits.cpu().argmax().item()

        fin_targets.append(targets)
        fin_outputs.append(outputs)

outputs2, targets2 = fin_outputs, fin_targets

accuracy = metrics.accuracy_score(targets2, outputs2)
f1_score = metrics.f1_score(targets2, outputs2, average='weighted')
precision_score = metrics.precision_score(targets2, outputs2, average='weighted')
recall_score = metrics.recall_score(targets2, outputs2, average='weighted')
print("Evaluation results on claims:")
print(f"Accuracy Score = {accuracy}")
print(f"F1 Score (Weighted) = {f1_score}")
print(f"Precision (Weighted) = {precision_score}")
print(f"Recall (Weighted) = {recall_score}")
# save predictions
res_df2 = pd.DataFrame({'prediction': [mlb.classes_[label] for label in outputs2], 'actuals': [mlb.classes_[label] for label in targets2]})
res_df2.to_csv('./dataset/claims_pred.csv', index=False)
print("===============================")


# Validation for all
outputs3 = outputs1 + outputs2
targets3 = targets1 + targets2
accuracy = metrics.accuracy_score(targets3, outputs3)
f1_score = metrics.f1_score(targets3, outputs3, average='weighted')
precision_score = metrics.precision_score(targets3, outputs3, average='weighted')
recall_score = metrics.recall_score(targets3, outputs3, average='weighted')
print("Evaluation results on abstracts + claims:")
print(f"Accuracy Score = {accuracy}")
print(f"F1 Score (Weighted) = {f1_score}")
print(f"Precision (Weighted) = {precision_score}")
print(f"Recall (Weighted) = {recall_score}")
# save predictions
res_df3 = pd.DataFrame({'prediction': [mlb.classes_[label] for label in outputs3], 'actuals': [mlb.classes_[label] for label in targets3]})
res_df3.to_csv('./dataset/abstract_claims_pred.csv', index=False)
print("===============================")