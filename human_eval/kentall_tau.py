from scipy.stats import kendalltau
import argparse
import json
import pandas as pd

import torch
import torch.nn as nn

import csv
csv.field_size_limit(100000000)

from tqdm import tqdm

from collections import defaultdict

from sklearn.preprocessing import MultiLabelBinarizer


def get_rank(annotation_list):
    res = []
    for line in annotation_list:
        if line["type"] == "pairwise":
            if line["value"]["selected"] == "left": 
                res.append(1)
                res.append(2)
            elif line["value"]["selected"] == "right": 
                res.append(2)
                res.append(1)
    if res == []:
        res = [1, 1]
    return res


def read_pairs(annotation_file):
    """
    read the selected pairs information from jsonl
    """
    with open(annotation_file) as f:  annotations = json.load(f)

    new_annotations, input1_data, input2_data, input_claims = [], [], [], []
    for annot in annotations:
        selected_annotations = []
        input1_data.append(annot["data"]["output1"])
        input2_data.append(annot["data"]["output2"])
        input_claims.append(annot["data"]["input_claim"])

        for la in annot["annotations"]:
            if "zuo" in la["completed_by"]["email"].lower():
                selected_annotations.append(la)
        if len(selected_annotations) > 0:
            annot["annotations"] = [d["result"] for d in selected_annotations][0]
            annot["annotations"] = [a for a in annot["annotations"] if a["type"] != "labels"]
            new_annotations.append(annot["annotations"])
        else: # skipped or emply annotations
            annot["annotations"] = []
            new_annotations.append(annot["annotations"])

    # get ranking for each pair
    human_rankings = []
    for line in new_annotations:
        human_rankings.append(get_rank(line))
    return input1_data, input2_data, input_claims, human_rankings
  

def get_ground_truths(ground_truths_file, task_name):
    df = pd.read_csv(ground_truths_file)

    if task_name == "c2a":
        idx = df[(df["domain"] == "G") | (df["domain"] == "A")].index.tolist()
        column_name = "abstract"
    elif task_name == "c2c":
        idx = df[(df["domain"] == "G") | (df["domain"] == "A") | (df["is_dependent"] == False)].index.tolist()
        column_name = "true_next_claim"

    # repeat each index two time continuously
    idx = [x for x in idx for i in range(2)]
    orig_data = df.iloc[idx].reset_index(drop=True)[column_name].tolist()
    return orig_data


def get_metric_rankings(output1_data, output2_data, input_claims, metric_name):
    rankings = []
    if metric_name == "semsim-ipc":
        from transformers import BertTokenizer, BertForSequenceClassification
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint_path = "../ipc_cls/output/checkpoint_epoch3.pt"

        model_name = "../ipc_cls/bert-for-patents/"
        tokenizer = BertTokenizer.from_pretrained(model_name)

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

        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(mlb.classes_))
        model.to(device)

        # Wrap the model with DataParallel
        model = nn.DataParallel(model)
        
        # load last checkpoint
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint)
        print(f"Resuming training from checkpoint: {checkpoint_path}")


        # from sentence_transformers import SentenceTransformer, util
        # model = SentenceTransformer(model_path, device="cuda")

        # #Sentences are encoded by calling model.encode()
        # emb1 = model.encode("This is a red cat with a hat.")
        # emb2 = model.encode("Have you seen my red cat?")

        # cos_sim = util.cos_sim(emb1, emb2)
        # print("Cosine-Similarity:", cos_sim)
    elif metric_name == "term-overlap":
        import spacy
        from pyate.term_extraction_pipeline import TermExtractionPipeline

        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("combo_basic")

        for claim, abstract1, abstract2 in zip(input_claims, output1_data, output2_data):
            terms_claim = nlp(claim)._.combo_basic.sort_values(ascending=False)
            terms_abstract1 = nlp(abstract1)._.combo_basic.sort_values(ascending=False)
            terms_abstract2 = nlp(abstract2)._.combo_basic.sort_values(ascending=False)

            # convert pandas series to dictinary, and keep those with value > 1.0
            terms_claim = [k for k, v in terms_claim.to_dict().items() if v > 1.0]
            terms_abstract1 = [k for k, v in terms_abstract1.to_dict().items() if v > 1.0]
            terms_abstract2 = [k for k, v in terms_abstract2.to_dict().items() if v > 1.0]

            # get the number of overlapping terms between claim and abstract
            overlap1 = len(set(terms_claim).intersection(set(terms_abstract1))) / len(set(terms_claim))
            overlap2 = len(set(terms_claim).intersection(set(terms_abstract2))) / len(set(terms_claim))

            if overlap1 > overlap2:
                rankings.append([1, 2])
            elif overlap1 < overlap2:
                rankings.append([2, 1])
            else:
                rankings.append([1, 1])

    return rankings



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="c2a")
    parser.add_argument("--metric", type=str, default="semsim-ipc")
    args = parser.parse_args()

    task_name = args.task
    metric_name = args.metric

    annotation_file = f"./annotations/version3_{task_name}.json"
    output1_data, output2_data, input_claims, human_rankings = read_pairs(annotation_file)

    # read ground truths data 
    if task_name == "c2c":
        ground_truths_file = "../data/eval_data_c2c.csv"
    elif task_name == "c2a":
        ground_truths_file = "../data/eval_data.csv"
    ground_truths = get_ground_truths(ground_truths_file, task_name)    # usually not allowed to use for real case ?

    metric_rankings = get_metric_rankings(output1_data, output2_data, input_claims, metric_name)

    # Calculate Kendall's Tau
    tau, p_value = kendalltau(metric_rankings, human_rankings)

    print(f"Kendall's Tau: {tau}")
    print(f"P-value: {p_value}")

    # Interpretation
    if tau > 0:
        print("Metric is positively correlated with human judgments.")
    elif tau < 0:
        print("Metric is negatively correlated with human judgments.")
    else:
        print("No correlation between metric and human judgments.")



if __name__ == "__main__":
    main()
