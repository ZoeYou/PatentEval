from scipy.stats import kendalltau
import argparse
import json
import pandas as pd

import csv
csv.field_size_limit(100000000)

from tqdm import tqdm
from nltk.corpus import stopwords
import string



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
    orig_data = df.iloc[idx].reset_index(drop=True)
    if task_name == "c2a":
        return orig_data[column_name].tolist()
    elif task_name == "c2c":
        return orig_data[column_name].tolist(), orig_data["is_dependent"].tolist()


def get_ngrams(text, n=4):
    # remove stopwords and punctuation
    text = [word for word in text.split() if word not in stopwords.words('english')]
    text = [word for word in text if word not in string.punctuation]

    ngrams = zip(*[text[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]


def get_metric_rankings(output1_data, output2_data, input_claims, task_name, metric_name, is_dependent_list=None):
    rankings = []
    if metric_name == "semsim-ipc":
        import torch
        from transformers import BertTokenizer
        from sentence_transformers import SentenceTransformer, util

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint_path = "../ipc_cls/output/checkpoint_epoch3"        
        model_name = "../ipc_cls/bert-for-patents/"
        # checkpoint_path = model_name

        tokenizer = BertTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(checkpoint_path)
        # model = BertForSequenceClassification.from_pretrained(checkpoint_path, num_labels=617)
        # model.to(device)
        # model = nn.DataParallel(model).cuda()
        model = SentenceTransformer(checkpoint_path, device=device)

        if task_name == "c2a":
            for claim, abstract1, abstract2 in tqdm(zip(input_claims, output1_data, output2_data), total=len(input_claims)):

                claim_embedding = model.encode("[claims] "+claim)
                abstract1_embedding = model.encode("[abstrct] "+abstract1)
                abstract2_embedding = model.encode("[abstract] "+abstract2)

                # compute cosine-similarities for each sentence with each other sentence
                cosine_scores1 = util.pytorch_cos_sim(claim_embedding, abstract1_embedding)[0]
                cosine_scores2 = util.pytorch_cos_sim(claim_embedding, abstract2_embedding)[0]

                if cosine_scores1 > cosine_scores2:
                    rankings.append([1, 2])
                elif cosine_scores1 < cosine_scores2:
                    rankings.append([2, 1])
                else:
                    rankings.append([1, 1])

        elif task_name == "c2c":
            from claim_rules import Rule_based_checker
            for claim, output_claim1, output_claim2, is_dependent in tqdm(zip(input_claims, output1_data, output2_data, is_dependent_list), total=len(input_claims)):
                # normlize by rule-based checker

                checker1 = Rule_based_checker(claim, output_claim1, required_dependent=is_dependent)
                checker2 = Rule_based_checker(claim, output_claim2, required_dependent=is_dependent)

                rule_based_score1 = checker1.score()
                rule_based_score2 = checker2.score()


                claim_embedding = model.encode("[claims] "+claim)

                output_claim1_embedding = model.encode("[claims] "+claim+"\n\n"+output_claim1)
                output_claim2_embedding = model.encode("[claims] "+claim+"\n\n"+output_claim2)

                # compute cosine-similarities for each sentence with each other sentence
                cosine_scores1 = util.pytorch_cos_sim(claim_embedding, output_claim1_embedding)[0] * rule_based_score1
                cosine_scores2 = util.pytorch_cos_sim(claim_embedding, output_claim2_embedding)[0] * rule_based_score2

                if cosine_scores1 > cosine_scores2:
                    rankings.append([1, 2])
                elif cosine_scores1 < cosine_scores2:
                    rankings.append([2, 1])
                else:
                    rankings.append([1, 1])

    elif task_name == "c2a" and metric_name == "term-overlap":
        import spacy
        from pyate.term_extraction_pipeline import TermExtractionPipeline

        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("combo_basic")

        # # keep only first claim for each input_claims
        # input_claims = [c.split("2.")[0].strip() for c in input_claims]

        for claim, abstract1, abstract2 in tqdm(zip(input_claims, output1_data, output2_data), total=len(input_claims)):
            terms_claim = nlp(claim)._.combo_basic.sort_values(ascending=False)
            terms_abstract1 = nlp(abstract1)._.combo_basic.sort_values(ascending=False)
            terms_abstract2 = nlp(abstract2)._.combo_basic.sort_values(ascending=False)

            # convert pandas series to dictinary, and keep those with value > 1.0
            terms_claim = [k for k, v in terms_claim.to_dict().items() if v > 1.0]
            terms_abstract1 = [k for k, v in terms_abstract1.to_dict().items() if v > 1.0]
            terms_abstract2 = [k for k, v in terms_abstract2.to_dict().items() if v > 1.0]

            # get the number of overlapping terms between claim and abstract
            try:
                overlap1_r = len(set(terms_claim).intersection(set(terms_abstract1))) / len(set(terms_claim))
            except ZeroDivisionError:
                overlap1_r = 0

            try:
                overlap2_r = len(set(terms_claim).intersection(set(terms_abstract2))) / len(set(terms_claim))
            except ZeroDivisionError:
                overlap2_r = 0

            try:
                overlap1_p = len(set(terms_claim).intersection(set(terms_abstract1))) / len(set(terms_abstract1))
            except ZeroDivisionError:
                overlap1_p = 0
            try:
                overlap2_p = len(set(terms_claim).intersection(set(terms_abstract2))) / len(set(terms_abstract2))
            except ZeroDivisionError:
                overlap2_p = 0

            try:
                overlap1 = 2 / (1/overlap1_p + 1/overlap1_r)
            except ZeroDivisionError:
                if overlap1_p == 0 and overlap1_r == 0:
                    overlap1 = 0
                elif overlap1_p == 0:
                    overlap1 = 2 / (1/0.0001 + 1/overlap1_r)
                elif overlap1_r == 0:
                    overlap1 = 2 / (1/overlap1_p + 1/0.0001)

            try:
                overlap2 = 2 / (1/overlap2_p + 1/overlap2_r)
            except ZeroDivisionError:
                if overlap2_p == 0 and overlap2_r == 0:
                    overlap2 = 0
                elif overlap2_p == 0:
                    overlap2 = 2 / (1/0.0001 + 1/overlap2_r)
                elif overlap2_r == 0:
                    overlap2 = 2 / (1/overlap2_p + 1/0.0001)

            if overlap1 > overlap2:
                rankings.append([1, 2])
            elif overlap1 < overlap2:
                rankings.append([2, 1])
            else:
                rankings.append([1, 1])

    elif task_name == "c2a" and metric_name == "ngram-overlap":
        # keep only first claim for each input_claims
        input_claims = [c.split("2.")[0].strip() for c in input_claims]

        for claim, abstract1, abstract2 in tqdm(zip(input_claims, output1_data, output2_data), total=len(input_claims)):
            claim_ngrams = get_ngrams(claim)
            abstract1_ngrams = get_ngrams(abstract1)
            abstract2_ngrams = get_ngrams(abstract2)

            # get the number of overlapping terms between claim and abstract
            overlap1_r = len(set(claim_ngrams).intersection(set(abstract1_ngrams))) / len(set(claim_ngrams))
            overlap2_r = len(set(claim_ngrams).intersection(set(abstract2_ngrams))) / len(set(claim_ngrams))

            overlap1_p = len(set(claim_ngrams).intersection(set(abstract1_ngrams))) / len(set(abstract1_ngrams))
            overlap2_p = len(set(claim_ngrams).intersection(set(abstract2_ngrams))) / len(set(abstract2_ngrams))

            try:
                overlap1 = 2 / (1/overlap1_p + 1/overlap1_r)
            except ZeroDivisionError:
                if overlap1_p == 0 and overlap1_r == 0:
                    overlap1 = 0
                elif overlap1_p == 0:
                    overlap1 = 2 / (1/0.0001 + 1/overlap1_r)
                elif overlap1_r == 0:
                    overlap1 = 2 / (1/overlap1_p + 1/0.0001)

            try:
                overlap2 = 2 / (1/overlap2_p + 1/overlap2_r)
            except ZeroDivisionError:
                if overlap2_p == 0 and overlap2_r == 0:
                    overlap2 = 0
                elif overlap2_p == 0:
                    overlap2 = 2 / (1/0.0001 + 1/overlap2_r)
                elif overlap2_r == 0:
                    overlap2 = 2 / (1/overlap2_p + 1/0.0001)

            if overlap1 > overlap2:
                rankings.append([1, 2])
            elif overlap1 < overlap2:
                rankings.append([2, 1])
            else:
                rankings.append([1, 1])

    elif task_name == "c2a" and metric_name == "qa-eval":
        from qafacteval import QAFactEval
        kwargs = {"cuda_device": 0, "use_lerc_quip": True, \
                "verbose": True, "generation_batch_size": 32, \
                "answering_batch_size": 32, "lerc_batch_size": 8}

        model_folder = "../QAFactEval/models" # path to models downloaded with download_models.sh
        metric = QAFactEval(
            lerc_quip_path=f"{model_folder}/quip-512-mocha",
            generation_model_path=f"{model_folder}/generation/model.tar.gz",
            answering_model_dir=f"{model_folder}/answering",
            lerc_model_path=f"{model_folder}/lerc/model.tar.gz",
            lerc_pretrained_model_path=f"{model_folder}/lerc/pretraining.tar.gz",
            **kwargs
        )

        input_claims = input_claims
        output1_data = [[o] for o in output1_data]
        output2_data = [[o] for o in output2_data]

        results1 = metric.score_batch_qafacteval(input_claims, output1_data, return_qa_pairs=True)
        results2 = metric.score_batch_qafacteval(input_claims, output2_data, return_qa_pairs=True)

        for result1, result2 in zip(results1, results2):
            score1 = result1[0]['qa-eval']['lerc_quip']
            score2 = result2[0]['qa-eval']['lerc_quip']

            if score1 > score2:
                rankings.append([1, 2])
            elif score1 < score2:
                rankings.append([2, 1])
            else:
                rankings.append([1, 1])

    elif task_name == "c2c" and metric_name == "claim_rules":
        from claim_rules import Rule_based_checker

        for claim, output_claim1, output_claim2, is_dependent in tqdm(zip(input_claims, output1_data, output2_data, is_dependent_list), total=len(input_claims)):
            checker1 = Rule_based_checker(claim, output_claim1, required_dependent=is_dependent)
            checker2 = Rule_based_checker(claim, output_claim2, required_dependent=is_dependent)

            if checker1.score() > checker2.score():
                rankings.append([1, 2])
            elif not checker1.score() < checker2.score():
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


    if task_name == "c2c":
        # read ground truths data 
        ground_truths_file = "../data/eval_data_c2c.csv"
        _, is_dependent_list = get_ground_truths(ground_truths_file, task_name)
        metric_rankings = get_metric_rankings(output1_data, output2_data, input_claims, task_name, metric_name, is_dependent_list)

    elif task_name == "c2a":
        # ground_truths_file = "../data/eval_data.csv"
        # ground_truths = get_ground_truths(ground_truths_file, task_name)
        metric_rankings = get_metric_rankings(output1_data, output2_data, input_claims, task_name, metric_name)

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
