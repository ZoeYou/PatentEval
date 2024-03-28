from scipy.stats import kendalltau
import argparse
import json
import pandas as pd

import csv
csv.field_size_limit(100000000)

from tqdm import tqdm

import string
import sys, os


# def get_rank(annotation_list):
#     res = []
#     for line in annotation_list:
#         if line["type"] == "pairwise":
#             if line["value"]["selected"] == "left":
#                 res.append(1)
#                 res.append(2)
#             elif line["value"]["selected"] == "right": 
#                 res.append(2)
#                 res.append(1)
#     if res == []:   # if no annotation, return 1, 1 (draw)
#         res = [1, 1]
#     return res


def read_pairs(annotation_file):
    """
    read the selected pairs information from jsonl
    """
    with open(annotation_file) as f:  annotations = json.load(f)

    human_rankings, input1_data, input2_data, input_claims = [], [], [], []
    for annot in annotations:
        input1_data.append(annot["output1"])
        input2_data.append(annot["output2"])
        input_claims.append(annot["input_claim"])

        preference = annot["comparison"][0]["selected"] if "comparison" in annot and "selected" in annot["comparison"][0] else "draw"
        if preference == "left":
            human_rankings.append([1, 2])
        elif preference == "right":
            human_rankings.append([2, 1])
        else:
            human_rankings.append([1, 1])

        # selected_annotations = []
        # input1_data.append(annot["data"]["output1"])
        # input2_data.append(annot["data"]["output2"])
        # input_claims.append(annot["data"]["input_claim"])

        # for la in annot["annotations"]:
        #     if "zuo" in la["completed_by"]["email"].lower():
        #         selected_annotations.append(la)

        # if len(selected_annotations) > 0:
        #     annot["annotations"] = [d["result"] for d in selected_annotations][0]
        #     annot["annotations"] = [a for a in annot["annotations"] if a["type"] != "labels"]
        #     annotation_lines.append(annot["annotations"])
        # else: # skipped or emply annotations
        #     annot["annotations"] = []
        #     annotation_lines.append(annot["annotations"])

    # # get ranking for each pair
    # human_rankings = []
    # for line in annotation_lines:
    #     human_rankings.append(get_rank(line))
    return input1_data, input2_data, input_claims, human_rankings
  

def get_ground_truths(ground_truths_file, task_name):
    df = pd.read_csv(ground_truths_file)

    if task_name == "c2a":
        idx = df[(df["domain"] == "G") | (df["domain"] == "A")].index.tolist()
        column_name = "abstract"
    elif task_name == "c2c":
        idx = df[(df["domain"] == "G") | (df["domain"] == "A") | (df["is_dependent"] == False)].index.tolist()
        column_name = "true_next_claim"

    # repeat each index two times to match the number of pairs
    idx = [x for x in idx for i in range(2)]
    orig_data = df.iloc[idx].reset_index(drop=True)
    if task_name == "c2a":
        return orig_data[column_name].tolist()
    elif task_name == "c2c":
        return orig_data[column_name].tolist(), orig_data["is_dependent"].tolist(), orig_data["domain"].tolist()


def get_ngrams(text, n=4):
    """
    Get n-grams from text
    """
    from nltk.corpus import stopwords
    
    # remove stopwords and punctuation
    text = [word for word in text.split() if word not in stopwords.words('english')]
    text = [word for word in text if word not in string.punctuation]

    ngrams = zip(*[text[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]


def get_metric_rankings(output1_data, output2_data, input_claims, task_name, metric_name, is_dependent_list=None, domains_list=None):
    
    rankings = []
    if metric_name == "semsim-ipc":
        import torch
        from transformers import BertTokenizer
        from sentence_transformers import SentenceTransformer, util

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint_path = "./ipc_cls/output/checkpoint_epoch3"        
        model_path = "./ipc_cls/bert-for-patents/"

        tokenizer = BertTokenizer.from_pretrained(model_path)
        tokenizer.save_pretrained(checkpoint_path)

        model = SentenceTransformer(checkpoint_path, device=device) # use mean pooling as default

        if task_name == "c2a":
            for claim, abstract1, abstract2 in tqdm(zip(input_claims, output1_data, output2_data), total=len(input_claims)):

                claim_embedding = model.encode("[claims] "+claim)
                abstract1_embedding = model.encode("[abstract] "+abstract1)
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
            for input_claims, output_claim1, output_claim2, is_dependent in tqdm(zip(input_claims, output1_data, output2_data, is_dependent_list), total=len(input_claims)):
                
                # normlize firstly by rule-based checker
                checker1 = Rule_based_checker(input_claims, output_claim1, required_dependent=is_dependent)
                checker2 = Rule_based_checker(input_claims, output_claim2, required_dependent=is_dependent)

                rule_based_score1 = checker1.score()
                rule_based_score2 = checker2.score()

                claim_embedding = model.encode("[claims] "+input_claims)
                output_claim1_embedding = model.encode("[claims] "+input_claims+"\n"+output_claim1)   # TODO output claims need to be normalized !
                output_claim2_embedding = model.encode("[claims] "+input_claims+"\n"+output_claim2)

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
        from nltk.corpus import stopwords

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
        sys.path.append(os.path.abspath("./QAFactEval"))
        from qafacteval import QAFactEval
        kwargs = {"cuda_device": 0, "use_lerc_quip": True, \
                "verbose": True, "generation_batch_size": 32, \
                "answering_batch_size": 32, "lerc_batch_size": 8}

        model_folder = "./QAFactEval/models" # path to models downloaded with download_models.sh
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


    elif task_name == "c2a" and metric_name.lower() == "factgraph": 
        model_dir = "./FactGraph/fact-graph/checkpoints/factgraph/"
        model_type = "factgraph"
        encoder_model_name = "google/electra-base-discriminator"

        if not os.path.exists("./eval_dataset1.processed") or not os.path.exists("./eval_dataset2.processed"):
            sys.path.append(os.path.abspath("./FactGraph/fact-graph/data/preprocess"))
            from preprocess_evaluate import extract_sents, extract_amrs, process_amr, save_data
            import torch

            num_gpus = torch.cuda.device_count()
            cuda_devices = ','.join(str(i) for i in range(num_gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            def preprocess_data(data, file, num_sents=5):
                torch.cuda.empty_cache()
                data = extract_sents(data, num_sents)
                amr_data, amr_data_summaries = extract_amrs(data)
                process_amr(data, amr_data, amr_data_summaries, num_sents)
                save_data(data, file)

            # create dataset 
            eval_dataset1 = [{"summary":output1, "article": input} for input, output1 in zip(input_claims, output1_data)]
            eval_dataset2 = [{"summary":output2, "article": input} for input, output2 in zip(input_claims, output2_data)]
            preprocess_data(eval_dataset1, "eval_dataset1")
            preprocess_data(eval_dataset2, "eval_dataset2")

        # module load cuda/11.4.0
        sys.path.append(os.path.abspath("./FactGraph/fact-graph/src"))
        from evaluate import test, load_data, load_model_and_tokenizer, preprocess_factgraph, preprocess_factgraph_edge

        model, tokenizer = load_model_and_tokenizer(model_type, encoder_model_name, 2, 32)

        test_data1 = load_data("./eval_dataset1.processed", model_type, tokenizer, num_document_graphs=5)
        test_data2 = load_data("./eval_dataset2.processed", model_type, tokenizer, num_document_graphs=5)

        results1 = test(model_dir, model_type, test_data1, num_document_graphs=5)['prob_pos'].tolist()
        results2 = test(model_dir, model_type, test_data2, num_document_graphs=5)['prob_pos'].tolist()

        for result1, result2 in zip(results1, results2):
            if result1 > result2:
                rankings.append([1, 2])
            elif result1 < result2:
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


    elif task_name == "c2c" and metric_name == "entity_coherence":
        import pickle
        from nltk import sent_tokenize

        cur_dir = "./coheoka/coheoka"
        sys.path.append(cur_dir)
        from coherence_probability import ProbabilityVector

        class Assessment(object):
            def __init__(self, pv):
                assert type(pv) == ProbabilityVector
                self.pv = pv

            def assess_pv(self, text):
                sents = [sent for sent in sent_tokenize(text) if len(sent) > 5]
                # if len(sents) <= 1:
                #     return -1
                pb = self.pv.evaluate_coherence(text)[0]
                return pb

        # load pre-fitted models
        pv_A = pickle.load(open(os.path.join(cur_dir, 'pickles', 'pv_A.pkl'), 'rb'))
        pv_G = pickle.load(open(os.path.join(cur_dir, 'pickles', 'pv_G.pkl'), 'rb'))

        assess_A = Assessment(pv_A)
        assess_G = Assessment(pv_G)

        for input_claims, output_claim1, output_claim2, is_dependent, domain in tqdm(zip(input_claims, output1_data, output2_data, is_dependent_list, domains_list), total=len(input_claims)):
         
            output_claim1 = input_claims + "\n" + output_claim1
            output_claim2 = input_claims + "\n" + output_claim2
            if domain == "A":
                score1 = assess_A.assess_pv(output_claim1) - assess_A.assess_pv(input_claims)
                score2 = assess_A.assess_pv(output_claim2) - assess_A.assess_pv(input_claims)
            else:   # domain == "G" by default
                score1 = assess_G.assess_pv(output_claim1) - assess_G.assess_pv(input_claims)
                score2 = assess_G.assess_pv(output_claim2) - assess_G.assess_pv(input_claims)

            if score1 > score2:
                rankings.append([1, 2])
            elif score1 < score2:
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

    annotation_file = f"./human_eval/annotations/final_{task_name}.json"
    output1_data, output2_data, input_claims, human_rankings = read_pairs(annotation_file)


    if task_name == "c2c":
        # read ground truths data 
        ground_truths_file = "./data/eval_data_c2c.csv"
        _, is_dependent_list, domains_list = get_ground_truths(ground_truths_file, task_name)
        metric_rankings = get_metric_rankings(output1_data, output2_data, input_claims, task_name, metric_name, is_dependent_list, domains_list)

    elif task_name == "c2a":
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
