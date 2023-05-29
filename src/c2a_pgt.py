"""
script for evaluating generation quality of claims => abstract for different models
"""

import torch

import argparse, os
import pandas as pd
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate



parser = argparse.ArgumentParser()
parser.add_argument('--path_data', type=str, default="./data/eval_data.csv")
parser.add_argument('--aspect', type=str, required=False, choices={"factuality", "coherence"})
parser.add_argument('--path_prediction', type=str, default="./predictions")


args = parser.parse_args()

if __name__ == '__main__':
    df = pd.read_csv(args.path_data)

    claims, actuals = df['claims'].to_list(), df['abstract'].to_list()

    tokenizer = AutoTokenizer.from_pretrained("christofid/pgt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained("christofid/pgt").to(device)

    # make prediction directory 
    path_prediction = args.path_prediction
    if not os.path.isdir(path_prediction):
        os.makedirs(path_prediction)
    path_output = os.path.join(path_prediction, 'pgt_abstract.pred')

    if os.path.exists(path_output) and os.path.getsize(path_output) > 0:
        df_res = pd.read_csv(path_output)
        predictions = df_res['abstract'].to_list()
    else:
        predictions = []
        for c in tqdm(claims):
            c = ' '.join(c.split(' ')[:180])
            input = f"{c} <|sep|> Given the above claims, suggest an abstract <|sep|>"
            text_encoded = tokenizer.encode(input, max_length = 512, return_tensors="pt").to(device)
            generated = model.generate(text_encoded, do_sample=True, top_k=5, num_return_sequences = 1, max_length=1024)
            generated_text = [tokenizer.decode(case).split("<|endoftext|>")[0].strip() for case in generated][0].split('<|sep|>')[-1]

            predictions.append(generated_text)
        df_res = pd.DataFrame({'abstract': predictions})
        df_res.to_csv(path_output, index=False)

    scores = []
    rouge = evaluate.load('rouge')
    scores.append(rouge.compute(predictions=predictions, references=[[act] for act in actuals])['rougeL'])
    
    print("rouge :" + str(scores))