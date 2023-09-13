import torch

import argparse, os
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--path_data', type=str, default="./data/eval_data.csv")
parser.add_argument('--path_prediction', type=str, default="./predictions")

args = parser.parse_args()

if __name__ == '__main__':
    df = pd.read_csv(args.path_data)
    claims_list, actuals = df['claims'].to_list(), df['abstract'].to_list()

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
        def get_abstract(claims, maxsize=256, numberTries=0, max_retries=5):
            if numberTries >= max_retries:
                print("Error: Retrying too many times! Please check the format of the input text!")
                return ""
            else:
                claims = ' '.join(claims.split(' ')[:maxsize])
                maxsize = int(len(claims.split(" ")) // 2)
                try:
                    input = f"{claims} <|sep|> Given the above claims, suggest an abstract <|sep|>"
                    text_encoded = tokenizer.encode(input, max_length = 1024, return_tensors="pt").to(device)
                    generated = model.generate(text_encoded, do_sample=True, top_k=50, top_p=0.95, max_length=1024)
                    generated_text = [tokenizer.decode(case).split("<|endoftext|>")[0].strip() for case in generated][0].split('Given the above claims, suggest an abstract <|sep|>')[-1]
                    return generated_text
                except:
                    return get_abstract(claims, numberTries=numberTries+1, maxsize=maxsize)

        predictions = []
        for claims in tqdm(claims_list):
            generated_text = get_abstract(claims)
            predictions.append(generated_text)
        df_res = pd.DataFrame({'abstract': predictions})
        df_res.to_csv(path_output, index=False)

    scores = []
    rouge = evaluate.load('rouge')
    scores.append(rouge.compute(predictions=predictions, references=[[act] for act in actuals])['rougeL'])
    
    print("rouge :" + str(scores))