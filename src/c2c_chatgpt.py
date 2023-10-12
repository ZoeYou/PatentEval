import pandas as pd
import argparse, os, re
import openai
import time, random
from tqdm import tqdm

from openai.error import *

openai.api_key = "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"

def generate_claim(claims, is_dependent, numberTries=0, gptChoice=0, maxsize=4097, exponential_base: float = 2, jitter: bool = True, max_retries: int = 5):
    if is_dependent:
        prompt = "Please assist me in drafting the next dependent claim based on the provided patent claims below. This dependent claim should be precise, legally sound, and in line with patent claim drafting conventions, using the existing claims as a basis for your draft.\n" \
            + "Claims: {claims}"
    else:
        prompt = "Please assist me in drafting the next independent claim based on the provided patent claims below. This independent claim should be precise, legally sound, and in line with patent claim drafting conventions, using the existing claims as a basis for your draft.\n" \
            + "Claims: {claims}"      

    if numberTries >= max_retries:
        print("Error: Retrying too many times! Please check the format of the input text!")
        return ""
    else:
        model_name = "gpt-3.5-turbo-0613"
        claims = " ".join(claims.split(" ")[:maxsize])
        content = prompt.format(claims=claims)
        maxsize = int(len(claims.split(" ")) // 2)
    
    try:
        completion = openai.ChatCompletion.create(
                model=model_name,
                temperature=0.0,
                messages=[
                # {"role": "system", "content": "You are a professional patent attorney."},
                {"role": "user", "content": content}]
                )
        return completion['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        if isinstance(e, RateLimitError):
            delay = exponential_base * (1 + jitter * random.random())
            time.sleep(delay)
            return generate_claim(claims, is_dependent, gptChoice=gptChoice, numberTries=numberTries+1, maxsize=maxsize)
        else:
            return generate_claim(claims, is_dependent, gptChoice=gptChoice+1, numberTries=numberTries+1, maxsize=maxsize)



parser = argparse.ArgumentParser()
parser.add_argument('--path_data', type=str, default="./data/eval_data_c2c.csv")
parser.add_argument('--path_prediction', type=str, default="./predictions")

args = parser.parse_args()

if __name__ == '__main__':
    df = pd.read_csv(args.path_data)
    input_claims = df['input_claims'].fillna('').to_list()
    dependencies = df['is_dependent'].fillna(True).to_list()

    # make prediction directory 
    path_prediction = args.path_prediction
    if not os.path.isdir(path_prediction):
        os.makedirs(path_prediction)
    path_output = os.path.join(path_prediction, 'chatgpt_claim.pred')

    predictions = []
    for c, d in tqdm(zip(input_claims, dependencies)):
        if len(c) < 10: 
            predictions.append("")  # given empty claim
            continue
        
        pred = generate_claim(c, d)
        predictions.append(pred)

    pattern_claim = '\d+\. (?!\(canceled\))'
    for i, pred in enumerate(predictions):
        numberings = re.findall(pattern_claim, pred)
        claims_split = [c.strip() for c in re.split(pattern_claim, pred) if c]
        if len(numberings)>1:
            predictions[i] = numberings[0]+claims_split[0]

    df_res = pd.DataFrame({'output_claim': predictions})
    df_res.to_csv(path_output, index=False)
