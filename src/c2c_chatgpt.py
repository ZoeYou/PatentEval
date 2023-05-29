"""
script for evaluating generation quality of next claim generation
"""
import pandas as pd
import argparse, os
import json, time
import requests
from tqdm import tqdm
import re



def generate_chat_completion(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens is not None:
        data["max_tokens"] = max_tokens

    response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    elif response.status_code == 400:
        print(f"Error {response.status_code}: {response.text}")
        return response.status_code
    elif response.status_code == 429:
        time.sleep(30)
        return generate_chat_completion(messages)   
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")


def generate_claim(claims):
    messages = [
        {"role": "user", "content": f"Based on the provided patent claims below, please draft the subsequent claim for a continuation. This claim, which may be either dependent or independent, should be precise, legally sound, and in line with patent claim drafting conventions. Use the existing claims as a basis for your draft.\n" \
        + f'Claims: {claims}'}]
    response_text = generate_chat_completion(messages)

    cnt, nb_words = 0, 3000
    while response_text == 400 and cnt <= 5:
        claims = ' '.join(claims.split(' ')[:nb_words])
        messages = [
            {"role": "user", "content": f"Based on the provided patent claims below, please draft the subsequent claim for a continuation. This claim, which may be either dependent or independent, should be precise, legally sound, and in line with patent claim drafting conventions. Use the existing claims as a basis for your draft.\n" \
            + f'Claims: {claims}'}]

        response_text = generate_chat_completion(messages)
        cnt += 1
        nb_words = nb_words//2

    if response_text == 400: response_text = ' '
    return response_text

API_KEY = "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"
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

parser = argparse.ArgumentParser()
parser.add_argument('--path_data', type=str, default="./data/eval_data_c2c.csv")
parser.add_argument('--path_prediction', type=str, default="./predictions")


args = parser.parse_args()

if __name__ == '__main__':
    df = pd.read_csv(args.path_data)

    input_claims = df['input_claims'].to_list()

    # make prediction directory 
    path_prediction = args.path_prediction
    if not os.path.isdir(path_prediction):
        os.makedirs(path_prediction)
    path_output = os.path.join(path_prediction, 'chatgpt_claim.pred')

    predictions = []
    for c in tqdm(input_claims):
        pred = generate_claim(c)
        predictions.append(pred)

    pattern_claim = '\d+\. (?!\(canceled\))'
    for i, pred in enumerate(predictions):
        numberings = re.findall(pattern_claim, pred)
        claims_split = [c.strip() for c in re.split(pattern_claim, pred) if c]
        if len(numberings)>1:
            predictions[i] = numberings[0]+claims_split[0]

    df_res = pd.DataFrame({'output_claim': predictions})
    df_res.to_csv(path_output, index=False)
