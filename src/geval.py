"""
script for evaluating customized aspects using GPT4
"""
import pandas as pd
import argparse
import openai
import json, os, requests
from tqdm import tqdm
import evaluate
import time

rouge = evaluate.load('rouge')


def generate_chat_completion(messages, model="gpt-3.5-turbo", temperature=0, nb_tries=0, max_tokens=None):
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

    if response.status_code == 200: # successful request
        return response.json()["choices"][0]["message"]["content"]
    elif response.status_code == 400:
        print(f"Error {response.status_code}: {response.text}")
        return response.status_code
    elif response.status_code in [429, 502]:
        print(f"Error {response.status_code}: {response.text}")
        time.sleep(30)
        return generate_chat_completion(messages)
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")


def eval_relevance(claims, abstract):
    
    messages = [
        {"role": "system", "content": "You are a professional patent practitioner."},
        {"role": "user", "content": f'Human Evaluation of patent claims to abstact generation system: ' + \
         'Coherence: How well is the generated abstract relevant to its source claims?\n' + \
         f'Claims: {claims}\n' + \
         f'Abstract: {abstract}\n' + \
         'Does the generated abstract coherent to the claims?'
         }]

    response_text = generate_chat_completion(messages)

    cnt, nb_words = 0, 3000
    while response_text == 400 and cnt <= 5:
        claims = ' '.join(claims.split(' ')[:nb_words])
        messages = [
            {"role": "system", "content": "You are a professional patent practitioner."},
            {"role": "user", "content": f'Human Evaluation of patent claims to abstact generation system: ' + \
            'Coherence: How well is the generated abstract relevant to its source claims?\n' + \
            f'Claims: {claims}\n' + \
            f'Abstract: {abstract}\n' + \
            'Does the generated abstract coherent to the claims?'
            }]
        response_text = generate_chat_completion(messages)
        cnt += 1
        nb_words = nb_words//2

    if response_text == 400: response_text = 'no,'

    return response_text

def eval_fact(claims, abstract):
    messages = [
        {"role": "system", "content": "You are a professional patent practitioner."},
        {"role": "user", "content": f'Human Evaluation of patent claims to abstact generation system: ' + \
         'Factual Consistency: Does the generated abstract preserve the factual statements of the source claims?\n' + \
         f'Claims: {claims}\n' + \
         f'Abstract: {abstract}\n' + \
         'Does the generated abstract contain factual inconsistency?'
         }]

    response_text = generate_chat_completion(messages)

    cnt, nb_words = 0, 3000
    while response_text == 400 and cnt <= 5:
        claims = ' '.join(claims.split(' ')[:nb_words])
        messages = [
            {"role": "system", "content": "You are a professional patent practitioner."},
            {"role": "user", "content": f'Human Evaluation of patent claims to abstact generation system: ' + \
            'Factual Consistency: Does the generated abstract preserve the factual statements of the source claims?\n' + \
            f'Claims: {claims}\n' + \
            f'Abstract: {abstract}\n' + \
            'Does the generated abstract contain factual inconsistency?'
            }]
        response_text = generate_chat_completion(messages)
        cnt += 1
        nb_words = nb_words//2

    if response_text == 400: response_text = 'yes,'

    return response_text

API_KEY = "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"
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

parser = argparse.ArgumentParser()
parser.add_argument('--path_data', type=str, default="./data/eval_data.csv")
parser.add_argument('--path_prediction', required=True, type=str)
parser.add_argument('--aspect', type=str, required=False, choices={"factuality", "relevance"})
parser.add_argument('--path_evalution', type=str, default="./evaluations")

args = parser.parse_args()

if __name__ == '__main__':
    df = pd.read_csv(args.path_data)

    abstracts, claims = df['abstract'].to_list(), df['claims'].to_list()
    predictions = pd.read_csv(args.path_prediction)['abstract'].to_list()

    evaluations, scores = [], []
    if args.aspect == 'factuality':
        for pred, c in tqdm(zip(predictions, claims), total=len(claims)):
            eval = eval_fact(c, pred)
            if eval.lower().startswith("no,"):
                evaluations.append(eval)     
                scores.append(1)  
            else:
                evaluations.append(eval)
                scores.append(0)
           
    elif args.aspect == 'relevance':
        for pred, c in tqdm(zip(predictions, claims), total=len(claims)):
            eval = eval_relevance(c, pred)
            if eval.lower().startswith('yes,'):
                evaluations.append(eval)
                scores.append(1)
            else:
                evaluations.append(eval)
                scores.append(0)

    print(args.aspect, sum(scores)/len(scores))

    # make output evaluation directory 
    path_eval = args.path_evalution
    if not os.path.isdir(path_eval):
        os.makedirs(path_eval)
    method = args.path_prediction.split('/')[-1].split('.')[0]
    path_output = os.path.join(path_eval, f'{method}_{args.aspect}.eval')
    eval_df = pd.DataFrame({"orig_claims": claims, "orig_abstract": abstracts, "generated_abstract": predictions, "eval": evaluations, "label": scores})
    eval_df.to_csv(path_output, index=False)
