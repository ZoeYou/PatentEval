"""
script for evaluating generation quality of claims => abstract for different models
conda activate pt_env
"""
import argparse, os, requests, time, json
import pandas as pd
from tqdm import tqdm

from transformers import pipeline
import evaluate

API_KEY = "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"
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

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
        messages[1]['content'] = ' '.join(messages[1]['content'].split(' ')[:4000])
        return generate_chat_completion(messages)
    else:
        time.sleep(5)
        # raise Exception(f"Error {response.status_code}: {response.text}")
        return "No,"
    
def eval_coherence(claims, abstract):
    messages = [
        {"role": "system", "content": "You are a professional patent practitioner."},
        {"role": "user", "content": f'Human Evaluation of patent claims to abstact generation system: ' + \
         'Coherence: How well is the generated abstract relevant to its source claims?\n' + \
         f'Claims: {claims}\n' + \
         f'Abstract: {abstract}\n' + \
         'Does the generated abstract coherent to the claims?'
         }]

    response_text = generate_chat_completion(messages)
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
    return response_text



parser = argparse.ArgumentParser()
parser.add_argument('--path_data', type=str, default="./data/eval_data.csv")
parser.add_argument('--aspect', type=str, required=False, choices={"factuality", "coherence"})
parser.add_argument('--path_prediction', type=str, default="./predictions")

args = parser.parse_args()

if __name__ == '__main__':
    df = pd.read_csv(args.path_data)
    actuals, inputs = df['abstract'].to_list(), df['claims'].to_list()

    summarizer = pipeline(task="summarization", model="HUPD/hupd-t5-small", device=-1)

    # make prediction directory 
    path_prediction = args.path_prediction
    if not os.path.isdir(path_prediction):
        os.makedirs(path_prediction)
    path_output = os.path.join(path_prediction, 'hupd_abstract.pred')

    if os.path.exists(path_output) and os.path.getsize(path_output) > 0:
        df_res = pd.read_csv(path_output)
        predictions = df_res['abstract'].to_list()
    else:
        predictions = []
        for claims in tqdm(inputs):
            pred = summarizer(claims)[0]['summary_text']
            predictions.append(pred)
        df_res = pd.DataFrame({'abstract': predictions})
        df_res.to_csv(path_output, index=False)


    scores = []
    rouge = evaluate.load('rouge')
    scores.append(rouge.compute(predictions=predictions, references=[[act] for act in actuals])['rougeL'])

    print("rouge :" + str(scores))


    
