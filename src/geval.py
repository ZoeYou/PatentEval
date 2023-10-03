"""
script for evaluating customized aspects using GPT4
"""
import pandas as pd
import argparse
import openai
import json, os, requests, re
from tqdm import tqdm
import time

API_KEY = "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"
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

openai.api_key = API_KEY

def generate_chat_completion(messages, n, model="gpt-4-0613"):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    tries = 0
    while tries < 2:
        try:
            _response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=1,
                max_tokens=None,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                # logprobs=40,
                n=n
            )
            time.sleep(0.5)

            all_responses = [_response['choices'][i]['message']['content'] for i in
                                range(len(_response['choices']))]
            return all_responses

        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(60)
            else:
                time.sleep(5)
                print('ignored', messages)
        tries += 1
    return




def eval_relevance(claims, abstract, n):
    messages = [
        {"role": "system", "content": "You are a professional patent practitioner tasked with evaluating an AI system that generates patent abstracts based on patent claims."},
        {"role": "user", "content": f"Your task involves assessing the relevance of the abstract generated by the system. Please rate the relevance on a scale from 1 to 5, where 1 means 'Not at all relevant' and 5 means 'Highly relevant'.\n" + \
         'Here are the claims and the generated abstract:\n' + \
         f'Claims: {claims}\n' + \
         f'Generated Abstract: {abstract}\n' + \
         'On a scale from 1 to 5, how relevant is the generated abstract to the provided claims? Please provide a brief explanation for your rating.'
         }]

    response_text = generate_chat_completion(messages, n)

    cnt, nb_words = 0, 3000
    while (not response_text) and cnt <= 5:
        claims = ' '.join(claims.split(' ')[:nb_words])
        messages = [
            {"role": "system", "content": "You are a professional patent practitioner tasked with evaluating an AI system that generates patent abstracts based on patent claims."},
            {"role": "user", "content": f"Your task involves assessing the relevance of the abstract generated by the system. Please rate the relevance on a scale from 1 to 5, where 1 means 'Not at all relevant' and 5 means 'Highly relevant'.\n" + \
            'Here are the claims and the generated abstract:\n' + \
            f'Claims: {claims}\n' + \
            f'Generated Abstract: {abstract}\n' + \
            'On a scale from 1 to 5, how relevant is the generated abstract to the provided claims? Please provide a brief explanation for your rating.'
            }]
        response_text = generate_chat_completion(messages, n)
        cnt += 1
        nb_words = nb_words//2

    if not response_text : response_text = ['Rating: 3' for _ in range(n)]

    return response_text

def eval_fact(claims, abstract, n):
    
    messages = [
        {"role": "system", "content": "You are a professional patent practitioner tasked with evaluating an AI system that generates patent abstracts based on patent claims."},
        {"role": "user", "content": f"Your task involves assessing the factual consistency of the abstract generated by the system. Specifically, check if the abstract preserves the core essence of the source claims without introducing new information or distorting the facts presented.\n" + \
         'Here are the claims and the generated abstract:\n' + \
         f'Claims: {claims}\n' + \
         f'Generated Abstract: {abstract}\n' + \
         "Does the generated abstract contain any factual inconsistencies when compared to the original claims? Please rate the factual consistency on a scale from 1 to 5, where 1 means 'Significant inconsistencies' and 5 means 'No inconsistencies'. Please provide a brief explanation for your rating."
         }]

    response_text = generate_chat_completion(messages, n)

    cnt, nb_words = 0, 3000
    while (not response_text) and cnt <= 5:
        claims = ' '.join(claims.split(' ')[:nb_words])
        messages = [
            {"role": "system", "content": "You are a professional patent practitioner tasked with evaluating an AI system that generates patent abstracts based on patent claims."},
            {"role": "user", "content": f"Your task involves assessing the factual consistency of the abstract generated by the system. Specifically, check if the abstract preserves the core essence of the source claims without introducing new information or distorting the facts presented.\n" + \
            'Here are the claims and the generated abstract:\n' + \
            f'Claims: {claims}\n' + \
            f'Generated Abstract: {abstract}\n' + \
            "Does the generated abstract contain any factual inconsistencies when compared to the original claims? Please rate the factual consistency on a scale from 1 to 5, where 1 means 'Significant inconsistencies' and 5 means 'No inconsistencies'. Please provide a brief explanation for your rating."
            }]
        response_text = generate_chat_completion(messages, n)
        cnt += 1
        nb_words = nb_words//2

    if not response_text : response_text = ['Rating: 3' for _ in range(n)]

    return response_text


def eval_coherence(claims, generated_claim, n):
    
    messages = [
        {"role": "system", "content": "You are a professional patent practitioner tasked with evaluating an AI system that generates subsequent patent claims based on existing ones."},
        {"role": "user", "content": f"Your task involves assessing the coherence of the newly generated claim. Specifically, evaluate whether the generated claim logically follows from the previous claims and maintains consistency in technical details and terminology.\n" + \
         'Here are the original claims and the AI-generated subsequent claim:\n' + \
         f'Original Claims: {claims}\n' + \
         f'Generated Claim:: {generated_claim}\n' + \
         "Please rate the coherence of the generated claim in relation to the original claims on a scale from 1 to 5, where 1 means 'Not at all coherent' and 5 means 'Highly coherent'. Please provide a brief explanation for your rating."
         }]

    response_text = generate_chat_completion(messages, n)

    cnt, nb_words = 0, 3000
    while (not response_text) and cnt <= 5:
        claims = ' '.join(claims.split(' ')[:nb_words])
        messages = [
            {"role": "system", "content": "You are a professional patent practitioner tasked with evaluating an AI system that generates subsequent patent claims based on existing ones."},
            {"role": "user", "content": f"Your task involves assessing the coherence of the newly generated claim. Specifically, evaluate whether the generated claim logically follows from the previous claims and maintains consistency in technical details and terminology.\n" + \
            'Here are the original claims and the AI-generated subsequent claim:\n' + \
            f'Original Claims: {claims}\n' + \
            f'Generated Claim:: {generated_claim}\n' + \
            "Please rate the coherence of the generated claim in relation to the original claims on a scale from 1 to 5, where 1 means 'Not at all coherent' and 5 means 'Highly coherent'. Please provide a brief explanation for your rating."
            }]
        response_text = generate_chat_completion(messages, n)
        cnt += 1
        nb_words = nb_words//2

    if not response_text : response_text = ['Rating: 3' for _ in range(n)]

    return response_text


def calculate_weighted_score(evaluations):
    scores = []
    for eval in evaluations:
        try:
            scores.append(re.search('[12345]\.?\d?', eval).group())
        except AttributeError:
            scores.append(score)

    # calculate weighted score, (weights are defined by the frequency of each score)
    weighted_score = 0
    for score in scores:
        weighted_score += float(score)
    weighted_score /= len(scores)

    return weighted_score, scores


parser = argparse.ArgumentParser()
parser.add_argument('--path_data', type=str, default="./data/eval_data.csv")
parser.add_argument('--path_prediction', required=True, type=str)
parser.add_argument('--aspect', type=str, required=False, choices={"factuality", "relevance", "coherence"})
parser.add_argument('--path_evalution', type=str, default="./evaluations")
parser.add_argument('--n', type=int, default=20)

args = parser.parse_args()

if __name__ == '__main__':
    df = pd.read_csv(args.path_data)

    if args.path_data == "./data/eval_data.csv":
        abstracts, claims = df['abstract'].to_list(), df['claims'].to_list()
        predictions = pd.read_csv(args.path_prediction)['abstract'].fillna('').to_list()
    elif args.path_data == "./data/eval_data_c2c.csv":
        claims = df['input_claims'].to_list()
        predictions = pd.read_csv(args.path_prediction)['output_claim'].fillna('').to_list()
        

    evaluations, scores, weighted_scores = [], [], []
    if args.aspect == 'factuality':
        for pred, c in tqdm(zip(predictions, claims), total=len(claims)):
            evals = eval_fact(c, pred, args.n)
            weighted_score, scores = calculate_weighted_score(evals)

            evaluations.append(evals)
            scores.append(scores)
            weighted_scores.append(weighted_score)
           
    elif args.aspect == 'relevance':
        for pred, c in tqdm(zip(predictions, claims), total=len(claims)):
            evals = eval_relevance(c, pred, args.n)
            weighted_score, scores = calculate_weighted_score(evals)

            evaluations.append(evals)
            scores.append(scores)
            weighted_scores.append(weighted_score)

    elif args.aspect == 'coherence':
        for pred, c in tqdm(zip(predictions, claims), total=len(claims)):
            evals = eval_coherence(c, pred, args.n)
            weighted_score, scores = calculate_weighted_score(evals)

            evaluations.append(evals)
            scores.append(scores)
            weighted_scores.append(weighted_score)

    print(args.path_prediction, args.aspect, sum(weighted_scores)/len(weighted_scores))
    print(args.aspect + " : ", sum(weighted_scores)/len(weighted_scores))

    # make output evaluation directory 
    path_eval = args.path_evalution
    if not os.path.isdir(path_eval):
        os.makedirs(path_eval)
    method = args.path_prediction.split('/')[-1].split('.')[0]
    path_output = os.path.join(path_eval, f'{method}_{args.aspect}.eval')
    if args.path_data == "./data/eval_data.csv":
        eval_df = pd.DataFrame({"orig_claims": claims, "orig_abstract": abstracts, "generated_abstract": predictions, "eval": evaluations, "label": weighted_scores})
    elif args.path_data == "./data/eval_data_c2c.csv":
        eval_df = pd.DataFrame({"orig_claims": claims, "generated_claim": predictions, "eval": evaluations, "label": scores})
   
    eval_df.to_csv(path_output, index=False)
