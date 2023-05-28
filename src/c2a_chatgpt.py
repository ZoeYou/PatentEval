"""
script for evaluating generation quality of claism => abstract for different models
"""
import pandas as pd
import argparse, os
import openai
import json, time, random
import requests
from tqdm import tqdm
import evaluate

rouge = evaluate.load('rouge')


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


def generate_abstract(claims):
    messages = [
        {"role": "user", "content": f"Please draft a patent abstract from the provided claims. The abstract should concisely summarize the technical disclosure, enabling any reader to quickly understand the subject matter. You'll be given an example to guide your creation.\n" \
         + 'Claims: 1. A liquid ejecting apparatus comprising: a carriage which is configured to move in a reciprocating direction, wherein the carriage is provided with an ejecting unit which ejects liquid, a circuit board which inputs a signal to the ejecting unit, and a fan which is configured to blow air toward the circuit board, and wherein the carriage includes a separator which separates a path through which mist which occurs along with ejecting of the liquid from the ejecting unit reaches the fan, wherein the separator is provided in the carriage so that at least a part thereof is arranged between the ejecting unit and the fan, when viewed in a direction intersecting the ejecting direction of the liquid. 2. The liquid ejecting apparatus according to claim 1, wherein at least ejection ports of the ejecting unit are below a portion of the separator in the liquid ejection direction and the fan is above the portion. 3. The liquid ejecting apparatus according to claim 1, wherein the separator includes convection generation units which are formed at both ends in the reciprocating direction in a shape in which at least a part thereof projects in the reciprocating direction. 4. The liquid ejecting apparatus according to claim 3, wherein the convection generation unit is provided so as to extend in a direction intersecting the ejecting direction of the liquid and the reciprocating direction. 5. The liquid ejecting apparatus according to claim 1, further comprising: a rail which is provided so as to extend in the reciprocating direction, wherein the carriage includes a first unit which is supported by the rail, and a second unit which is configured to move along the ejecting direction of the liquid with respect to the first unit, and wherein the ejecting unit and the separator are provided in the second unit.\n' \
         + 'Abstract: A liquid ejecting apparatus which includes a carriage which can move in a reciprocating direction, in which the carriage is provided with an ejecting unit which ejects liquid, a circuit board which inputs a signal to the ejecting unit, and a fan which can blow air toward the circuit board, and includes a separator which separates a path through which mist which occurs along with ejecting of liquid from the ejecting unit reaches the fan.\n' \
         + f'Claims: {claims}\nAbstract: '}]
    response_text = generate_chat_completion(messages)

    cnt, nb_words = 0, 3000
    while response_text == 400 and cnt <= 5:
        claims = ' '.join(claims.split(' ')[:nb_words])
        messages = [
            {"role": "user", "content": f"Please draft a patent abstract from the provided claims. The abstract should concisely summarize the technical disclosure, enabling any reader to quickly understand the subject matter. You'll be given an example to guide your creation.\n" \
            + 'Claims: 1. A liquid ejecting apparatus comprising: a carriage which is configured to move in a reciprocating direction, wherein the carriage is provided with an ejecting unit which ejects liquid, a circuit board which inputs a signal to the ejecting unit, and a fan which is configured to blow air toward the circuit board, and wherein the carriage includes a separator which separates a path through which mist which occurs along with ejecting of the liquid from the ejecting unit reaches the fan, wherein the separator is provided in the carriage so that at least a part thereof is arranged between the ejecting unit and the fan, when viewed in a direction intersecting the ejecting direction of the liquid. 2. The liquid ejecting apparatus according to claim 1, wherein at least ejection ports of the ejecting unit are below a portion of the separator in the liquid ejection direction and the fan is above the portion. 3. The liquid ejecting apparatus according to claim 1, wherein the separator includes convection generation units which are formed at both ends in the reciprocating direction in a shape in which at least a part thereof projects in the reciprocating direction. 4. The liquid ejecting apparatus according to claim 3, wherein the convection generation unit is provided so as to extend in a direction intersecting the ejecting direction of the liquid and the reciprocating direction. 5. The liquid ejecting apparatus according to claim 1, further comprising: a rail which is provided so as to extend in the reciprocating direction, wherein the carriage includes a first unit which is supported by the rail, and a second unit which is configured to move along the ejecting direction of the liquid with respect to the first unit, and wherein the ejecting unit and the separator are provided in the second unit.\n' \
            + 'Abstract: A liquid ejecting apparatus which includes a carriage which can move in a reciprocating direction, in which the carriage is provided with an ejecting unit which ejects liquid, a circuit board which inputs a signal to the ejecting unit, and a fan which can blow air toward the circuit board, and includes a separator which separates a path through which mist which occurs along with ejecting of liquid from the ejecting unit reaches the fan.\n' \
            + f'Claims: {claims}\nAbstract: '}]      
        response_text = generate_chat_completion(messages)
        cnt += 1
        nb_words = nb_words//2

    if response_text == 400: response_text = ' '

    return response_text

API_KEY = "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"
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

parser = argparse.ArgumentParser()
parser.add_argument('--path_data', type=str, default="./data/eval_data.csv")
parser.add_argument('--metric', type=str, required=True, choices={"rouge", "chatgpt" ,"geval"})
parser.add_argument('--aspect', type=str, required=False, choices={"factuality", "coherence"})
parser.add_argument('--path_prediction', type=str, default="./predictions")


args = parser.parse_args()

if __name__ == '__main__':
    df = pd.read_csv(args.path_data)

    actuals, inputs = df['abstract'].to_list(), df['claims'].to_list()

    # make prediction directory 
    path_prediction = args.path_prediction
    if not os.path.isdir(path_prediction):
        os.makedirs(path_prediction)
    path_output = os.path.join(path_prediction, 'chatgpt_abstract.pred')

    if os.path.exists(path_output) and os.path.getsize(path_output) > 0:
        df_res = pd.read_csv(path_output)
        predictions = df_res['abstract'].to_list()
    else:
        predictions = []
        for c in tqdm(inputs):
            pred = generate_abstract(c)
            predictions.append(pred)
        df_res = pd.DataFrame({'abstract': predictions})
        df_res.to_csv(path_output, index=False)

    if args.metric == "rouge":
        scores = []
        rouge = evaluate.load('rouge')
        scores.append(rouge.compute(predictions=predictions, references=[[act] for act in actuals])['rougeL'])

        print(args.metric + ":" + str(scores))
    elif args.metric == "geval":
        scores = []
        # TODO
        print(args.metric + ":" + str(scores))

    elif args.metric == "ipc":
        scores = []
        # TODO
        print(args.metric + ":" + str(scores))
