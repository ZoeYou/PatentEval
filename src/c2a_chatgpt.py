import pandas as pd
import argparse, os
import openai
import time, random
from tqdm import tqdm
import evaluate
from openai.error import *

openai.api_key = "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"
    

def reduceToMaxPT(text, maxpt=5000):
    # for text longer than maxpt tokens, we need to reduce it to less than maxpt tokens, if not the chatgpt will be possible to refuse it
    return ' '.join(text.split()[:maxpt])


def generate_abstract(claims, numberTries=0, gptChoice=0, maxsize=4097, exponential_base: float = 2, jitter: bool = True, max_retries: int = 5):
    prompt = "Please draft a patent abstract from the provided claims. The abstract should concisely summarize the technical disclosure, enabling any reader to quickly understand the subject matter. You'll be given an example to guide your creation.\n" \
         + 'Claims: 1. A liquid ejecting apparatus comprising: a carriage which is configured to move in a reciprocating direction, wherein the carriage is provided with an ejecting unit which ejects liquid, a circuit board which inputs a signal to the ejecting unit, and a fan which is configured to blow air toward the circuit board, and wherein the carriage includes a separator which separates a path through which mist which occurs along with ejecting of the liquid from the ejecting unit reaches the fan, wherein the separator is provided in the carriage so that at least a part thereof is arranged between the ejecting unit and the fan, when viewed in a direction intersecting the ejecting direction of the liquid. 2. The liquid ejecting apparatus according to claim 1, wherein at least ejection ports of the ejecting unit are below a portion of the separator in the liquid ejection direction and the fan is above the portion. 3. The liquid ejecting apparatus according to claim 1, wherein the separator includes convection generation units which are formed at both ends in the reciprocating direction in a shape in which at least a part thereof projects in the reciprocating direction. 4. The liquid ejecting apparatus according to claim 3, wherein the convection generation unit is provided so as to extend in a direction intersecting the ejecting direction of the liquid and the reciprocating direction. 5. The liquid ejecting apparatus according to claim 1, further comprising: a rail which is provided so as to extend in the reciprocating direction, wherein the carriage includes a first unit which is supported by the rail, and a second unit which is configured to move along the ejecting direction of the liquid with respect to the first unit, and wherein the ejecting unit and the separator are provided in the second unit.\n' \
         + 'Abstract: A liquid ejecting apparatus which includes a carriage which can move in a reciprocating direction, in which the carriage is provided with an ejecting unit which ejects liquid, a circuit board which inputs a signal to the ejecting unit, and a fan which can blow air toward the circuit board, and includes a separator which separates a path through which mist which occurs along with ejecting of liquid from the ejecting unit reaches the fan.\n' \
         + 'Claims: {claims}\nAbstract: '

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
            return generate_abstract(claims, gptChoice=gptChoice, numberTries=numberTries+1, maxsize=maxsize)
        else:
            return generate_abstract(claims, gptChoice=gptChoice+1, numberTries=numberTries+1, maxsize=maxsize)



parser = argparse.ArgumentParser()
parser.add_argument('--path_data', type=str, default="./data/eval_data.csv")
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

    scores = []
    rouge = evaluate.load('rouge')
    scores.append(rouge.compute(predictions=predictions, references=[[act] for act in actuals])['rougeL'])

    print("rouge :" + str(scores))

