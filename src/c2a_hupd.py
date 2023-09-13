import argparse, os
import pandas as pd
from tqdm import tqdm

from transformers import pipeline
import evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--path_data', type=str, default="./data/eval_data.csv")
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


    
