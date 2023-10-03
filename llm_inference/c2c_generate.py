import argparse

import os
from tqdm import tqdm
import re
import datasets

import pandas as pd
import logging
import os
import sys
from typing import List

from text_generation.types import Response

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)
# set it to INFO
logger.setLevel(logging.DEBUG)


def get_first_sentences_within_n_words(text, n):
    # tokenize claims into senteces 
    re_claims = re.compile(r'(\d+\.\s[A-Z][^\.!?]*[\.!?])', re.M)
    sentences = re_claims.findall(text)

    # get first sentences until n words
    first_sentences = ""
    for sent in sentences:
        if len(first_sentences.split(" ")) + len(sent.split(" ")) < n:
            first_sentences = first_sentences + " " + sent
        elif first_sentences == "":
            return " ".join(sent.split(" ")[:n])
        else:
            return first_sentences


def get_first_words(text, n):
    return " ".join(text.split(" ")[:n])


def process_examples(examples, prompt, n=512):
    results = {
        "text": [],
    }

    for i in range(len(examples["claims"])):
        try:
            text = get_first_sentences_within_n_words(examples["claims"][i], n)
            text = prompt.replace("{{ claims }}", text)
        except TypeError:
            text = get_first_words(examples["claims"][i], n)
            text = prompt.replace("{{ claims }}", text)

        results["text"].append(" ".join(text.split(" ")))
    return results
    

def predict_process(args, df):

    input_claims = df['input_claims'].fillna('').to_list()

    # create our own dataset object
    dataset = datasets.Dataset.from_dict({"claims": input_claims, "id": list(range(len(input_claims))), "split": ["test"] * len(input_claims)})

    logger.info("Dataset loaded")

    prompt = "Based on the provided patent claims below, please draft ONLY the subsequent claim for a continuation. This claim, which may be either dependent or independent, should be precise, legally sound, and in line with patent claim drafting conventions. Use the existing claims as a basis for your draft.\n" \
        + "Claims: {{ claims }}\n" \
        + "Next claim: "

    dataset = dataset.map(
        process_examples,
        batched=True,
        batch_size=10,
        fn_kwargs={"prompt": prompt,
                   "n": args.n_words},
    )

    logger.info("Dataset length: %d", len(dataset))

    gen_outputs: List[Response] = []

    if args.batch_size == 1:
        logger.info("Generating samples sequentially since batch_size=1")

        from text_generation import Client

        client = Client(f"http://{args.host}:{args.port}", timeout=300)

        for i, example in enumerate(tqdm(dataset)):
            count = 0
            while count < 10:        
                try:
                    logger.info("Generating for %s", example["text"])
                    _output = client.generate(
                        example["text"],
                        do_sample=args.do_sample,
                        max_new_tokens=args.max_new_tokens,
                        repetition_penalty=args.repetition_penalty,
                        return_full_text=False,
                        stop_sequences=None,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        truncate=None,
                        decoder_input_details=True,
                    )
                    count += 1
                    logger.info("Generated: %s", _output.dict()["generated_text"])
                    if (_output.dict()["generated_text"].count('*') > 15 or len(_output.dict()["generated_text"]) < 30):
                        example["text"] = process_examples({"claims": [dataset[i]["claims"]]}, prompt, n=len(example["text"].split(" "))//2)["text"][0]
                        logger.info("Retrying...")
                        continue
                    gen_outputs.append(_output.dict())
                    break
                except:
                    logger.info("Retrying...")
                    example["text"] = process_examples({"claims": [dataset[i]["claims"]]}, prompt, n=len(example["text"].split(" "))//2)["text"][0]

    else:
        import asyncio

        from text_generation import AsyncClient

        client = AsyncClient(f"http://{args.host}:{args.port}", timeout=300)

        logger.info(
            "Generating samples in parallel since batch_size=%d", args.batch_size
        )

        async def generate_async(text):
            return await client.generate(
                prompt=text,
                do_sample=args.do_sample,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
                seed=args.seed,
                return_full_text=False,
                stop_sequences=None,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                truncate=None,
                decoder_input_details=True,
            )

        for i in tqdm(range(0, len(dataset), args.batch_size)):

            async def main_async():
                prompts = dataset[i : i + args.batch_size]["text"]
                outputs = await asyncio.gather(*[generate_async(p) for p in prompts])
                gen_outputs.extend(outputs)

            asyncio.run(main_async())

    if not gen_outputs:
        logger.info("No outputs generated")
        return

    path_output = os.path.join(args.path_prediction, f'{args.model_name.split("/")[-1]}_claim.pred')

    logger.info(
        "Writing outputs to %s", path_output
    )
    os.makedirs(args.path_prediction, exist_ok=True)
    
    predictions = [i['generated_text'].strip() for i in gen_outputs]
    pattern_claim = '\d+\. (?!\(canceled\))'
    for i, pred in enumerate(predictions):
        numberings = re.findall(pattern_claim, pred)
        claims_split = [c.strip() for c in re.split(pattern_claim, pred) if c]
        if len(numberings)>1:
            predictions[i] = numberings[0]+claims_split[0]

    predictions = [re.sub(r'Next claim:', '', p) for p in predictions]
    df_res = pd.DataFrame({'output_claim': predictions})
    df_res.to_csv(path_output, index=False)
    


def main(args):
    df = pd.read_csv(args.path_data)

    # make prediction directory 
    path_prediction = args.path_prediction
    if not os.path.isdir(path_prediction):
        os.makedirs(path_prediction)
    path_output = os.path.join(path_prediction, '{}_claim.pred'.format(args.model_name.split("/")[-1]))

    if not (os.path.exists(path_output) and os.path.getsize(path_output) > 0):
        predict_process(args, df)
       






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, default=None, required=True, help='model name (with size)')
    parser.add_argument('-d', '--path_data', type=str, default='../data/eval_data_c2c.csv', help='path to dataset')
    parser.add_argument('-o', '--path_prediction', type=str, default='../predictions/', help='dir to save results')

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=60550)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--do_sample", type=bool, default=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.9)

    parser.add_argument('-n', '--n_words', type=int, default=512, help='number of input words')

    args = parser.parse_args()

    logger.info("Args: %s", args)

    main(args)