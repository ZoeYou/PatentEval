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


def get_first_n_words(text, n):
    return " ".join(text.split(" ")[:n])

def process_examples(examples, prompt, n=1536//2):
    results = {
        "text": [],
    }
    for i in range(len(examples["text"])):
        text = " " + get_first_n_words(examples["text"][i], n)
        if prompt:
            text = prompt.replace("{{ claims }}", text)

        results["text"].append(" ".join(text.split(" ")))
    return results



def predict(args, df):
    input_claims = df['input_claims'].fillna('').to_list()

    # create our own dataset object
    dataset = datasets.Dataset.from_dict({"text": input_claims, "id": list(range(len(input_claims))), "split": ["test"] * len(input_claims)})

    logger.info("Dataset loaded")

    prompt = "Based on the provided patent claims below, please draft the subsequent claim for a continuation. This claim, which may be either dependent or independent, should be precise, legally sound, and in line with patent claim drafting conventions. Use the existing claims as a basis for your draft.\n" \
        + "Claims: {claims}"

    dataset = dataset.map(
        process_examples,
        batched=True,
        batch_size=10,
        fn_kwargs={"prompt": prompt},
    )

    logger.info("Dataset length: %d", len(dataset))

    gen_outputs: List[Response] = []

    if args.batch_size == 1:
        logger.info("Generating samples sequentially since batch_size=1")

        from text_generation import Client

        client = Client(f"http://{args.host}:{args.port}", timeout=300)

        for example in tqdm(dataset):
            _output = client.generate(
                example["text"],
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
                typical_p=args.typical_p,
                watermark=args.watermark,
                decoder_input_details=True,
            )
            gen_outputs.append(_output)
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
                typical_p=args.typical_p,
                watermark=args.watermark,
                decoder_input_details=True,
            )

        for i in tqdm(range(0, len(dataset), args.batch_size)):

            async def main_async():
                prompts = dataset[i : i + args.batch_size]["text"]
                outputs = await asyncio.gather(*[generate_async(p) for p in prompts])
                gen_outputs.extend(outputs)

            asyncio.run(main_async())

    gen_outputs = [i.dict() for i in gen_outputs]
    if not gen_outputs:
        logger.info("No outputs generated")
        return

    path_output = os.path.join(args.path_prediction, f'{args.model_name.split("/")[-1]}_claim.pred')

    logger.info(
        "Writing outputs to %s", path_output
    )
    os.makedirs(args.path_prediction, exist_ok=True)
    df_res = pd.DataFrame({'output_claim': [i['text'] for i in gen_outputs]})
    df_res.to_csv(path_output, index=False)
    


def main(args):
    df = pd.read_csv(args.path_data)[:5]

    # make prediction directory 
    path_prediction = args.path_prediction
    if not os.path.isdir(path_prediction):
        os.makedirs(path_prediction)
    path_output = os.path.join(path_prediction, '{}_claim.pred'.format(args.model_name.split("/")[-1]))

    
    if os.path.exists(path_output) and os.path.getsize(path_output) > 0:
        df_res = pd.read_csv(path_output)
        predictions = df_res['output_claim'].to_list()
    else:
        predict(args, df)
        df_res = pd.read_csv(path_output)
        predictions = df_res['output_claim'].to_list()

    pattern_claim = '\d+\. (?!\(canceled\))'
    for i, pred in enumerate(predictions):
        numberings = re.findall(pattern_claim, pred)
        claims_split = [c.strip() for c in re.split(pattern_claim, pred) if c]
        if len(numberings)>1:
            predictions[i] = numberings[0]+claims_split[0]

    df_res = pd.DataFrame({'output_claim': predictions})
    df_res.to_csv(path_output, index=False)




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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--typical_p", type=float, default=0.95)
    parser.add_argument("--watermark", type=bool, default=False)

    args = parser.parse_args()

    logger.info("Args: %s", args)

    main(args)