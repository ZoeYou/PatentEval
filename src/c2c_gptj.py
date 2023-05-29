"""
script for evaluating generation quality of next claim generation
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import argparse
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

foward_start_tags = {'title':'<|startoftitle|>', \
                    'abstract':'<|startofabstract|>', \
                    'claim': '<|startoftext|>', \
                    'dep': '<|startoftext|>'}
foward_end_tags = {'title':'<|endoftitle|>', 
                'abstract':'<|endofabstract|>', \
                'claim': '<|endoftext|>', \
                'dep': '<|startoftext|>'}
backward_start_tags = {'title':'<|backwardtitlestart>', \
                    'abstract':'<|backwardabstractstart>', \
                    'claim': '<|startofbackward|>'}
backward_end_tags = {'title':'<|backwardtitleend|>', \
                'abstract':'<|backwardabstractend|>', \
                'claim': '<|endofbackward|>'}

# text2text mapping
tag_title2abstract = '<|title2abstract|>'
tag_abstract2title = '<|abstract2title|>'
tag_abstract2claim = '<|abstract2claim|>'
tag_claim2abstract = '<|claim2abstract|>'
dep_separator = '<|dep|>'

def text2text_mapping(input_text, mapping, gen_count=1):
  all_results = []
  if mapping == 'dep':
    meta1 = meta2 = 'claim'
    print('[ dependent claim ]')
  else:
    meta1, meta2 = mapping.split('2')
    print('[ %s --> %s ]' % (meta1, meta2))
  raw_text = ''

  count = 0 
  raw_text = ' '.join([foward_start_tags[meta1], input_text, \
    foward_end_tags[meta1]]) 
  raw_text += ' <|' + mapping + '|> ' + foward_start_tags[meta2]
  return raw_text
  # while count < gen_count:
  #   batch_results = generate_output(context, count, 
  #     gen_count, sess, raw_text, sampler, enc, 
  #     batch_size, foward_end_tags[meta2])
  #   count += len(batch_results)
  #   all_results += batch_results

  # all_results = [row.replace('<|span|>', '\n\t') for row in all_results]
  # return all_results

parser = argparse.ArgumentParser()
parser.add_argument('--path_data', type=str, default="./data/eval_data_c2c.csv")
parser.add_argument('--path_prediction', type=str, default="../predictions")

args = parser.parse_args()

if __name__ == '__main__':
  df = pd.read_csv(args.path_data)
  
  # tokenizer = AutoTokenizer.from_pretrained("patent/PatentGPT-J-6B")
  # model = AutoModelForCausalLM.from_pretrained("patent/PatentGPT-J-6B")
  tokenizer = AutoTokenizer.from_pretrained("patent/PatentGPT-J-1.6B")
  model = AutoModelForCausalLM.from_pretrained("patent/PatentGPT-J-1.6B")

  input_claims = df['input_claims'].to_list()

  # make prediction directory 
  path_prediction = args.path_prediction
  if not os.path.isdir(path_prediction):
      os.makedirs(path_prediction)
  path_output = os.path.join(path_prediction, 'patentgptj_claim.pred')

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)



  # Generate text
  max_length = 1024  # Maximum length of the generated text
  temperature = 0.001  # Controls the randomness of the generated text (higher values make it more random)
  repetition_penalty = 1.2  # Controls the tendency to repeat generated text (higher values make it less repetitive)

  model.eval()
  for input in input_claims[:3]:
    pred = text2text_mapping(input_text=input, mapping='dep', gen_count=1)[0]
    input_ids = tokenizer.encode(input, return_tensors="pt").to(device)

    # Generate text using the model
    output = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the generated output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    try:
      generated_text = generated_text.replace('\n',' ').split('<|backward_claim_start|>')[1].split('<|backward_claim_end|>')[0]
    except IndexError:
      generated_text = ""

    print("Input text:")
    print(input)
    print("==========================================")
    print("Generated text:")
    print(generated_text)
  