import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import argparse
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch



parser = argparse.ArgumentParser()
parser.add_argument('--path_data', type=str, default="./data/eval_data_c2c.csv")
parser.add_argument('--path_prediction', type=str, default="./predictions")

args = parser.parse_args()

if __name__ == '__main__':
  df = pd.read_csv(args.path_data)
  input_claims = df['input_claims'].fillna('').to_list()
  
  # tokenizer = AutoTokenizer.from_pretrained("patent/PatentGPT-J-6B")
  # model = AutoModelForCausalLM.from_pretrained("patent/PatentGPT-J-6B")
  tokenizer = AutoTokenizer.from_pretrained("patent/PatentGPT-J-1.6B")
  model = AutoModelForCausalLM.from_pretrained("patent/PatentGPT-J-1.6B")

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
  repetition_penalty = 1.5  # Controls the tendency to repeat generated text (higher values make it less repetitive)

  model.eval()

  def get_claim(claims, maxsize=512, numberTries=0, max_retries=5):
    if numberTries >= max_retries:
      print("Error: Retrying too many times! Please check the format of the input text!")
      return ""
    else:
      claims = ' '.join(claims.split(' ')[:maxsize])
      maxsize = int(len(claims.split(" ")) // 2)
      try:
        input = '<|start_of_claim|> ' + str(claims).replace('\n', ' <|dep|> ') +  f' <|dep|>'
        input_ids = tokenizer.encode(input, return_tensors="pt").to(device)

        # Generate text using the model
        output = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id
        )
      except RuntimeError:
        return get_claim(claims, numberTries=numberTries+1, maxsize=maxsize)
      
      # Decode the generated output
      generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
      generated_text = generated_text.replace(input, '').split('<|end_of_claim|>')[0].split('<|dep|>')[0].strip()
      return generated_text

  predictions = []
  for input in tqdm(input_claims):
    generated_text = get_claim(input)
    predictions.append(generated_text)  

  df_res = pd.DataFrame({'output_claim': predictions})
  df_res.to_csv(path_output, index=False)
