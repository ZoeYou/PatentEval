"""
create dataset for next claim generation
"""

import pandas as pd
import re, random

pattern_claim = '\d+\. [AT]' #'\d+\. (?!\(canceled\))'
df = pd.read_csv('./data/eval_data.csv')

inputs = []
for _, row in df.iterrows():
    claims = row['claims']

    numberings = re.findall(pattern_claim, claims)
    claims_split = [c.strip() for c in re.split(pattern_claim, claims) if c]

    index = random.choice([1,2,3])
    numberings = numberings[:index]
    claims_split = claims_split[:index]
    
    constructed_input = []
    for numbering, claim in zip(numberings, claims_split):
        constructed_input.append(numbering + claim)

    inputs.append('\n'.join(constructed_input))
    
res_df = pd.DataFrame({'input_claims': inputs})
res_df.to_csv('./data/eval_data_c2c.csv', index=False)