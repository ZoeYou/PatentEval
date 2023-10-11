"""
create dataset for next claim generation
"""

import pandas as pd
import re, random
from tqdm import tqdm


pattern_numbering = r"\. \d+[\.:)] |^1[\.:)] "
df = pd.read_csv('./data/eval_data.csv')

inputs, real_next_claims, is_dependent = [], [], []
domains = df['domain'].tolist()

for _, row in tqdm(df.iterrows()):
    claims = row['claims']

    numberings = re.findall(pattern_numbering, claims)
    claims_split = [c.strip("\n.") for c in re.split(pattern_numbering, claims) if c]

    if len(numberings) == 1:
        index = 0
        assert type(numberings[index]) == str, f"numberings: {numberings} is not a string"
        previous_numberings = [numberings[index]]
        previous_claims_split = [claims_split[index]]

    else:
        if len(numberings) > 3:
            index = random.choice([1,2,3])
        else:
            index = len(numberings) - 1
        previous_numberings = numberings[:index]
        previous_claims_split = claims_split[:index]

    assert len(previous_numberings) > 0, f"previous_numberings: {previous_numberings} is empty"

    constructed_input = []
    for numbering, claim in zip(previous_numberings, previous_claims_split):
        constructed_input.append(numbering + claim)

    input = ''.join(constructed_input).strip(" .") + "."

    if re.search(r"\. \d+\. ", input):
        input = re.sub(r"\. (\d+)\. ", r".\n\1. ", input)
    elif re.search(r"\. \d+: ", input):
        input = re.sub(r"\. (\d+): ", r".\n\1: ", input)
    elif re.search(r"\. \d+\) ", input):
        input = re.sub(r"\. (\d+)\) ", r".\n\1) ", input)

    if len(numbering) == 1:
        real_next_claim = "[end]"
        dependency = False
    else:
        real_next_claim = (numberings[index] + claims_split[index]).strip(" .") + "."
        dependency = re.search(r"\d+[\.:)] (The|An?) .+ of claim \d+|\s+one of the aforementioned claims[ ,]+|\s+one of claims \d+ to \d+[ ,]+|\s+according (to )?claim \d+(?: or claim \d+)*[ ,.;]+|\d+[\.:)] (The |An? )?.+ (as )?(recited |claimed |set forth |defined |)in claim \d+", real_next_claim)
        if dependency:
            dependency = True
        else:
            dependency = False

    inputs.append(input)
    real_next_claims.append(real_next_claim)
    is_dependent.append(dependency)


res_df = pd.DataFrame({'domain': domains, 'input_claims': inputs, 'true_next_claim': real_next_claims, 'is_dependent': is_dependent})
res_df.to_csv('./data/eval_data_c2c.csv', index=False)