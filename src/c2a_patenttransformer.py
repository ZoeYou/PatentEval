"""
script for evaluating generation quality of claism => abstract for different models
"""
import pandas as pd
import argparse
import json
import os
import numpy as np
import tensorflow as tf
import sys

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)





# the following code is copied from: 
# https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039

import requests

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    save_response_content(response, destination) 

download_links = {
'M1':{
    'checkpoint': '1fbCvkKhDsVNhIQ8BoL_rbXF5TCQ2P86i',
    'model.ckpt-1000000.index': '1ufRUSOC3kov4MO-tIPFTpvlxyunjMAN-',
    'model.ckpt-1000000.meta': '1gzqD7PLL2oc4akZ-MZoAKpQldRTzlAtB',
    'model.ckpt-1000000.data-00000-of-00001': '1bHr1aNlx966-k62jXZ1tzwWXPJ-hlu1B'
},
'M2':{
    'checkpoint': '1Ccovn7Bi7VRzwvHs0d_V565J1Jr5pVnf',
    'model.ckpt-1000000.index': '1Cj4w-Qw9Ph52qbhf_VEjra41iYwoHMjq',
    'model.ckpt-1000000.meta': '1xMUJUE16UxexVX_j2XnbNY2QLg9OyTYQ',
    'model.ckpt-1000000.data-00000-of-00001': '1TUZR1O7cAS0UAqjlt0T7t61PgIv9dilG'
},
'M3':{
    'checkpoint': '1jJUI5JmeZtHEViVOWoVrih-bdxbvvuQk',
    'model.ckpt-1000000.index': '1_3lDxDaqzHufGEVZMICvPR4_3qun9PRB',
    'model.ckpt-1000000.meta': '1-FWkpAVA6n_6BbdORzsoN5QO-PvcOd8S',
    'model.ckpt-1000000.data-00000-of-00001': '1bzcJ7JffJIRgobseYqkokEcZmqsG_k5o'
},
'M4':{
    'checkpoint': '1PpF9mlCq45NlFhiUEroP6HfejrHM_lDk',
    'model.ckpt-1000000.index': '1BCqepKM8FOdFI2LGuqyo6ylVr8Df2b0j',
    'model.ckpt-1000000.meta': '1UFuiF-9uA-0icH3Gk-X2NMIozeUK8ecv',
    'model.ckpt-1000000.data-00000-of-00001': '1rTJ8Z9pQQSuLR2bGQyNh4rpYfJx3Rbta'
},
}

def generate_output(context, count, num_of_generation, sess, text, 
                       sampler, enc, batch_size, cut_tag):
  results = []

  # forward
  text = text.strip()
  context_tokens = enc.encode(text)

  out = sess.run(sampler, feed_dict={
      context: [context_tokens for _ in range(batch_size)]
  })[:, len(context_tokens):]
  
  for i in range(batch_size):
    text = enc.decode(out[i])
    pos = text.find(cut_tag)
    if pos >= 0:
      text = text[:pos].strip()
    if text == '':
      continue
      
    results.append(text)
    count += 1
    if count >= num_of_generation:
      break
      
  return results  

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
  while count < gen_count:
    batch_results = generate_output(context, count, 
      gen_count, sess, raw_text, sampler, enc, 
      batch_size, foward_end_tags[meta2])
    count += len(batch_results)
    all_results += batch_results

  for i, row in enumerate(all_results):
    row = row.replace('<|span|>', '\n\t')
    print('%s' % row) 
    #print('[ %s ] %s' % (i, row))
  print('')

  return all_results

def patent_text_gen(input_text, metadata, direction='forward', gen_count=1):
  all_results = []

  print('[ %s ] direction=%s, input_text=%s' % (metadata, direction, input_text))
  count = 0 
  if direction == 'forward':
    raw_text = foward_start_tags[metadata] + ' ' + input_text
    while count < gen_count:
      batch_results = generate_output(context, count, 
        gen_count, sess, raw_text, sampler, enc, 
        batch_size, foward_end_tags[metadata])
      count += len(batch_results)
      for i, row in enumerate(batch_results):
        s = input_text + ' ' + row
        all_results.append(s.strip())
  elif direction == 'backward':
    reversed_text = ' '.join(input_text.split()[::-1])
    raw_text = backward_end_tags[metadata] + ' ' + reversed_text
    while count < gen_count:
      batch_results = generate_output(context, count, 
        gen_count, sess, raw_text, sampler, enc, 
        batch_size, backward_start_tags[metadata])
      count += len(batch_results)       
      for i, row in enumerate(batch_results):
        reversed_row = ' '.join(row.split()[::-1])
        all_results.append(reversed_row + ' ' + input_text)
  elif direction == 'both':
    raw_text = foward_start_tags[metadata] + ' ' + input_text
    # forward
    while count < gen_count:
      batch_results = generate_output(context, count, 
        gen_count, sess, raw_text, sampler, enc, 
        batch_size, foward_end_tags[metadata])
      count += len(batch_results) 
      for i, row in enumerate(batch_results):
        all_results.append(input_text + ' ' + row)

    # backward, generate one by one
    for i, one_record in enumerate(all_results):
      reversed_text = ' '.join(one_record.split()[::-1])
      raw_text = backward_end_tags[metadata] + ' ' + reversed_text
      batch_results = generate_output(context, count, 
        1, sess, raw_text, sampler, enc, 
        batch_size, backward_start_tags[metadata])
      reversed_result = ' '.join(batch_results[0].split()[::-1])
      all_results[i] = reversed_result + ' ' + one_record
  else: 
    print('unknown direction: %s' % direction)
  
  for i, row in enumerate(all_results):
    print('%s' % row)
    #print('[ %s ] %s' % (i, row))
  print('')

  return all_results 

parser = argparse.ArgumentParser()
parser.add_argument('--path_data', type=str, default="./test_data.csv")
parser.add_argument('--metric', type=str, required=True, choices={"rouge", "chatgpt" ,"geval"})
parser.add_argument('--aspect', type=str, required=False, choices={"factuality", "coherence"})
parser.add_argument('--pretrained_model', type=str, default='M2')

args = parser.parse_args()

if __name__ == '__main__':
    df = pd.read_csv(args.path_data)

    pretrained_model = args.pretrained_model
    # M1: small model for 1976~2016
    # M2: medium model for 1976~2016
    # M3: small model for 2016
    # M4: medium model for 2016
    if pretrained_model in ['M1', 'M3']:
        model_name= '124M'
    elif pretrained_model in ['M2', 'M4']:
        model_name= '355M'
    else:
        print('unknown mode: %s' % pretrained_model)
        sys.exit(1)

    # download gpt-2 environment
    proj_folder = '/home/yzuo/scratch/PatEval/gpt-2'
    git_src = 'https://github.com/openai/gpt-2' 
    if not os.path.exists(proj_folder):
        os.system(f'git clone {git_src}')
    else:
        print('existed: %s' % proj_folder)
    os.chdir(proj_folder)  
    os.system('git pull origin master')

    os.chdir(proj_folder)
    os.system('pip3 install -r requirements.txt')

    print('tf version: %s' % tf.__version__)
    device_name = tf.test.gpu_device_name()
    if 'GPU' in device_name:
        print('GPU ready: %s' % device_name) 
        GPU_FLAG = True
    else:
        print('CPU only.....')    

    src_path = '/home/yzuo/scratch/PatEval/gpt-2/src'
    if src_path not in sys.path:
        sys.path += [src_path]

    os.chdir(proj_folder)
    if os.path.exists(os.path.join('models', model_name)) == False:
        print('download model %s....' % model_name)
        os.system(f'PYTHONPATH=src; python ./download_model.py {model_name}')
    else:
        print('existed: model %s' % model_name)  

    # donwload fine-tuned model for patents
    ckpt_path = 'saved_checkpoint_%s' % model_name
    if os.path.exists(ckpt_path):
        print('Existed: %s' % ckpt_path)
        os.system(f'ls {ckpt_path}')
    else:
        os.mkdir(ckpt_path)
        os.chdir(ckpt_path)
        print('Downloading files to %s....' % ckpt_path)
        for k, v in download_links[pretrained_model].items():
            download_file_from_google_drive(v, k)
        os.system('ls -al')
    print('Download: ok')
    os.chdir(proj_folder)

    sys.path.append('/home/yzuo/scratch/PatEval/gpt-2/src/')

    import encoder, model, sample

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

    seed=None
    nsamples=1
    batch_size=1
    length=None
    temperature=1
    top_k=40

    models_dir = 'models'
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    sess = tf.InteractiveSession() 
    context = tf.placeholder(tf.int32, [batch_size, None])
    sampler = sample.sample_sequence(
    hparams=hparams, length=length,
    context=context,
    batch_size=batch_size,
    temperature=temperature, top_k=top_k
    )
    saver = tf.compat.v1.train.Saver()
    
    ckpt = tf.train.latest_checkpoint(ckpt_path)
    saver.restore(sess, ckpt)

    #seed_text = 'temperature optimization'
    while True:
        print('Demo: a few words --> title --> abstract --> independent claim --> dependent claims')
        print('Input text or "exit" or "Enter" key for unconditional sampling.....')
        seed_text = input(">>> ")
        direction = 'both'
        if seed_text == 'exit':
            break
        if seed_text == '':
            direction = 'forward'

    # from a few words to a patent title
    outputs = patent_text_gen(input_text=seed_text, metadata='title', 
                                direction=direction, gen_count=1)

    # from the patent title to a patent abstract
    results = text2text_mapping(input_text=outputs[0], mapping='title2abstract', gen_count=1)

    # from the patent abstract to an independent claim
    results = text2text_mapping(input_text=outputs[0], mapping='abstract2claim', gen_count=1)

    # from the independent claim to two dependent claims
    results = text2text_mapping(input_text=outputs[0], mapping='dep', gen_count=2)
    
    print('Thank you for testing Augmented Inventing.')

