"""
Script for generating synthetic data for FactCC training.

Script expects source documents in `jsonl` format with each source document
embedded in a separate json object.

Json objects are required to contain `id` and `text` keys.
"""

import argparse
import json
import os

from tqdm import tqdm
import multiprocessing

import augmentation_ops as ops



def load_source_docs(file_path, to_dict=False):
    with open(file_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    if to_dict:
        data = {example["id"]: example for example in data}
    return data


def save_data(args, data, name_suffix):
    output_file = os.path.splitext(args.data_file)[0] + "-" + name_suffix + ".jsonl"

    with open(output_file, "w", encoding="utf-8") as fd:
        for example in data:
            example = dict(example)
            if not isinstance(example["text"], str):
                example["text"] = example["text"].text
            if not isinstance(example["claim"], str):
                example["claim"] = example["claim"].text
            fd.write(json.dumps(example, ensure_ascii=False) + "\n")


def load_data(args, name_suffix):
    input_file = os.path.splitext(args.data_file)[0] + "-" + name_suffix + ".jsonl"

    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

def worker(batch, operation):
    results = []
    for example in batch:
        try:
            new_example = operation.transform(example)
            if new_example:
                results.append(new_example)
        except Exception as e:
            print("Caught exception:", e)
    return results

def apply_transformation_parallel(data, operation, num_processes=4):
    def worker(input_queue, output_queue, progress_bar):
        for example in iter(input_queue.get, None):
            try:
                new_example = operation.transform(example)
                if new_example:
                    output_queue.put(new_example)
                progress_bar.update(1)  # Update the progress bar for this worker
            except Exception as e:
                print("Caught exception:", e)

    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()

    # Calculate the number of iterations per worker
    num_items_per_worker = len(data) // num_processes

    # Create and start worker processes
    processes = []
    progress_bars = []
    for i in range(num_processes):
        start_idx = i * num_items_per_worker
        end_idx = (i + 1) * num_items_per_worker if i < num_processes - 1 else len(data)
        worker_data = data[start_idx:end_idx]

        progress_bar = tqdm(total=len(worker_data), desc=f"Worker-{i}")
        progress_bars.append(progress_bar)

        process = multiprocessing.Process(target=worker, args=(input_queue, output_queue, progress_bar), name=f"Worker-{i}")
        process.start()
        processes.append(process)

        # Add data for this worker to the input queue
        for example in worker_data:
            input_queue.put(example)

    # Add None to signal the workers to exit
    for _ in range(num_processes):
        input_queue.put(None)

    # Wait for all processes to finish
    for process in processes:
        process.join()

    # Close the progress bars
    for progress_bar in progress_bars:
        progress_bar.close()

    # Collect results from the output queue
    new_data = []
    while not output_queue.empty():
        new_data.append(output_queue.get())

    return new_data


def apply_transformation(data, operation, do_parallel=False):
    if do_parallel:
        new_data = apply_transformation_parallel(data, operation, num_processes=4)
    else:
        new_data = []
        for example in tqdm(data):
            try:
                new_example = operation.transform(example)
                if new_example:
                    new_data.append(new_example)
            except Exception as e:
                print("Caught exception:", e)
    return new_data           


def main(args):
    # load data
    source_docs = load_source_docs(args.data_file, to_dict=False)
    print("Loaded %d source documents." % len(source_docs))

    # create or load positive examples
    try:
        data = load_data(args, "clean")
    except FileNotFoundError:
        print("Creating data examples")
        sclaims_op = ops.SampleSentences()
        data = apply_transformation(source_docs, sclaims_op)
        print("Created %s example pairs." % len(data))
        if args.save_intermediate:
            save_data(args, data, "clean")

    # backtranslate
    data_btrans = []
    if args.all_augmentations or "backtranslation" in args.augmentations:
        try:
            btrans_op = load_data(args, 'btrans')
        except FileNotFoundError:
            print("Creating backtranslation examples")
            btrans_op = ops.Backtranslation()
            data_btrans = apply_transformation(data, btrans_op)
            print("Backtranslated %s example pairs." % len(data_btrans))

            if args.save_intermediate:
                save_data(args, data_btrans, "btrans")

    data_positive = data + data_btrans
    save_data(args, data_positive, "positive")

    # create negative examples
    data_pronoun = []
    if args.all_augmentations or "pronoun_swap" in args.augmentations:
        try:
            data_pronoun = load_data(args, "pronoun")
        except FileNotFoundError:
            print("Creating pronoun examples")
            pronoun_op = ops.PronounSwap()
            data_pronoun = apply_transformation(data_positive, pronoun_op)
            print("PronounSwap %s example pairs." % len(data_pronoun))

            if args.save_intermediate:
                save_data(args, data_pronoun, "pronoun")

    data_dateswp = []
    if args.all_augmentations or "date_swap" in args.augmentations:
        try:
            data_dateswp = load_data(args, "dateswp")
        except FileNotFoundError:
            print("Creating date swap examples")       
            dateswap_op = ops.DateSwap()
            data_dateswp = apply_transformation(data_positive, dateswap_op)
            print("DateSwap %s example pairs." % len(data_dateswp))

            if args.save_intermediate:
                save_data(args, data_dateswp, "dateswp")

    data_numswp = []
    if args.all_augmentations or "number_swap" in args.augmentations:
        try:
            data_numswp = load_data(args, "numswp")
        except FileNotFoundError: 
            print("Creating number swap examples")
            numswap_op = ops.NumberSwap()
            data_numswp = apply_transformation(data_positive, numswap_op)
            print("NumberSwap %s example pairs." % len(data_numswp))

            if args.save_intermediate:
                save_data(args, data_numswp, "numswp")

    data_entswp = []
    if args.all_augmentations or "entity_swap" in args.augmentations:
        try:
            data_entswp = load_data(args, "entswp")
        except FileNotFoundError: 
            print("Creating entity swap examples")
            entswap_op = ops.EntitySwap()
            data_entswp = apply_transformation(data_positive, entswap_op)
            print("EntitySwap %s example pairs." % len(data_entswp))

            if args.save_intermediate:
                save_data(args, data_entswp, "entswp")

    data_negation = []
    if args.all_augmentations or "negation" in args.augmentations:
        try:
            data_negation = load_data(args, "negation")
        except FileNotFoundError: 
            print("Creating negation examples")
            negation_op = ops.NegateSentences()
            data_negation = apply_transformation(data_positive, negation_op)
            print("Negation %s example pairs." % len(data_negation))

            if args.save_intermediate:
                save_data(args, data_negation, "negation")

    # add noise to all
    data_negative = data_pronoun + data_dateswp + data_numswp + data_entswp + data_negation
    save_data(args, data_negative, "negative")

    # ADD NOISE
    data_pos_low_noise = []
    data_neg_low_noise = []
    
    if args.all_augmentations or "noise" in args.augmentations:
        # add light noise
        print("Adding light noise to data")
        low_noise_op = ops.AddNoise()

        data_pos_low_noise = apply_transformation(data_positive, low_noise_op)
        print("PositiveNoisy %s example pairs." % len(data_pos_low_noise))
        save_data(args, data_pos_low_noise, "positive-noise")

        data_neg_low_noise = apply_transformation(data_negative, low_noise_op)
        print("NegativeNoisy %s example pairs." % len(data_neg_low_noise))
        save_data(args, data_neg_low_noise, "negative-noise")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("data_file", type=str, help="Path to file containing source documents.")
    PARSER.add_argument("--augmentations", type=str, nargs="+", default=(), help="List of data augmentation applied to data.")
    PARSER.add_argument("--all_augmentations", action="store_true", help="Flag whether all augmentation should be applied.")
    PARSER.add_argument("--save_intermediate", action="store_true", help="Flag whether intermediate data from each transformation should be saved in separate files.")
    ARGS = PARSER.parse_args()
    main(ARGS)
