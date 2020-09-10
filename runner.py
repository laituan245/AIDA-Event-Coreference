import os
import math
import torch
import tqdm
import random

from transformers import *
from models import BasicCorefModel
from utils import RunningAverage, prepare_configs
from scorer import aida_evaluate
from data import load_oneie_dataset, load_aida_dataset
from argparse import ArgumentParser

# Constants
LIFU_SYSTEM = 'lifu'
YING_SYSTEM = 'ying'
SYSTEM = YING_SYSTEM


# Main Functions
def train(config_name):
    # Prepare tokenizer, dataset, and model
    configs = prepare_configs(config_name)
    tokenizer = BertTokenizer.from_pretrained(configs['transformer'], do_basic_tokenize=False)
    if configs['use_arguments_features']: assert(configs['use_groundtruth'])

    # Use the entire ACE-05 dataset for training
    train_set, ace_dev_set, ace_test_set = load_oneie_dataset('resources/ACE05-E', tokenizer, 'resources/ACE05-E-Preds')
    train_set.data = train_set.data + ace_dev_set.data + ace_test_set.data
    print('Number of training documents (ACE-05 dataset) is: {}'.format(len(train_set)))

    # Load the AIDA data for validation/testing
    cs_filepath = 'resources/AIDA/{}_system/events_fine_all_clean.cs'.format(SYSTEM)
    ltf_dir = 'resources/AIDA/ltf'
    dev_set = load_aida_dataset(cs_filepath = cs_filepath, ltf_dir = ltf_dir, tokenizer=tokenizer)[-1]
    print('Number of dev documents (AIDA dataset) is: {}'.format(len(dev_set)))

    # Load the model
    model = BasicCorefModel(configs)
    print('Initialized tokenier, dataset, and model')

    # Initialize the optimizer
    num_train_docs = len(train_set)
    epoch_steps = int(math.ceil(num_train_docs / configs['batch_size']))
    num_train_steps = int(epoch_steps * configs['epochs'])
    num_warmup_steps = int(num_train_steps * 0.1)
    optimizer = model.get_optimizer(num_warmup_steps, num_train_steps)
    print('Initialized optimizer')

    # Main training loop
    best_dev_score, iters, batch_loss = 0.0, 0, 0
    for epoch in range(configs['epochs']):
        #print('Epoch: {}'.format(epoch))
        print('\n')
        progress = tqdm.tqdm(total=epoch_steps, ncols=80,
                             desc='Train {}'.format(epoch))
        accumulated_loss = RunningAverage()

        train_indices = list(range(num_train_docs))
        random.shuffle(train_indices)
        for train_idx in train_indices:
            iters += 1
            inst = train_set[train_idx]
            iter_loss = model(inst, is_training=True)[0]
            iter_loss /= configs['batch_size']
            iter_loss.backward()
            batch_loss += iter_loss.data.item()
            if iters % configs['batch_size'] == 0:
                accumulated_loss.update(batch_loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), configs['max_grad_norm'])
                optimizer.step()
                optimizer.zero_grad()
                batch_loss = 0
                # Update progress bar
                progress.update(1)
                progress.set_postfix_str('Average Train Loss: {}'.format(accumulated_loss()))

            # Evaluation and Report
            if iters % configs['report_frequency'] == 0:
                print('Evaluation on the dev set', flush=True)
                dev_score = aida_evaluate(model, dev_set, configs)['avg']

                # Save model if it has better dev score
                if dev_score > best_dev_score:
                    best_dev_score = dev_score
                    # Save the model
                    save_path = os.path.join(configs['saved_path'], 'model.pt')
                    torch.save({'model_state_dict': model.state_dict()}, save_path)
                    print('Saved the model', flush=True)

        progress.close()

def evaluate(config_name, saved_path):
    # Prepare tokenizer, dataset, and model
    configs = prepare_configs(config_name)
    tokenizer = BertTokenizer.from_pretrained(configs['transformer'], do_basic_tokenize=False)

    # Load the AIDA data for validation/testing
    cs_filepath = 'resources/AIDA/{}_system/events_fine_all_clean.cs'.format(SYSTEM)
    ltf_dir = 'resources/AIDA/ltf'
    dev_set = load_aida_dataset(cs_filepath = cs_filepath, ltf_dir = ltf_dir, tokenizer=tokenizer)[-1]
    print('Number of dev documents (AIDA dataset) is: {}'.format(len(dev_set)))

    # Load the model
    model = BasicCorefModel(configs)
    checkpoint = torch.load(saved_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Reloaded model')

    print('Evaluation on the dev set', flush=True)
    dev_score = aida_evaluate(model, dev_set, configs)['avg']

if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('-c', '--config_name', default='basic')
    args = parser.parse_args()

    # Start training
    train(args.config_name)
