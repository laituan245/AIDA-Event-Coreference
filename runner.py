import os
import math
import torch
import tqdm
import random
import copy

from transformers import *
from models import BasicCorefModel
from utils import RunningAverage, prepare_configs
from scorer import evaluate_coref
from data import load_oneie_dataset
from argparse import ArgumentParser

PRETRAINED_MODEL = 'model.pt'

# Main Functions
def train(config_name):
    # Prepare tokenizer, dataset, and model
    configs = prepare_configs(config_name)
    tokenizer = AutoTokenizer.from_pretrained(
            configs['transformer'],
            do_basic_tokenize=False
    )
    print('Loaded tokenizer {}'.format(configs['transformer']))

    # Load the ACE05-CN dataset
    train_set, dev_set, test_set = load_oneie_dataset('resources/ACE05-CN', tokenizer, dataset_name='ACE')
    print('[ACE05-CN dataset] Train/Dev/Test size is: {}/{}/{}'.format(len(train_set), len(dev_set), len(test_set)))

    # Load the model
    model = BasicCorefModel(configs)
    print('Initialized tokenier, dataset, and model')

    if PRETRAINED_MODEL and os.path.exists(PRETRAINED_MODEL):
        checkpoint = torch.load(PRETRAINED_MODEL)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print('Reloaded pretrained ckpt')
        with torch.no_grad():
            # Evaluation on the English dataset
            print('Evaluation on the (English) test set')
            evaluate_coref(model, test_set, configs)

            # Evaluation on the Spanish dataset
            print('Evaluation on the (Spanish) test set')
            evaluate_coref(model, es_test_set, configs)

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

        progress.close()

        # Evaluation and Report
        print('Evaluation on the dev set')
        dev_score = evaluate_coref(model, dev_set, configs)['avg']

        # Save model if it has better dev score
        if dev_score > best_dev_score:
            best_dev_score = dev_score
            # Save the model
            save_path = os.path.join('model.pt')
            torch.save({'model_state_dict': model.state_dict()}, save_path)
            print('Saved the model', flush=True)


if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('-c', '--config_name', default='basic')
    args = parser.parse_args()

    # Start training
    train(args.config_name)
