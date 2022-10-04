import torch
import pyhocon
import json
import numpy as np

from os.path import join
from transformers import AutoTokenizer
from models import BasicCorefModel
from utils import prepare_configs, flatten
from data import load_aida_dataset
from data.base import Dataset, Document
from scorer import get_predicted_antecedents
from argparse import ArgumentParser
from shutil import copyfile
from aida_event_coref import generate_coref_preds

SAVED_PATH = 'model.pt'

def read_lorelei_jsonl(input_fp, tokenizer):
    doc_ids = set()
    docid2sentid = {}
    docid2sentences = {}
    docid2events = {}
    docid2tokenoffset = {}
    ev_count = 0
    default_doc_id = 'document_1'
    #tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
    with open(input_fp, 'r') as f:
        for line in f:
            x = json.loads(line)
            if 'sent_id' in x:
                doc_id = x['sent_id'].rsplit('.', 1)[0]
                doc_ids.add(doc_id)
                sent_id = int(x['sent_id'].rsplit('.', 1)[1])
            else:
                doc_id = default_doc_id
                doc_ids.add(doc_id)
                sent_id = 1 + docid2sentid.get(doc_id, 0)
            assert(sent_id == 1 + docid2sentid.get(doc_id, 0))
            docid2sentid[doc_id] = sent_id
            tokens = tokenizer(x['sentence'])
            if not doc_id in docid2events:
                docid2events[doc_id] = []
            token_offset = docid2tokenoffset.get(doc_id, 0)
            for e in x['events']:
                trigger = e['trigger']
                token_start = tokens.char_to_token(trigger[0])
                token_end = tokens.char_to_token(trigger[1]-1)
                ev_count += 1
                docid2events[doc_id].append({
                    'id': 'ev{}'.format(ev_count),
                    'trigger': {
                        'start': token_start + token_offset - 1,
                        'end': token_end + token_offset
                    },
                    'original_text': x['sentence'][trigger[0]:trigger[1]],
                    'event_type': trigger[-1],
                    'arguments': [],
                    'doc_id': doc_id,
                })
            token_offset += len(tokenizer.tokenize(x['sentence']))
            docid2tokenoffset[doc_id] = token_offset
            tokens = tokenizer.tokenize(x['sentence'])
            if not doc_id in docid2sentences:
                docid2sentences[doc_id] = []
            docid2sentences[doc_id].append(tokens)
    # Create Document
    data = []
    for doc_id in doc_ids:
        doc = Document(doc_id, docid2sentences[doc_id], docid2events[doc_id], [], 'LORELEI')
        data.append(doc)
    return Dataset(data, tokenizer)

# Main Code
if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', default='resources/LORELEI/event_outputs_sep8.jsonl')
    parser.add_argument('-o', '--output', default='resources/LORELEI/event_outputs_sep8_coref.jsonl')
    args = parser.parse_args()

    # Initialize configs, tokenizer, and model
    configs = prepare_configs('basic')
    tokenizer = AutoTokenizer.from_pretrained(
        configs['transformer'],
        do_basic_tokenize=False
    )
    model = BasicCorefModel(configs)
    if SAVED_PATH:
        if torch.cuda.is_available():
            checkpoint = torch.load(SAVED_PATH)
        else:
            checkpoint = torch.load(SAVED_PATH, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Reloaded model')
    print('Initialized configs, tokenizer, and model')

    # Load data
    test = read_lorelei_jsonl(args.input, tokenizer)
    print('Loaded test data ({} docs)'.format(len(test.data)))

    # Extract clusters
    predictions, pair_scores = generate_coref_preds(model, test)
    all_clusters = []
    for p in predictions.values():
        all_clusters.append(p['predicted_clusters'])
    all_clusters = flatten(all_clusters)
    for c in all_clusters:
        c.sort(key=lambda x: x['id'])
    all_clusters.sort(key=lambda x: x[0]['id'])

    # Prepare data for final output
    docid2data = {}
    for doc in test.data:
        docid2data[doc.doc_id] = doc
        doc.clusters = []
    for cluster in all_clusters:
        # Sanity check
        cur_doc_id = cluster[0]['doc_id']
        for mention in cluster:
            assert(mention['doc_id'] == cur_doc_id)
        # Update docid2data
        cur_clusters = [m['id'] for m in cluster]
        docid2data[cur_doc_id].clusters.append(cur_clusters)

    # Write to a jsonl output file
    with open(args.output, 'w') as f:
        for doc_id in docid2data:
            cur_doc = docid2data[doc_id]
            f.write(json.dumps({
                'doc_id': cur_doc.doc_id,
                'clusters': cur_doc.clusters,
                'words': cur_doc.words,
                'event_mentions': cur_doc.event_mentions
            }))
            f.write('\n')