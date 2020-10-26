import os
import json

from os import listdir
from utils import read_ltf
from os.path import isfile, join
from data.base import Dataset, Document

def load_oneie_dataset(
        base_path, tokenizer, dataset_name, remove_doc_with_no_events=True
    ):
    id2split, id2sents = {}, {}

    # Read ground-truth data files
    for split in ['train', 'dev', 'test']:
        path = join(base_path, '{}.oneie.json'.format(split))
        with open(path, 'r', encoding='utf-8') as r:
            for line in r:
                sent_inst = json.loads(line)
                doc_id = sent_inst['doc_id']
                id2split[doc_id] = split
                # Update id2sents
                if not doc_id in id2sents:
                    id2sents[doc_id] = []
                id2sents[doc_id].append(sent_inst)

    # Parse documents one-by-one
    train, dev, test = [], [], []
    for doc_id in id2sents:
        words_ctx, pred_trigger_ctx, pred_entities_ctx = 0, 0, 0
        sents = id2sents[doc_id]
        sentences, event_mentions, entity_mentions = [], [], []
        for sent_index, sent in enumerate(sents):
            sentences.append(sent['tokens'])
            # Parse entity mentions
            for entity_mention in sent['entity_mentions']:
                entity_mention['start'] += words_ctx
                entity_mention['end'] += words_ctx
                entity_mentions.append(entity_mention)
            # Parse event mentions
            for event_mention in sent['event_mentions']:
                event_mention['sent_index'] = sent_index
                event_mention['trigger']['start'] += words_ctx
                event_mention['trigger']['end'] += words_ctx
                event_mentions.append(event_mention)
            # Update words_ctx
            words_ctx += len(sent['tokens'])
        doc = Document(doc_id, sentences, event_mentions, entity_mentions, dataset_name)
        split = id2split[doc_id]
        if split == 'train':
            if not remove_doc_with_no_events or len(event_mentions) > 0:
                train.append(doc)
        if split == 'dev': dev.append(doc)
        if split == 'test': test.append(doc)

    # Convert to Document class
    train, dev, test = Dataset(train, tokenizer), Dataset(dev, tokenizer), Dataset(test, tokenizer)

    return train, dev, test


# AIDA
def str_to_location(location_str):
    doc_id, text_loc = location_str.split(':')
    text_start, text_end = text_loc.split('-')
    return doc_id, int(text_start), int(text_end)
