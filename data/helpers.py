import os
import json

from os import listdir
from utils import read_ltf
from os.path import isfile, join
from data.base import Dataset, Document

def load_oneie_dataset(
        base_path, tokenizer,
        predictions_path=None, remove_doc_with_no_events=True
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

    # Read prediction files (if available)
    if predictions_path:
        sentid2graph = {}
        for split in ['train', 'dev', 'test']:
            path = join(predictions_path, '{}.json'.format(split))
            with open(path, 'r', encoding='utf-8') as r:
                for line in r:
                    sent_preds = json.loads(line)
                    sentid2graph[sent_preds['sent_id']] = sent_preds['graph']

    # Parse documents one-by-one
    train, dev, test = [], [], []
    for doc_id in id2sents:
        words_ctx, pred_trigger_ctx, pred_entities_ctx = 0, 0, 0
        sents = id2sents[doc_id]
        sentences, event_mentions, entity_mentions, pred_graphs = [], [], [], []
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
            # Update pred_graphs
            if predictions_path:
                graph = sentid2graph.get(sent['sent_id'], {})
                if len(graph) > 0:
                    for entity in graph['entities']:
                        entity[0] += words_ctx
                        entity[1] += words_ctx
                    for trigger in graph['triggers']:
                        trigger[0] += words_ctx
                        trigger[1] += words_ctx
                    for relation in graph['relations']:
                        relation[0] += pred_entities_ctx
                        relation[1] += pred_entities_ctx
                    for role in graph['roles']:
                        role[0] += pred_trigger_ctx
                        role[1] += pred_entities_ctx
                    pred_trigger_ctx += len(graph['triggers'])
                    pred_entities_ctx += len(graph['entities'])
                pred_graphs.append(graph)
            # Update words_ctx
            words_ctx += len(sent['tokens'])
        doc = Document(doc_id, sentences, event_mentions, entity_mentions, pred_graphs)
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

def load_aida_dataset(cs_filepath, ltf_dir, tokenizer):
    # Load CS file
    event2info, entity2info = {}, {}
    with open(cs_filepath, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            assert(line.startswith(':Event_'))
            es = line.split('\t')
            event_id = es[0][1:]
            if not event_id in event2info: event2info[event_id] = {}
            if es[1] == 'mention.actual':
                text, location_str, confidence = es[2:]
                doc_id, text_start, text_end = str_to_location(location_str)
                event2info[event_id]['actual_text'] = text[1:-1]
                event2info[event_id]['doc_id'] = doc_id
                event2info[event_id]['text_start'] = int(text_start)
                event2info[event_id]['text_end'] = int(text_end)
            elif es[1] == 'type':
                event2info[event_id]['type'] = es[2]
            elif es[1] != 'canonical_mention.actual':
                argument_role, entity_id, entity_loc_str, confidence = es[1:]
                entity_doc_id, entity_start, entity_end = str_to_location(entity_loc_str)
                assert(doc_id == entity_doc_id) # Sanity check
                if not entity_id in entity2info:
                    entity2info[entity_id] = {
                        'doc_id': doc_id, 'id': entity_id,
                        'start': entity_start, 'end': entity_end
                    }

    # Determine aida_doc_ids
    aida_doc_ids = []
    for file in os.listdir(ltf_dir):
        if file.endswith('ltf.xml'):
            aida_doc_ids.append(file[:-8])

    # Read LTF folder
    train, dev, test = [], [], []
    for doc_id in aida_doc_ids:
        ltf_filepath = join(ltf_dir, '{}.ltf.xml'.format(doc_id))
        tokens = read_ltf(ltf_filepath)
        words = [token[-1] for token in tokens]
        start2word, end2word = {}, {}
        for i in range(len(tokens)):
            start2word[tokens[i][0]] = i
            end2word[tokens[i][1]] = i

        # Fix entity mentions start/end
        entity_mentions
        for entity in entity2info.values():
            if entity['doc_id'] == doc_id:
                entity['start'] = start2word[entity['start']]
                entity['end'] = end2word[entity['end']] + 1
                entity['text'] = ' '.join([t[-1] for t in tokens[entity['start']:entity['end']]])
                entity_mentions.append(entity)

        # Extract event mentions
        event_mentions = []
        for eid in event2info:
            info = event2info[eid]
            if info['doc_id'] == doc_id:
                event_mention = {'id': eid, 'event_type': info['type'], 'arguments': []}
                event_mention['trigger'] = {'text': info['actual_text']}
                event_mention['trigger']['start'] = start2word[info['text_start']]
                event_mention['trigger']['end'] = end2word[info['text_end']] + 1
                event_mentions.append(event_mention)
        event_mentions.sort(key=lambda x: x['trigger']['start'])

        # Create a new documents
        doc = Document(doc_id, [words], event_mentions, entity_mentions, [])
        doc.pred_event_mentions = doc.event_mentions
        test.append(doc)

    # Convert to Document class
    train, dev, test = Dataset(train, tokenizer), Dataset(dev, tokenizer), Dataset(test, tokenizer)

    return train, dev, test
