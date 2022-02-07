import nltk
from utils import *

def mentionid2eventid(mention_id, dataset_name):
    if dataset_name == 'ERE':
        if not '-' in mention_id: return mention_id
        es = mention_id.split('-')
        event_id = es[0] + '-' + es[2]
        return event_id
    elif dataset_name == 'ACE':
        return mention_id[:mention_id.rfind('-')]
    elif dataset_name == 'AIDA':
        return mention_id

class Document:
    def __init__(self, doc_id, sentences, event_mentions, entity_mentions, dataset_name):
        self.doc_id = doc_id
        self.sentences = sentences
        self.words = flatten(sentences)
        self.event_mentions = event_mentions
        self.entity_mentions = entity_mentions
        self.num_words = len(self.words)
        self.dataset_name = dataset_name

        # Post-process self.event_mentions
        for e in self.event_mentions:
            _arguments = []
            for argument in e['arguments']:
                for entity_mention in self.entity_mentions:
                    if entity_mention['id'] == argument['entity_id']:
                        _arguments.append({
                            #'text': argument['text'],
                            'role': argument['role'],
                            'entity': entity_mention,
                            'entity_id': argument['entity_id'],
                        })
            #assert(len(_arguments) == len(e['arguments']))
            e['arguments'] = _arguments

        # Sort by trigger start
        self.event_mentions.sort(key=lambda x: x['trigger']['start'])

        # Update self.events
        self.events = {}
        for event_mention in event_mentions:
            mention_id = event_mention['id']
            event_id = mentionid2eventid(mention_id, dataset_name)
            if not event_id in self.events:
                self.events[event_id] = []
            self.events[event_id].append(event_mention)

        # Build self.coreferential_pairs
        self.coreferential_pairs = set()
        for i in range(len(event_mentions)):
            for j in range(i+1, len(event_mentions)):
                # Find the event id of the first event mention
                mention_i = event_mentions[i]
                event_id_i = mentionid2eventid(mention_i['id'], dataset_name)
                # Find the event id of the second event mention
                mention_j = event_mentions[j]
                event_id_j = mentionid2eventid(mention_j['id'], dataset_name)
                # Check if refer to the same event
                if event_id_i == event_id_j:
                    loc_i = (mention_i['trigger']['start'], mention_i['trigger']['end'])
                    loc_j = (mention_j['trigger']['start'], mention_j['trigger']['end'])
                    self.coreferential_pairs.add((loc_i, loc_j))
                    self.coreferential_pairs.add((loc_j, loc_i))

class Dataset:
    def __init__(self, data, tokenizer, sliding_window_size = 512):
        '''
            data: A list of GroundTruthDocument
            tokenizer: A transformer Tokenizer
            sliding_window_size: Size of sliding window (for a long document, we split it into overlapping segments)
        '''
        self.data = data

        # Tokenize the documents
        for doc in self.data:
            # Build doc_tokens, doc.word_starts_indexes
            doc_tokens, word_starts_indexes, start_index = [], [], 0
            for w in doc.words:
                word_tokens = tokenizer.tokenize(w)
                doc_tokens += word_tokens
                word_starts_indexes.append(start_index)
                start_index += len(word_tokens)
            doc.word_starts_indexes = word_starts_indexes
            assert(len(doc.word_starts_indexes) == len(doc.words))

            # Build token_windows, mask_windows, and input_masks
            doc_token_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
            doc.token_windows, doc.mask_windows = convert_to_sliding_window(doc_token_ids, 512, tokenizer)
            doc.input_masks = extract_input_masks_from_mask_windows(doc.mask_windows)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
