import nltk
from utils import *

class Document:
    def __init__(self, doc_id, sentences, event_mentions, entity_mentions, pred_graphs):
        self.doc_id = doc_id
        self.sentences = sentences
        self.words = flatten(sentences)
        self.event_mentions = event_mentions
        self.entity_mentions = entity_mentions
        self.num_words = len(self.words)
        self.pred_graphs = pred_graphs

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
                        })
            assert(len(_arguments) == len(e['arguments']))
            e['arguments'] = _arguments

        # Update self.events
        self.events = {}
        for event_mention in event_mentions:
            mention_id = event_mention['id']
            event_id = mention_id[:mention_id.rfind('-')]
            if not event_id in self.events:
                self.events[event_id] = []
            self.events[event_id].append(event_mention)

        # Build self.coreferential_pairs
        self.coreferential_pairs = set()
        for i in range(len(event_mentions)):
            for j in range(i+1, len(event_mentions)):
                # Find the event id of the first event mention
                mention_i = event_mentions[i]
                mention_id_i = mention_i['id']
                event_id_i = mention_id_i[:mention_id_i.rfind('-')]
                # Find the event id of the second event mention
                mention_j = event_mentions[j]
                mention_id_j = mention_j['id']
                event_id_j = mention_id_j[:mention_id_j.rfind('-')]
                # Check if refer to the same event
                if event_id_i == event_id_j:
                    loc_i = (mention_i['trigger']['start'], mention_i['trigger']['end'])
                    loc_j = (mention_j['trigger']['start'], mention_j['trigger']['end'])
                    self.coreferential_pairs.add((loc_i, loc_j))
                    self.coreferential_pairs.add((loc_j, loc_i))

        # Extract pred_triggers, pred_entities, pred_relations, pred_event_mentions
        assert(len(pred_graphs) == 0 or len(pred_graphs) == len(sentences))
        self.pred_trigges, self.pred_entities = [], []
        self.pred_relations, self.pred_event_mentions = [], []
        for graph in pred_graphs:
            if len(graph) > 0:
                for trigger in graph['triggers']:
                    self.pred_trigges.append({
                        'tokens': self.words[trigger[0]:trigger[1]],
                        'start': trigger[0], 'end': trigger[1],
                        'confidence': trigger[3]
                    })
                    self.pred_event_mentions.append({
                        'event_type': trigger[2],
                        'trigger': self.pred_trigges[-1],
                        'arguments': []
                    })
                for entity in graph['entities']:
                    self.pred_entities.append({
                        'tokens': self.words[entity[0]:entity[1]],
                        'start': entity[0], 'end': entity[1],
                        'entity_type': entity[2], 'mention_type': entity[3],
                        'confidence': entity[4]
                    })
                for relation in graph['relations']:
                    arg1 = self.pred_entities[relation[0]]
                    arg2 = self.pred_entities[relation[1]]
                    self.pred_relations.append({
                        'arg1': arg1, 'arg2': arg2,
                        'relation_type': relation[2],
                        'confidence': relation[3]
                    })
                for role in graph['roles']:
                    event_mention = self.pred_event_mentions[role[0]]
                    entity = self.pred_entities[role[1]]
                    event_mention['arguments'].append({
                        'entity': entity,
                        'role': role[2],
                        'confidence': role[-1]
                    })

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
            doc_tokens = tokenizer.tokenize(' '.join(doc.words))
            doc_token_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
            doc.token_windows, doc.mask_windows = \
                convert_to_sliding_window(doc_token_ids, sliding_window_size, tokenizer)
            doc.input_masks = extract_input_masks_from_mask_windows(doc.mask_windows)

            # Compute the starting index of each word
            doc.word_starts_indexes = []
            for index, word in enumerate(doc_tokens):
                if not word.startswith('##'):
                    doc.word_starts_indexes.append(index)
            assert(len(doc.word_starts_indexes) == len(doc.words))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
