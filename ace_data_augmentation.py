import json
import copy
import random
from os.path import join

input_dir = 'resources/ACE05-E'
output_dir = 'resources/ACE05-E-Augmented'
k = 3

id2sents = {}
train_ids, dev_ids, test_ids = set(), set(), set()

# Read ground-truth data files
for split in ['train', 'dev', 'test']:
    path = join(input_dir, '{}.oneie.json'.format(split))
    with open(path, 'r', encoding='utf-8') as r:
        for line in r:
            sent_inst = json.loads(line)
            doc_id = sent_inst['doc_id']
            # Update id2sents
            if not doc_id in id2sents:
                id2sents[doc_id] = []
            id2sents[doc_id].append(line)
            # Update train_ids, dev_ids, test_ids
            if split == 'train': train_ids.add(doc_id)
            if split == 'dev': dev_ids.add(doc_id)
            if split == 'test': test_ids.add(doc_id)


# Create augmented data files
for split in ['train', 'dev', 'test']:
    if split == 'train': doc_ids = train_ids
    if split == 'dev': doc_ids = dev_ids
    if split == 'test': doc_ids = test_ids

    path = join(output_dir, '{}.oneie.json'.format(split))
    with open(path, 'w+', encoding='utf-8') as f:
        # Write original docs
        for doc_id in doc_ids:
            for line in id2sents[doc_id]:
                f.write(line)

        # Start doing augmentation
        for doc_id in doc_ids:
            for i in range(k):
                new_doc_id = doc_id + '_augmented_{}'.format(i)
                new_lines = copy.deepcopy([str(line) for line in id2sents[doc_id]])
                random.shuffle(new_lines)
                for sent_index, line in enumerate(new_lines):
                    sent_inst = json.loads(line)
                    sent_inst['doc_id'] = new_doc_id
                    sent_inst['sent_id'] = new_doc_id + '-' + str(sent_index)
                    new_line = json.dumps(sent_inst).strip()
                    f.write('{}\n'.format(new_line))
