import os
import json
import numpy as np

from argparse import ArgumentParser

def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Main Code
if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', default='resources/LORELEI/event_outputs_sep8.jsonl')
    parser.add_argument('-o', '--output', default='resources/LORELEI/sample_inputs')
    parser.add_argument('-n', '--num', default=5, type=int)
    args = parser.parse_args()
    create_dir_if_not_exists(args.output)

    # Read the original input file
    docid2lines = {}
    doc_ids = set()
    with open(args.input, 'r') as f:
        for line in f:
            x = json.loads(line)
            doc_id = x['sent_id'].rsplit('.', 1)[0]
            sent_id = int(x['sent_id'].rsplit('.', 1)[1])
            if not doc_id in docid2lines:
                docid2lines[doc_id] = []
            del x['sent_id']
            docid2lines[doc_id].append((sent_id, x))
            if len(docid2lines[doc_id]) >= 2:
                assert(docid2lines[doc_id][-1][0] == docid2lines[doc_id][-2][0] + 1)
            doc_ids.add(doc_id)
    print('Number of documents: {}'.format(len(doc_ids)))
    doc_ids = list(doc_ids)[:args.num]

    for i in range(args.num):
        doc_id = doc_ids[i]
        lines = docid2lines[doc_id]
        lines = [x[1] for x in lines]
        with open(os.path.join(args.output, 'doc_{}.jsonl'.format(i+1)), 'w+') as f:
            for x in lines:
                f.write(json.dumps(x))
                f.write('\n')