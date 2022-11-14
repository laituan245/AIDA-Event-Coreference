import os
import json
import visualize

from os.path import join
from copy import deepcopy
from argparse import ArgumentParser

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Main Code
if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('-i', '--base_dir', default='resources/demo')
    args = parser.parse_args()

    coref_outputs_dir = join(args.base_dir, 'outputs')
    demo_dir = join(args.base_dir, 'visualization')
    create_dir_if_not_exists(demo_dir)
    dir_list = os.listdir(coref_outputs_dir)
    for fn in dir_list:
        if not fn.endswith('.coref'): continue
        raw_input_fp = join(coref_outputs_dir, fn)
        with open(raw_input_fp, 'r') as f:
            data = json.loads(f.read())['output']

        # Filtering requirements
        assert(len(data) == 1)
        if len(data[0]['clusters']) == 0:
            continue
        has_non_singleton = False
        for cs in data[0]['clusters']:
            if len(cs) > 1:
                has_non_singleton = True
        if not has_non_singleton: continue

        # Write to temp file and visualization
        with open('temp.jsonl', 'w+') as f:
            for line in data:
                f.write(json.dumps(line))
                f.write('\n')
        visualize.main(
            'temp.jsonl',
            join(demo_dir, fn.replace('.coref', '.html'))
        )
        os.remove('temp.jsonl')

