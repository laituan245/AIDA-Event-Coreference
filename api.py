import os
import gc
import argparse
import logging
import uuid
import json
import lorelei_event_coref

import concurrent.futures
from shutil import rmtree
from flask import Flask, request

app = Flask(__name__)
logger = logging.getLogger()

TMP_DIR = 'tmp'

def process_data(data):
    # Create tmp dir
    run_tmp_dir = os.path.join(TMP_DIR, str(uuid.uuid4()))
    os.makedirs(run_tmp_dir, exist_ok=True)
    logger.info('Created tmp output directory: {}'.format(run_tmp_dir))

    # Write an input file
    input_fp = os.path.join(run_tmp_dir, 'input.jsonl')
    with open(input_fp, 'w+') as input_f:
        for d in data:
            input_f.write('{}\n'.format(json.dumps(d)))

    # Processing
    output_fp = os.path.join(run_tmp_dir, 'output.jsonl')
    lorelei_event_coref.main(input_fp, output_fp)

    # Read the output file
    final_output = []
    with open(output_fp, 'r') as output_f:
        for line in f:
            final_output.append(json.loads(line))

    # Remove the tmp dir
    rmtree(run_tmp_dir)
    return final_output

@app.route('/process', methods=['POST'])
def process():
    form = request.get_json()
    data = form.get('data')

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(process_data, data)
        final_output = future.result()

    gc.collect()

    return final_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=20202)
    args = parser.parse_args()

    logger.info('done.')
    logger.info('start...')
    app.run('0.0.0.0', port=int(args.port))

