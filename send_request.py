import os
import glob
import json
import requests

from os.path import join

def read_data(jsonl_file):
    data = []
    f = open(jsonl_file, 'r')
    for line in f:
        data.append(json.loads(line))
    f.close()
    return data

input_data = read_data('resources/LORELEI/sample_inputs/doc_1.jsonl')

response = requests.post('http://localhost:25202/process', json={'data': input_data})
with open('sample_response.txt', 'w+') as f:
    f.write(response.text)

