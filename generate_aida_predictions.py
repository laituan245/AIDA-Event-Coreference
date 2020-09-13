import torch
import pyhocon
import json

from os.path import join
from transformers import *
from models import BasicCorefModel
from utils import prepare_configs, flatten
from data import load_aida_dataset
from scorer import get_predicted_antecedents

# Constants
LIFU_SYSTEM = 'lifu'
YING_SYSTEM = 'ying'
SYSTEM = YING_SYSTEM
CONFIG_NAME = 'basic'
SAVED_PATH = None


# Helper Functions
def generate_coref_preds(model, data, output_path='predictions.json', split='TEST'):
    predictions = {}
    for inst in data:
        doc_words = inst.words
        event_mentions = inst.event_mentions
        preds = model(inst, is_training=False)[1]
        preds = [x.cpu().data.numpy() for x in preds]
        top_antecedents, top_antecedent_scores = preds[2:]
        predicted_antecedents = get_predicted_antecedents(top_antecedents, top_antecedent_scores)

        predicted_clusters, m2cluster = [], {}
        for ix, e in enumerate(event_mentions):
            if predicted_antecedents[ix] < 0:
                cluster_id = len(predicted_clusters)
                predicted_clusters.append([e])
            else:
                antecedent_idx = predicted_antecedents[ix]
                p_e = event_mentions[antecedent_idx]
                cluster_id = m2cluster[p_e['id']]
                predicted_clusters[cluster_id].append(e)
            m2cluster[e['id']] = cluster_id
        # Update predictions
        predictions[inst.doc_id] = {}
        predictions[inst.doc_id]['words']= doc_words
        predictions[inst.doc_id]['predicted_clusters'] = predicted_clusters

    with open(output_path, 'w+') as outfile:
        json.dump(predictions, outfile)

def generate_visualizations(sample_outputs, output_path='visualization.html'):
    with open(sample_outputs) as json_file:
        data = json.load(json_file)

    with open(output_path, 'w+') as output_file:
        doc_ids = list(data.keys())
        doc_ids.sort()
        for doc_id in doc_ids:
            doc = data[doc_id]
            doc_words = doc['words']
            clusters = doc['predicted_clusters']
            event_mentions = flatten(clusters)
            output_file.write('<b>Document {}</b><br>'.format(doc_id))
            output_file.write('{}<br><br><br>'.format(doc_to_html(doc, event_mentions)))
            for ix, cluster in enumerate(doc['predicted_clusters']):
                if len(cluster) == 1: continue
                output_file.write('<b>Cluster {}</b></br>'.format(ix+1))
                for em in cluster:
                    output_file.write('{}<br>'.format(event_mentions_to_html(doc_words, em)))
                output_file.write('<br><br>')
            output_file.write('<br><hr>')

def doc_to_html(doc, event_mentions):
    doc_words = doc['words']
    doc_words = [str(word) for word in doc_words]
    for e in event_mentions:
        t_start, t_end = e['trigger']['start'], e['trigger']['end'] - 1
        doc_words[t_start] = '<span style="color:blue">' + doc_words[t_start]
        doc_words[t_end] = doc_words[t_end] + '({})</span>'.format(e['id'])
    return ' '.join(doc_words)

def event_mentions_to_html(doc_words, em):
    trigger_start = em['trigger']['start']
    trigger_end = em['trigger']['end']
    context_left = ' '.join(doc_words[trigger_start-10:trigger_start])
    context_right = ' '.join(doc_words[trigger_end:trigger_end+10])
    final_str = context_left + ' <span style="color:red">' + em['trigger']['text'] + '</span> ' + context_right
    final_str = '<i>Event {} (Type {}) </i> | '.format(em['id'], em['event_type']) + final_str
    return final_str

# Main Code
if __name__ == "__main__":
    configs = prepare_configs(CONFIG_NAME)
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(configs['transformer'], do_basic_tokenize=False)
    # Load AIDA dataset
    cs_filepath = 'resources/AIDA/{}_system/events_fine_all_clean.cs'.format(SYSTEM)
    ltf_dir = 'resources/AIDA/ltf'
    test = load_aida_dataset(cs_filepath = cs_filepath, ltf_dir = ltf_dir, tokenizer=tokenizer)[-1]
    # Load model
    model = BasicCorefModel(configs)
    if SAVED_PATH:
        checkpoint = torch.load(SAVED_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Reloaded model')
    # Generate predictions and visualizations
    generate_coref_preds(model, test, 'predictions_{}.json'.format(SYSTEM))
    generate_visualizations('predictions_{}.json'.format(SYSTEM), output_path='visualization_{}.html'.format(SYSTEM))
