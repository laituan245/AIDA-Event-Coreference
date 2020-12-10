import torch
import pyhocon
import json
import numpy as np

from os.path import join
from transformers import *
from models import BasicCorefModel
from utils import prepare_configs, flatten
from data import load_aida_dataset
from scorer import get_predicted_antecedents
from argparse import ArgumentParser
from shutil import copyfile

SAVED_PATH = 'model.pt'
SAMPLE_DATA_PATH = '/shared/nas/data/m1/manling2/aida_docker_test/uiuc_ie_pipeline_fine_grained/output/output_dryrun_E32_gpu/'
SAMPLE_LANGUAGE = 'en'

# Helper Functions
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def generate_coref_preds(model, data):
    predictions, pair_scores = {}, []
    for inst in data:
        doc_words = inst.words
        event_mentions = inst.event_mentions

        if len(event_mentions) == 0:
            # Corner case (when no events were extracted)
            predictions[inst.doc_id] = {}
            predictions[inst.doc_id]['words']= doc_words
            predictions[inst.doc_id]['predicted_clusters'] = []
            continue

        with torch.no_grad():
            preds = model(inst, is_training=False)[1]
        preds = [x.cpu().data.numpy() for x in preds]
        top_antecedents, top_antecedent_scores = preds[2:]
        top_antecedent_scores = sigmoid(top_antecedent_scores)
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

        # Update pair_scores
        for ix, ei in enumerate(event_mentions):
            for jx in range(ix):
                ej = event_mentions[jx]
                pair_scores.append((inst.doc_id, ei, ej, top_antecedent_scores[ix, jx+1]))

    return predictions, pair_scores

# Main Code
if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', default=SAMPLE_DATA_PATH + SAMPLE_LANGUAGE + '/event/events_fine_all_clean.cs')
    parser.add_argument('-c', '--cs_file', default='/shared/nas/data/m1/tuanml2/aida_rerun/{}/events_corefer.cs'.format(SAMPLE_LANGUAGE))
    parser.add_argument('-t', '--tab_file', default='/shared/nas/data/m1/tuanml2/aida_rerun/{}/events_corefer_confidence.tab'.format(SAMPLE_LANGUAGE))
    parser.add_argument('-l', '--ltf_dir', default=SAMPLE_DATA_PATH + 'ltf')
    args = parser.parse_args()

    # Copy original input cs file to output cs file at the beginning
    copyfile(args.input, args.cs_file)

    # Load configs
    configs = prepare_configs('basic')
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(configs['transformer'], do_basic_tokenize=False)
    # Load AIDA dataset
    test = load_aida_dataset(cs_filepath = args.input,
                             ltf_dir = args.ltf_dir,
                             tokenizer=tokenizer)[-1]
    # Load model
    model = BasicCorefModel(configs)
    if SAVED_PATH:
        if torch.cuda.is_available():
            checkpoint = torch.load(SAVED_PATH)
        else:
            checkpoint = torch.load(SAVED_PATH, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Reloaded model')

    # Extract clusters
    predictions, pair_scores = generate_coref_preds(model, test)
    all_clusters = []
    for p in predictions.values():
        all_clusters.append(p['predicted_clusters'])
    all_clusters = flatten(all_clusters)
    for c in all_clusters:
        c.sort(key=lambda x: x['id'])
    all_clusters.sort(key=lambda x: x[0]['id'])

    # Output
    event2lines = {}
    with open(args.input, 'r', encoding='utf8') as f:
        for line in f:
            es = line.strip().split('\t')
            event_id = es[0][1:]
            if not event_id in event2lines:
                event2lines[event_id] = []
            event2lines[event_id].append(line)
    with open(args.cs_file, 'w+', encoding='utf8') as f:
        for c in all_clusters:
            first_id = c[0]['id']
            for e in c:
                lines = event2lines[e['id']]
                for line in lines:
                    f.write(line.replace(':' + e['id'], ':' + first_id))

    # Output tab file
    with open(args.tab_file, 'w+', encoding='utf8') as f:
        for doc_id, e1, e2, score in pair_scores:
            loc1 = '{},{}'.format(e1['trigger']['original_start'], e1['trigger']['original_end'])
            loc2 = '{},{}'.format(e2['trigger']['original_start'], e2['trigger']['original_end'])
            f.write('{}\t{}\t{}\t{}\n'.format(doc_id, loc1, loc2, score))
