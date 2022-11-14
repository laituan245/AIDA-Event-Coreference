import os
import json
import visualize

from os.path import join
from copy import deepcopy
from argparse import ArgumentParser

from deep_translator import GoogleTranslator

CTX = 15

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_visualization(output_fp, sents, clusters):

    print('working on', output_fp)
    _id2event = {}
    with open(output_fp, 'w+') as f:
        f.write('<h2>Document</h2>')
        for sent in sents:
            f.write(sent['sentence'] + ' ')
        f.write('\n<br><br>\n')
        for sent in sents:
            for e in sent['events']:
                _id2event[e['_id']] = {
                    'trigger': e['trigger'],
                    'sentence': sent
                }
        cluster_count = 0
        f.write('Event mentions are highlighted in <span style="color:red">red</span>.<br>\n')
        for cs in clusters:
            if len(cs) > 1:
                cluster_count += 1
                f.write('<h3>Non-singleton cluster {}</h3>'.format(cluster_count))
                for e in cs:
                    f.write('<i>Event {} (Type {})</i>: '.format(e, _id2event[e]['trigger'][-1]))
                    orig_sent = _id2event[e]['sentence']['sentence']
                    start, end = _id2event[e]['trigger'][:2]

                    outtext = orig_sent[max(0, start-CTX):start]
                    outtext += '<span style="color:red">'+ orig_sent[start:end] + '</span>'
                    outtext += orig_sent[end:min(end+CTX, len(orig_sent))]
                    f.write(outtext)


                    translated = GoogleTranslator(source='auto', target='en').translate(outtext)
                    f.write('\n<br>')
                    f.write('<span style="color:orange">Translated context:</span>')
                    #f.write(_id2event[e]['sentence']['english'])
                    if '<span style="color:red"></span>' in translated:
                        return False
                    f.write(translated)
                    f.write('\n<br><br>')
                f.write('\n<br><br>\n')
    return True

# Main Code
if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('-i', '--base_dir', default='resources/demo')
    args = parser.parse_args()

    # Paths
    coref_input_dir = join(args.base_dir, 'inputs')
    coref_outputs_dir = join(args.base_dir, 'outputs')
    demo_dir = join(args.base_dir, 'visualization')
    translation_fp = join(args.base_dir, 'brichi-oct2022.json')
    create_dir_if_not_exists(demo_dir)

    # Read translation file
    chinese2eng = {}
    with open(translation_fp, 'r') as f:
        for line in f:
            translated_d = json.loads(line)
            try:
                chinese = translated_d['_source']['event_annotations']['tokens']
            except:
                chinese = translated_d['_source']['originaltext']
            try:
                english = translated_d['_source']['translatedtext']
                chinese2eng[chinese] = english
            except:
                pass


    # Process
    dir_list = os.listdir(coref_outputs_dir)
    for fn in dir_list:
        demo_vis_fp = join(demo_dir, fn.replace('.coref', '.html'))
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
        clusters = data[0]['clusters']

        # read original input file
        has_errors = False
        original_input_f = join(coref_input_dir, fn.replace('.coref', '.args'))
        l_datas = []
        ev_count = 0
        all_ev_ids = set()
        with open(original_input_f, 'r') as f:
            for line in f:
                l_data = json.loads(line)
                try:
                    english = chinese2eng[l_data['sentence']]
                    l_data['english'] = english
                    l_datas.append(l_data)
                    for e in l_data['events']:
                        ev_count += 1
                        e['_id'] = 'ev{}'.format(ev_count)
                        all_ev_ids.add(e['_id'])
                except:
                    has_errors = True
        if has_errors: continue
        for cs in clusters:
            for _e in cs:
                assert(_e in all_ev_ids)

        # Generate visualization
        succeeded = generate_visualization(demo_vis_fp, l_datas, clusters)
        if not succeeded:
            os.remove(demo_vis_fp)

