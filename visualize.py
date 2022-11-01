import json
from copy import deepcopy
from argparse import ArgumentParser

def doc_to_html(doc_words, event_mentions):
    for e in event_mentions:
        t_start, t_end = e['trigger']['start'], e['trigger']['end'] - 1
        doc_words[t_start] = '<span style="color:blue">' + doc_words[t_start]
        doc_words[t_end] = doc_words[t_end] + '</span>'
    return ' '.join(doc_words)

def event_mentions_to_html(doc_words, em):
    trigger_start = em['trigger']['start']
    trigger_end = em['trigger']['end']
    context_left = ' '.join(doc_words[trigger_start-10:trigger_start])
    context_right = ' '.join(doc_words[trigger_end:trigger_end+10])
    final_str = context_left + ' <span style="color:red">' + em['original_text'] + '</span> ' + context_right
    final_str = '<i>Event {} (Type {}) </i> | '.format(em['id'], em['event_type']) + final_str
    return final_str

def main(args_input, args_output):
    # Read input
    docs = []
    with open(args_input, 'r') as f:
        for line in f:
            docs.append(json.loads(line))

    # Write output
    with open(args_output, 'w+', encoding='utf-8') as output_file:
        for doc in docs:
            doc_id = doc['doc_id']
            doc_words = doc['words']
            clusters = doc['clusters']
            event_mentions = doc['event_mentions']
            output_file.write('<b>Document {} (Number event mentions: {})</b><br>'.format(doc_id, len(event_mentions)))
            output_file.write('{}<br><br><br>'.format(doc_to_html(deepcopy(doc_words), event_mentions)))
            for ix, cluster in enumerate(clusters):
                if len(cluster) == 1: continue
                output_file.write('<b>Cluster {}</b></br>'.format(ix+1))
                for em_id in cluster:
                    em = [_e for _e in event_mentions if _e['id'] == em_id][0]
                    output_file.write('{}<br>'.format(event_mentions_to_html(deepcopy(doc_words), em)))
                output_file.write('<br><br>')
            output_file.write('<br><hr>')


# Main Code
if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', default='resources/LORELEI/event_outputs_sep8_coref.jsonl')
    parser.add_argument('-o', '--output', default='resources/LORELEI/visualization_sep8.html')
    args = parser.parse_args()

    main(args.input, args.output)
