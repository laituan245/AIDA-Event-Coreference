# Event Coreference Resolution (LORELEI - Chinese)

The docker is currently available at: https://hub.docker.com/r/laituan245/chinese_event_coref

## Basic Instructions

### Coreference Resolution
A sample command for conducting **within-doc** event coreference resolution is shown below.

```
docker run --gpus '"device=1"' --rm -v /shared:/shared laituan245/chinese_event_coref -i input.jsonl -o output.jsonl
```

A sample input file is available [here](https://github.com/laituan245/AIDA-Event-Coreference/blob/chinese/resources/LORELEI/sample_inputs/doc_1.jsonl).
The assumption is that **all the sentences in the same input file belong to the same document**. If you need to do event coreference resolution for multiple documents, please call the docker for each document one at a time.

A sample output file is available [here](https://github.com/laituan245/AIDA-Event-Coreference/blob/chinese/resources/LORELEI/sample_outputs/doc_1.jsonl). The field `clusters` contains the coreference results. For example, in the sample output file, the value of the field is `[["ev1"], ["ev10"], ["ev2", "ev3"], ["ev4"], ["ev5", "ev6", "ev8", "ev9"], ["ev7"]]`. This means that the event mentions ev5, ev6, ev8, and ev9 are predicted to be coreferential.

### Visualization
You can use the script `visualize.py` to visualize the output. The script requires minimal dependencies, so it should be runnable as it is.

```
python visualize.py -i output.jsonl -o visualization.html
```
