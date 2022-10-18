# Event Coreference Resolution (LORELEI - Chinese)

This is the API-based version. Refer to [this branch](https://github.com/laituan245/AIDA-Event-Coreference/tree/chinese) for information on the non-API version.

The docker is currently available at: https://hub.docker.com/r/laituan245/chinese_event_coref (the tag is `api`)

## Basic Instructions

### Coreference Resolution Server
A sample command for starting the coreference resolution server is shown below. Note that here the designated port number is 20250.

```
docker run --gpus '"device=1"' --rm -p 20250:20250 -v /shared:/shared laituan245/chinese_event_coref:api bash -c "/opt/conda/envs/aida_coreference/bin/python3.6 api.py --port 20250"
```

You can refer to the script `send_request.py` on how to send a request:
```
python send_request.py
```

A sample input file is available [here](https://github.com/laituan245/AIDA-Event-Coreference/blob/chinese/resources/LORELEI/sample_inputs/doc_1.jsonl).
The assumption is that **all the sentences in the same input file belong to the same document**. If you need to do event coreference resolution for multiple documents, please call the docker for each document one at a time.

A sample output file is available [here](https://github.com/laituan245/AIDA-Event-Coreference/blob/chinese/resources/LORELEI/sample_outputs/doc_1.jsonl). The field `clusters` contains the coreference results. For example, in the sample output file, the value of the field is `[["ev1"], ["ev10"], ["ev2", "ev3"], ["ev4"], ["ev5", "ev6", "ev8", "ev9"], ["ev7"]]`. This means that the event mentions ev5, ev6, ev8, and ev9 are predicted to be coreferential.

### Visualization
You can use the script `visualize.py` to visualize the output. The script requires minimal dependencies, so it should be runnable as it is.

```
python visualize.py -i output.jsonl -o visualization.html
```
A sample visualization file is available [here](https://github.com/laituan245/AIDA-Event-Coreference/blob/chinese/resources/LORELEI/sample_visualizations/doc_1.html).
