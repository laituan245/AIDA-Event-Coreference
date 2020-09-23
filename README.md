# Event Coreference Resolution


The models were trained on ERE-ES dataset and then applied directly to AIDA data.

```
docker build --tag laituan245/es_event_coref .
docker push laituan245/es_event_coref
docker run --gpus '"device=1"' --rm -v /shared:/shared laituan245/es_event_coref -i input.cs -c output.cs -t output.tab -l ltf_dir
```
