# Event Coreference Resolution


The models were trained on ACE05-E and ERE-ES datasets and then applied directly to AIDA data.

```
docker run --gpus '"device=1"' --rm -v /shared:/shared laituan245/es_event_coref -i input.cs -c output.cs -t output.tab -l ltf_dir
docker run --gpus '"device=1"' --rm -v /shared:/shared laituan245/spanbert_coref -i input.cs -c output.cs -t output.tab -l ltf_dir
```
