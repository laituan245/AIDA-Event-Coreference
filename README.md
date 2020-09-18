# Event Coreference Resolution


The models were trained on ACE-2005 dataset and then applied directly to AIDA data.

```
docker build --tag laituan245/spanbert_coref .
docker push laituan245/spanbert_coref
docker run --gpus '"device=1"' --rm -v /shared:/shared laituan245/spanbert_coref -i input.cs -c output.cs -t output.tab -l ltf_dir
```
