# Event Coreference Resolution


The models were trained on ACE-2005 dataset and then applied directly to AIDA data.

```
docker build --tag laituan245/spanbert_coref .
docker push laituan245/spanbert_coref
docker run --rm -v /shared:/shared laituan245/spanbert_coref -i input.cs -o coref.cs -l ltf_dir
```
