cp /shared/nas/data/m1/tuanml2/cross_lingual_event_coref/model.pt model.pt
docker build --tag laituan245/es_event_coref .
docker push laituan245/es_event_coref
docker build --tag laituan245/spanbert_coref .
docker push laituan245/spanbert_coref
