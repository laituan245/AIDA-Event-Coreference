cp /shared/nas/data/m1/tuanml2/chinese_event_coref/model.pt model.pt
docker build --tag laituan245/chinese_event_coref .
docker push laituan245/chinese_event_coref
