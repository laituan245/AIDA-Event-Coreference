import torch

def get_span_emb(context_features, span_starts, span_ends):
    num_tokens = context_features.size()[0]

    features = []
    for s, e in zip(span_starts, span_ends):
        sliced_features = context_features[s:e, :]
        features.append(torch.mean(sliced_features, dim=0, keepdim=True))
    features = torch.cat(features, dim=0)
    return features

def compute_place_conflict(e1, e2):
    place1 = [arg for arg in e1['arguments'] if 'place' in arg['role'].lower()]
    place2 = [arg for arg in e2['arguments'] if 'place' in arg['role'].lower()]
    if len(place1) > 0 and len(place2) > 0:
        if place1[0]['entity_id'] != place2[0]['entity_id']:
            return 1
    return 0

def compute_coref_num(e1, e2):
    coref_num = 0
    args1 = sorted(e1['arguments'], key=lambda x: x['entity_id'])
    args2 = sorted(e2['arguments'], key=lambda x: x['entity_id'])
    for arg1, arg2 in zip(args1, args2):
        if arg1['entity_id'] == arg2['entity_id'] \
        and arg1['role'] != arg2['role']:
            coref_num += 1
    return coref_num

def compute_overlap_args(e1, e2):
    overlap_args = 0
    args1, args2 = e1['arguments'], e2['arguments']

    used = set()
    for arg1 in args1:
        for i2, arg2 in enumerate(args2):
            if i2 in used: continue
            if arg1['role'] == arg2['role'] \
            and arg1['entity_id'] == arg2['entity_id']:
                overlap_args += 1
                used.add(i2)
                break
    return overlap_args

def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.
    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.
    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs
