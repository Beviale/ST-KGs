import sys
import os
sys.path.insert(0, os.path.abspath("BoxE"))
import argparse
import numpy as np
import msgpack
import msgpack_numpy as m
m.patch()

from BoxEModel import BoxEMulti
from ModelOptions import ModelOptions

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--weights", type=str)
parser.add_argument("--embedding_dim", type=int)
parser.add_argument("--output", type=str)
args = parser.parse_args()

options = ModelOptions()
options.embedding_dim = args.embedding_dim
options.use_bumps = True
options.nb_neg_examples_per_pos = 0

model = BoxEMulti(args.dataset, options)
model.load_params(param_loc=args.weights)

with open(f"DatasetsMulti/{args.dataset}/test.kb", "rb") as f:
    data = msgpack.unpack(f, raw=False)
test_batch = np.array(data)

nb_entities = model.nb_entities
all_scores = []
hrt_list = []

for triple in test_batch:
    rel_id, head_id, true_tail_id = triple[0], triple[1], triple[2]
    candidates = np.zeros((nb_entities, 4), dtype=np.int32)
    candidates[:, 0] = rel_id
    candidates[:, 1] = head_id
    candidates[:, 2] = np.arange(nb_entities)
    candidates[:, 3] = 1
    scores = model.score_forward_pass(candidates, reload_params=False).flatten()
    all_scores.append(-scores)
    hrt_list.append([head_id, rel_id, true_tail_id])

np.save(os.path.join(args.output, "scores.npy"), np.stack(all_scores))
np.save(os.path.join(args.output, "hrt.npy"), np.array(hrt_list))
print("DONE")