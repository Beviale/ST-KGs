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
parser.add_argument("--type", type=str)
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
print(test_batch.shape)
print(test_batch[0])

nb_entities = model.nb_entities
all_scores = []
hrt_list = []

if args.type.lower() == "tail":
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
elif args.type.lower() == "head":
    for triple in test_batch:
        rel_id, true_head_id, tail_id = triple[0], triple[1], triple[2]
        candidates = np.zeros((nb_entities, 4), dtype=np.int32)
        candidates[:, 0] = rel_id
        candidates[:, 1] = np.arange(nb_entities)
        candidates[:, 2] = tail_id
        candidates[:, 3] = 1
        scores = model.score_forward_pass(candidates, reload_params=False).flatten()
        all_scores.append(-scores)
        hrt_list.append([true_head_id, rel_id, tail_id])
else:
    raise Exception("Argument type value not valid!")

np.save(os.path.join(args.output, f"scores_{args.type}.npy"), np.stack(all_scores))
np.save(os.path.join(args.output, f"hrt_{args.type}.npy{args.type}"), np.array(hrt_list))
print("DONE")