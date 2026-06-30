"""Generate Sem@K and rdf:type dataset variants from a base knowledge-graph dataset.

The base dataset's train/test/validation splits (read from ``abox/splits/*.nt``) are
first mirrored to ``.tsv`` in place, then four self-contained variants are written next
to the base directory, each with its own rebuilt id mappings:

    _NOSEM         no Sem@K filter, no rdf:type triples
    _NO_SEM_TYPE   no Sem@K filter, rdf:type triples added to the train split only
    _SEM           Sem@K filter applied, no rdf:type triples
    _SEM_TYPE      Sem@K filter applied, rdf:type triples added to the train split only

The Sem@K filter keeps a triple (h, r, t) only when both h and t have at least one
asserted class and r declares both a domain and a range; the surviving triples are
shuffled and re-split 80/10/10. The *_TYPE variants additionally materialise
(individual, rdf:type, class) triples in the train split for every asserted class of
the individuals occurring in the dataset.

Usage:
    python make_dataset_variants.py [<base_dataset_dir>]

If no directory is given, a built-in default dataset is used.
"""

import json
import random
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))
from jdex.loaders.torch import KnowledgeGraph

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
SEED = 42

DEFAULT_DATASET = Path(__file__).resolve().parent / "datasets" / "ARCO_20_ROFF"


def read_nt(path):
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith(" ."):
                line = line[:-2].strip()
            parts = line.split(" ", 2)
            if len(parts) != 3:
                continue
            h, r, t = (p.strip("<>").strip('"') for p in parts)
            triples.append((h, r, t))
    return triples


def write_nt(path, triples):
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"<{h}> <{r}> <{t}> .\n")


def write_tsv(path, triples):
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


def _typed(uri, kg):
    return len(kg.individual_classes(kg.individual_to_id(uri)).tolist()) > 0


def _has_domain_range(uri, kg):
    rid = kg.obj_prop_to_id(uri)
    return (len(kg.obj_prop_domain(rid).tolist()) > 0
            and len(kg.obj_prop_range(rid).tolist()) > 0)


def sem_filter(triples, kg):
    return [(h, r, t) for (h, r, t) in triples
            if _typed(h, kg) and _typed(t, kg) and _has_domain_range(r, kg)]


def make_plain_splits(base):
    return {s: read_nt(base / "abox" / "splits" / f"{s}.nt")
            for s in ("train", "test", "valid")}


def make_sem_splits(base, kg):
    pool = []
    for s in ("train", "test", "valid"):
        pool += read_nt(base / "abox" / "splits" / f"{s}.nt")
    pool = sem_filter(pool, kg)
    random.seed(SEED)
    random.shuffle(pool)
    n = len(pool)
    train = pool[:int(n * 0.8)]
    test = pool[int(n * 0.8):int(n * 0.9)]
    valid = pool[int(n * 0.9):]
    ents = {h for h, _, _ in train} | {t for _, _, t in train}
    rels = {r for _, r, _ in train}

    def ok(tr):
        return [(h, r, t) for (h, r, t) in tr if h in ents and t in ents and r in rels]

    return {"train": train, "test": ok(test), "valid": ok(valid)}


def type_triples_for(individuals, class_assertions):
    out = []
    for ind in individuals:
        for c in class_assertions.get(ind, []):
            out.append((ind, RDF_TYPE, c))
    return out


def build_mappings(splits):
    ents, rels = set(), set()
    for tr in splits.values():
        for h, r, t in tr:
            ents.add(h); ents.add(t); rels.add(r)
    ent2id = {e: i for i, e in enumerate(sorted(ents))}
    rel2id = {r: i for i, r in enumerate(sorted(rels))}
    return ent2id, rel2id


def write_variant(base, out, splits, ent2id, rel2id):
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(base, out)
    for stale in out.glob("ont_train_graph.owl*"):
        stale.unlink()
    sp = out / "abox" / "splits"
    for s, tr in splits.items():
        write_nt(sp / f"{s}.nt", tr)
        write_tsv(sp / f"{s}.tsv", tr)
    mp = out / "mappings"
    json.dump(ent2id, open(mp / "individual_to_id.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    json.dump(rel2id, open(mp / "object_property_to_id.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)


def main():
    if len(sys.argv) == 1:
        base = DEFAULT_DATASET
        print(f"No dataset given, using default: {base}")
    elif len(sys.argv) == 2:
        base = Path(sys.argv[1]).resolve()
    else:
        print("Usage: python make_dataset_variants.py [<base_dataset_dir>]")
        sys.exit(1)

    if not base.exists():
        print(f"ERROR: dataset not found: {base}")
        sys.exit(1)
    name = base.name
    parent = base.parent

    plain = make_plain_splits(base)

    sp_base = base / "abox" / "splits"
    for s, tr in plain.items():
        write_tsv(sp_base / f"{s}.tsv", tr)
    print(f"Converted base splits to .tsv: {base.name}")

    kg = KnowledgeGraph(path=base)
    ca_path = base / "abox" / "class_assertions.json"
    class_assertions = json.load(open(ca_path, encoding="utf-8")) if ca_path.exists() else {}

    sem = make_sem_splits(base, kg)

    variants = {
        f"{name}_NOSEM":       (plain, False),
        f"{name}_NO_SEM_TYPE": (plain, True),
        f"{name}_SEM":         (sem,   False),
        f"{name}_SEM_TYPE":    (sem,   True),
    }

    for vname, (src, add_type) in variants.items():
        splits = {k: list(v) for k, v in src.items()}
        if add_type:
            if not class_assertions:
                print(f"WARNING: no class_assertions.json -> {vname} will have no type triples")
            inds = ({h for tr in splits.values() for (h, _, _) in tr}
                    | {t for tr in splits.values() for (_, _, t) in tr})
            splits["train"] = splits["train"] + type_triples_for(inds, class_assertions)
        ent2id, rel2id = build_mappings(splits)
        write_variant(base, parent / vname, splits, ent2id, rel2id)
        print(f"Created {vname}: "
              f"train={len(splits['train'])} valid={len(splits['valid'])} test={len(splits['test'])} "
              f"| entities={len(ent2id)} relations={len(rel2id)}")


if __name__ == "__main__":
    main()
