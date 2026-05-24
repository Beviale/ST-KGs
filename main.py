import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import numpy as np
import jdex.utils.conventions.paths as pc
import json
from pykeen.triples import TriplesFactory
from rdflib import Graph
import os
from pykeen.pipeline import pipeline
from pykeen.evaluation import RankBasedEvaluator
from pykeen.evaluation import InconsistentEvaluator
from pykeen.evaluation import InconsistencyMetric
from pykeen.metrics.ranking import HitsAtK
from pykeen.predict import predict_target
import pandas as pd
from tqdm import tqdm
import pandas as pd
from rdflib import Graph, URIRef
import random
from jdex.loaders.torch import KnowledgeGraph

def nt_to_tsv(input_path: str, output_path: str) -> None:
    """Convert a.nt file in a .tsv file."""
    with open(input_path, "r", encoding="utf-8") as nt_file, \
         open(output_path, "w", encoding="utf-8") as tsv_file:
                
        for line in nt_file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # rimuovi il punto finale
            if line.endswith(" ."):
                line = line[:-2].strip()
            
            # splitta in h, r, t
            parts = line.split(" ", 2)
            if len(parts) != 3:
                continue
            
            head, relation, tail = parts
            
            # rimuovi < > dagli URI
            head = head.strip("<>")
            relation = relation.strip("<>")
            tail = tail.strip("<>").strip('"')
            
            tsv_file.write(f"{head}\t{relation}\t{tail}\n")

def create_tsv_files(dataset_path: str, force_tsv_creation: bool):
    train_nt_path = dataset_path / "abox" / "splits" / "train.nt"
    train_tsv_path = dataset_path / "abox" / "splits" / "train.tsv"
    test_nt_path = dataset_path / "abox" / "splits" / "test.nt"
    test_tsv_path = dataset_path / "abox" / "splits" / "test.tsv"
    valid_nt_path = dataset_path / "abox" / "splits" / "valid.nt"
    valid_tsv_path = dataset_path / "abox" / "splits" / "valid.tsv"

    if not Path.exists(train_nt_path):
        raise ValueError(f"The path '{train_nt_path}' does not exist!")
    if not Path.exists(test_nt_path):
        raise ValueError(f"The path '{test_nt_path}' does not exist!")
    if not Path.exists(valid_nt_path):
        raise ValueError(f"The path '{valid_nt_path}' does not exist!")

    if force_tsv_creation:
        nt_to_tsv(train_nt_path, train_tsv_path)
        nt_to_tsv(test_nt_path, test_tsv_path)
        nt_to_tsv(valid_nt_path, valid_tsv_path)
    else:
        if not Path.exists(train_tsv_path):
            nt_to_tsv(train_nt_path, train_tsv_path)
        if not Path.exists(test_tsv_path):
            nt_to_tsv(test_nt_path, test_tsv_path)
        if not Path.exists(valid_tsv_path):
            nt_to_tsv(valid_nt_path, valid_tsv_path)

def select_dataset(base_path: str) -> str:
    datasets = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    print("Available dataset:")
    for i, dataset in enumerate(datasets):
        print(f"  {i+1}. {dataset}")
    
    choice = int(input("Select a dataset: ")) - 1
    return Path(os.path.join(base_path, datasets[choice]))


def select_subgraph(graph_path: str, out_dir: None, ratio = 50):
    if out_dir is None: 
        p = Path(graph_path)
        out_dir = p.parent

    graph_triplets = parse_nt(graph_path)
    print(f"[1/4] Reading the original file...")
    print(f"      Total:  {len(graph_triplets):,} triplets")

 
    print(f"\n[2/4] Sampling triplets at {ratio*100:.0f}%...")
    sampled = sample_graph(graph_triplets, ratio)
    print(f"      Selected {len(sampled):,} triples on {len(graph_triplets):,}")
 
    print(f"\n[3/4] Computing entities and relations in sampled train...")
    train_entities, train_relations = get_entities_and_relations(sampled)
    print(f"      Entities:   {len(train_entities):,}")
    print(f"      Relations: {len(train_relations):,}")
 
    print(f"\n[4/4] Saving to '{out_dir}/'...")
    write_nt(sampled, out_dir / "sub_graph.nt")
 


def main():
    dataset_path = select_dataset("datasets")
    print(f"Dataset selected: {dataset_path}")
    create_tsv_files(dataset_path, True)
    kg = KnowledgeGraph(
        path=dataset_path
    )
    print(f"{'Dataset Component':<35} | {'Shape'}")
    print("-" * 50)
    print(f"{'Training triples':<35} | {kg.train.shape}")
    print(f"{'Test triples':<35} | {kg.test.shape}")
    print(f"{'Validation triples':<35} | {kg.valid.shape}")
    print(f"{'Class assertions':<35} | {kg.class_assertions.shape}")
    print(f"{'Taxonomy (TBox)':<35} | {kg.taxonomy.shape}")
    print(f"{'Object property hierarchy':<35} | {kg.obj_props_hierarchy.shape}")
    print(f"{'Object property domains':<35} | {kg.obj_props_domain.shape}")
    print(f"{'Object property ranges':<35} | {kg.obj_props_range.shape}")
    
    entity_to_id_path = dataset_path / pc.INDIVIDUAL_MAPPINGS
    relation_to_id_path = dataset_path / pc.OBJ_PROP_MAPPINGS

    with open(entity_to_id_path, "r") as f:
        entity_mapping = json.load(f)
    with open(relation_to_id_path, "r") as f:
        relation_mapping = json.load(f)


    train_tf = TriplesFactory.from_path(
        dataset_path / pc.TRAIN,
        entity_to_id=entity_mapping,
        relation_to_id=relation_mapping,
    )
    valid_tf = TriplesFactory.from_path(
        dataset_path / pc.VALID,
        entity_to_id=entity_mapping,
        relation_to_id=relation_mapping,
    )

    test_tf = TriplesFactory.from_path(
        dataset_path / pc.TEST,
        entity_to_id=entity_mapping,
        relation_to_id=relation_mapping,
    )

    evaluator = RankBasedEvaluator(
        metrics=["mrr", HitsAtK(k=10)],
        filtered=True,
    )

    result = pipeline(
        training=train_tf,
        validation=valid_tf, 
        testing=test_tf,        
        model="TransE",
        evaluator = evaluator,
        model_kwargs=dict(
            embedding_dim=50,
        ),
        optimizer="Adam",
        optimizer_kwargs=dict(
            lr=0.01,
        ),
        training_kwargs=dict(
            num_epochs=100,  
            batch_size=256,
            checkpoint_name='transE-checkpoint.pt',   
            checkpoint_directory=dataset_path /'checkpoints/',    
            checkpoint_on_failure=True
        ),
        
        stopper="early",
        stopper_kwargs=dict(
            frequency=5,       
            patience=2,       
            relative_delta=0.01,
            metric="mrr",     
        ),
        evaluation_kwargs=dict(
            batch_size=256,  
            use_tqdm=True,
        ),
        
        device="gpu",
        random_seed=42,
    )
    print(result.metric_results.to_flat_dict())
    ontology_path = dataset_path / "ontology.owl"
    train_path = dataset_path / "abox" / "splits" / "train.nt"
    output_kg_path = dataset_path / "ont_train_graph.owl"
    reasoner_path = Path().absolute() / "reasoners"
    class_assertsions_json_path = dataset_path / "abox" / "class_assertions.json"
    relation_json_path =  dataset_path / "rbox" / "roles_domain_range.json"
    inc_evaluator = InconsistentEvaluator(ontology_path, train_path, output_kg_path, reasoner_path, entity_to_id_path, relation_to_id_path, 
            InconsistencyMetric.SEM_AT_K, k=5, class_assertsions_json_path=class_assertsions_json_path, relation_json_path=relation_json_path, filtered=True)
    inc_results = inc_evaluator.evaluate(
        model=result.model,
        mapped_triples=test_tf.mapped_triples,
        additional_filter_triples=[train_tf.mapped_triples, valid_tf.mapped_triples],
    )
    print(f"Inc@5: {str(inc_results.data)}")



def parse_nt(path: str) -> list[tuple[str, str, str]]:
    """
    Legge un file .nt e restituisce lista di triple (s, p, o).
    Ignora commenti e righe vuote.
    Ogni tripla è una riga: <s> <p> <o> .
    """
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Rimuovi il punto finale e splitta
            if line.endswith("."):
                line = line[:-1].strip()
            parts = line.split(None, 2)
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples

def write_nt(triples: list[tuple], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s, p, o in triples:
            f.write(f"{s} {p} {o} .\n")

def get_entities_and_relations(triples):
    entities = set()
    relations = set()
    for s, p, o in triples:
        entities.add(s)
        entities.add(o)
        relations.add(p)
    return entities, relations

def sample_graph(triples, ratio, seed):
    rng = random.Random(seed)
    k = max(1, int(len(triples) * ratio))
    return rng.sample(triples, k)


if __name__ == "__main__":
    main()


