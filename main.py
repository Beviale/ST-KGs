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


def main():
    dataset_name = input("Select the dataset name: ")
    cwd = Path.cwd()
    dataset_path = cwd / "WHOW_5_ROFF"
    print("Dataset path: " + str(dataset_path))
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
            num_epochs=3,  
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
    inc_evaluator = InconsistentEvaluator(ontology_path, train_path, output_kg_path, reasoner_path, entity_to_id_path, relation_to_id_path, InconsistencyMetric.SEM_ATK_K, k=5, filtered=True)
    inc_results = inc_evaluator.evaluate(
        model=result.model,
        mapped_triples=test_tf.mapped_triples,
        additional_filter_triples=[train_tf.mapped_triples, valid_tf.mapped_triples],
    )
    print(f"Inc@5: {str(inc_results.data)}")


if __name__ == "__main__":
    main()


