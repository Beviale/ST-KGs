import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
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
from pykeen.hpo import hpo_pipeline
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
import shutil
import subprocess
import itertools

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



def train_TransE(dataset_path: str, entity_mapping, relation_mapping, experiments, output_directory):
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

    best_metric = -1.0  
    best_pipeline_result = None
    best_params = None
    print(f"Number of hypeparameter combinations to test: {len(experiments)}\n")
    for i, params in enumerate(experiments):
        print(f"--- Experiment {i+1}/{len(experiments)} ---")
        print(f"Current hyparameter values: {params}")

        try:
            result = pipeline(
                model="TransE",
                training=train_tf,
                validation=valid_tf,
                testing=test_tf,
                optimizer="Adam",
                optimizer_kwargs=dict(lr=params["lr"]),
                model_kwargs=dict(embedding_dim=params["embedding_dim"]),
                training_kwargs=100,
                device="cuda",
                evaluator=RankBasedEvaluator,  # Passi la classe
                evaluator_kwargs=dict(
                    metrics=["mrr", HitsAtK(k=1), HitsAtK(k=3), HitsAtK(k=10)],
                    filtered=True,
                ),
            )
            current_metric = result.metric_results.get_metric("hits@10")
            print(f"Hits@10 obtained: {current_metric:.4f}")

            if current_metric > best_metric:
                best_metric = current_metric
                best_pipeline_result = result
                best_params = params
                print("New BEST found!")

        except Exception as e:
            print(f"Error while experimenting with parameters {params}: {e}")
            continue

    print("\n" + "=" * 40)
    print("GRID SEARCH COMPLETED")
    print(f"Best Hits@10: {best_metric:.4f}")
    print(f"Best hyparameter combination: {best_params}")
    print("=" * 40)
    print(result.metric_results.to_flat_dict())
    os.makedirs(output_directory, exist_ok=True)
    metrics_file_path = os.path.join(output_directory, "best_metrics_TransE.json")
    with open(metrics_file_path, "w", encoding="utf-8") as f:
        json.dump(result.metric_results, f, indent=4, sort_keys=True)

    if best_pipeline_result is not None:
        os.makedirs(os.path.join(output_directory, "Wights"), exist_ok=True)
        best_pipeline_result.save_to_directory(output_directory)
        print(f"\nBest model successfully saved in directory: '{output_directory}'")
    return best_pipeline_result

def evaluate_inc_best_model_TransE(ontology_path: str, train_path:str, output_kg_path: str, reasoner_path: str,  best_model, dataset_path: str, entity_mapping, relation_mapping, entity_to_id_path, relation_to_id_path, output_directory):
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
    for k in [1, 3, 10]:
        inc_evaluator = InconsistentEvaluator(ontology_path, train_path, output_kg_path, reasoner_path, InconsistencyMetric.INC_AT_K, k, filtered=True)
        inc_results = inc_evaluator.evaluate(
            model=best_model.model,
            mapped_triples=test_tf.mapped_triples,
            additional_filter_triples=[train_tf.mapped_triples, valid_tf.mapped_triples],
        )
        print(f"Inc@{k}: {str(inc_results.data)}")
        metrics_file_path = os.path.join(output_directory, "Inconsistent_Metrics_TransE.txt")
        with open(metrics_file_path, "w", encoding="utf-8") as f:
            f.write(f"Inc@{k}: {inc_results.data}\n")

        inc_evaluator = InconsistentEvaluator(ontology_path, train_path, output_kg_path, reasoner_path, entity_to_id_path, relation_to_id_path, InconsistencyMetric.SEM_AT_K, k, filtered=True)
        inc_results = inc_evaluator.evaluate(
            model=best_model.model,
            mapped_triples=test_tf.mapped_triples,
            additional_filter_triples=[train_tf.mapped_triples, valid_tf.mapped_triples],
        )
        print(f"Sem@{k}: {str(inc_results.data)}")
        metrics_file_path = os.path.join(output_directory, "Inconsistent_Metrics_TransE.txt")
        with open(metrics_file_path, "w", encoding="utf-8") as f:
            f.write(f"Sem@{k}: {inc_results.data}\n")

        inc_evaluator = InconsistentEvaluator(ontology_path, train_path, output_kg_path, reasoner_path, entity_to_id_path, relation_to_id_path, InconsistencyMetric.AC_AT_K_1, k, filtered=True)
        inc_results = inc_evaluator.evaluate(
            model=best_model.model,
            mapped_triples=test_tf.mapped_triples,
            additional_filter_triples=[train_tf.mapped_triples, valid_tf.mapped_triples],
        )
        print(f"Ac@{k}_1: {str(inc_results.data)}")
        metrics_file_path = os.path.join(output_directory, "Inconsistent_Metrics_TransE.txt")
        with open(metrics_file_path, "w", encoding="utf-8") as f:
            f.write(f"Ac@{k}_1: {inc_results.data}\n")

        inc_evaluator = InconsistentEvaluator(ontology_path, train_path, output_kg_path, reasoner_path, entity_to_id_path, relation_to_id_path, InconsistencyMetric.AC_AT_K_2, k, filtered=True)
        inc_results = inc_evaluator.evaluate(
            model=best_model.model,
            mapped_triples=test_tf.mapped_triples,
            additional_filter_triples=[train_tf.mapped_triples, valid_tf.mapped_triples],
        )
        print(f"Ac@{k}_2: {str(inc_results.data)}")
        metrics_file_path = os.path.join(output_directory, "Inconsistent_Metrics_TransE.txt")
        with open(metrics_file_path, "w", encoding="utf-8") as f:
            f.write(f"Ac@{k}_2: {inc_results.data}\n")


def train_BoxE(dataset_path: str, dataset_name: str):
    dataset_dir = Path("Boxe") / "Datasets" / dataset_name
    dataset_dir_multi = Path("Boxe") / "DatasetsMulti" / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir_multi.mkdir(parents=True, exist_ok=True)
    train_tsv_path = dataset_path / "abox" / "splits" / "train.tsv"
    train_txt_path = dataset_dir / "train.txt"
    train_txt_path_multi = dataset_dir_multi / "train.txt"
    if train_tsv_path.exists():
        shutil.copy2(train_tsv_path, train_txt_path)
        print(f"File successfully overwritten at: {train_txt_path}")
        shutil.copy2(train_tsv_path, train_txt_path_multi)
        print(f"File successfully overwritten at: {train_txt_path_multi}")
    else:
        print(f"Error: The source file '{train_tsv_path}' was not found.")

    valid_tsv_path = dataset_path / "abox" / "splits" / "valid.tsv"
    valid_txt_path = dataset_dir / "valid.txt"
    valid_txt_path_multi = dataset_dir_multi / "valid.txt"
    if valid_tsv_path.exists():
        shutil.copy2(valid_tsv_path, valid_txt_path)
        print(f"File successfully overwritten at: {valid_txt_path}")
        shutil.copy2(valid_tsv_path, valid_txt_path_multi)
        print(f"File successfully overwritten at: {valid_txt_path_multi}")
    else:
        print(f"Error: The source file '{valid_tsv_path}' was not found.")

    test_tsv_path = dataset_path / "abox" / "splits" / "test.tsv"
    test_txt_path = dataset_dir / "test.txt"
    test_txt_path_multi = dataset_dir_multi / "test.txt"
    if test_tsv_path.exists():
        shutil.copy2(test_tsv_path, test_txt_path)
        print(f"File successfully overwritten at: {test_txt_path}")
        shutil.copy2(test_tsv_path, test_txt_path_multi)
        print(f"File successfully overwritten at: {test_txt_path_multi}")
    else:
        print(f"Error: The source file '{test_tsv_path}' was not found.")

    #Pre-processing
    command = f"conda activate boxe && cd Boxe && python KBUtils.py"
    result = subprocess.run(
        command,
        shell=True, 
        text=True,
        capture_output=False, 
    )
    if result.returncode == 0:
        print("=== PREPROCESSING COMPLETED SUCCESSFULLY ===")
    else:
        print(
        f"=== ERROR DURING PREPROCESSING (Exit Code: {result.returncode}) ==="
    )
    


def main():
    repeat = True
    while(repeat):
        model_selected = None
        print("1. TransE")
        print("2. BoxE")
        model_selection = input("Select a model: ")
        if model_selection == "1":
            model_selected = "TransE"
        elif model_selection == "2":
             model_selected = "BoxE"
        else:
            print("Repeat the insertion")
        if model_selected is None:
            repeat = True
        else: 
            repeat = False

    dataset_path = select_dataset("datasets")
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    output_directory = f"results_{model_selected}_{dataset_name}"
    os.makedirs(output_directory, exist_ok=True)
    print(f"Dataset selected: {dataset_name}")
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


    grid_hyperparameters = {
        "embedding_dim": [50, 100, 200],
        "lr": [1e-2, 1e-3],
    }
    keys, values = zip(*grid_hyperparameters.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    ontology_path = dataset_path / "ontology.owl"
    train_path = dataset_path / "abox" / "splits" / "train.nt"
    output_kg_path = dataset_path / "ont_train_graph.owl"
    reasoner_path = Path().absolute() / "reasoners"

    if model_selected == "TransE":
        best_model = train_TransE(dataset_path, entity_mapping, relation_mapping, experiments, output_directory)
        evaluate_inc_best_model_TransE(ontology_path, train_path, output_kg_path, reasoner_path, best_model, entity_to_id_path, relation_to_id_path, output_directory)
    elif model_selected == "BoxE":
        train_BoxE(dataset_path, dataset_name)

        


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


def execute_subprocess(script_path: str, working_dir: str):
    try:
        process = subprocess.run(
        [sys.executable, str(script_path.resolve())],
        cwd=working_dir,
        check=True,          # Solleva eccezione se il codice di uscita != 0
        capture_output=True, # Cattura output e errori
        text=True            # Converte in stringhe
    )
        
        print("Script finished successfully!")
        print("Output:", process.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running the script in {working_dir}: {e}")
        print(f"Error output: {e.stderr}")

if __name__ == "__main__":
    main()


