import os
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import jdex.utils.conventions.paths as pc
import json
import os
from jdex.loaders.torch import KnowledgeGraph
import itertools
import train_and_evaluate_BoxE
import train_and_evaluate_TransE
import train_and_evaluate_TransOWL
import shutil
from pykeen.evaluation import InconsistencyMetric
import random


def nt_to_tsv(input_path: str, output_path: str) -> None:
    """Convert a.nt file in a .tsv file."""
    with open(input_path, "r", encoding="utf-8") as nt_file, \
         open(output_path, "w", encoding="utf-8") as tsv_file:
                
        for line in nt_file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            if line.endswith(" ."):
                line = line[:-2].strip()
            
            parts = line.split(" ", 2)
            if len(parts) != 3:
                continue
            
            head, relation, tail = parts
            
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
    
    print("Select a dataset:")
    for i, dataset in enumerate(datasets):
        print(f"  {i+1}. {dataset}")
    
    choice = int(input("Select a dataset: ")) - 1
    return Path(os.path.join(base_path, datasets[choice]))

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def is_sem_variant(dataset_name: str) -> bool:
    """Whether the dataset is a Sem@K-filtered variant, from its name suffix."""
    if dataset_name.endswith("_NO_SEM_TYPE"):
        return False
    if dataset_name.endswith("_SEM_TYPE"):
        return True
    if dataset_name.endswith("_NOSEM"):
        return False
    if dataset_name.endswith("_SEM"):
        return True
    return False


def has_type_triples(relation_mapping: dict) -> bool:
    """Whether the dataset contains rdf:type triples (classes are entities)."""
    return RDF_TYPE in relation_mapping


def main():
    repeat = True
    while(repeat):
        model_selected = None
        print("1. TransE")
        print("2. BoxE")
        print("3. TransOWL")
        model_selection = input("Select a model: ")
        if model_selection == "1":
            model_selected = "TransE"
        elif model_selection == "2":
             model_selected = "BoxE"
        elif model_selection == "3":
             model_selected = "TransOWL"
        else:
            print("Repeat the insertion")
        if model_selected is None:
            repeat = True
        else: 
            repeat = False
    dataset_path = select_dataset("datasets")
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    output_directory = f"results_{dataset_name}"
    os.makedirs(output_directory, exist_ok=True)
    print(f"Dataset selected: {dataset_name}")

    create_tsv_files(dataset_path, True)
    kg = KnowledgeGraph(path=dataset_path)
    sem = is_sem_variant(dataset_name)

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
    print("\n")

    metrics = []
    if sem:
        output_directory = output_directory + "_Sem"
        repeat = True
        while(repeat):
            metric_selected = None
            metric_selection = None
            print("1. Inc@K")
            print("2. Sem@k")
            print("3. AC@K (Inc@k)")
            print("4. AC@K (Sem@k)")


            metric_selected = input("Select a metric: ")
            if metric_selected == "1":
                metric_selection = InconsistencyMetric.INC_AT_K
            elif metric_selected == "2":
                metric_selection = InconsistencyMetric.SEM_AT_K
            elif metric_selected == "3":
                metric_selection = InconsistencyMetric.AC_AT_K_1
            elif metric_selected == "4":
                metric_selection = InconsistencyMetric.AC_AT_K_2
            else:
                print("ERROR! Repeat the insertion!")
            if metric_selection is None:
                repeat = True
            else: 
                if metric_selection not in metrics:
                    metrics.append(metric_selection)
                another = input("Do you want to choose another metric? (yes/no): ")
                if another.lower() == "yes":
                    repeat = True
                else:
                    repeat = False
    else:
        repeat = True
        while(repeat):
            metric_selected = None
            metric_selection = None
            print("1. Inc@K")
            print("2. AC@K (Inc@k)")

            metric_selected = input("Select a metric: ")
            if metric_selected == "1":
                metric_selection = InconsistencyMetric.INC_AT_K
            elif metric_selected == "2":
                metric_selection = InconsistencyMetric.AC_AT_K_1
            else:
                print("ERROR! Repeat the insertion!")
            if metric_selection is None:
                repeat = True
            else: 
                if metric_selection not in metrics:
                    metrics.append(metric_selection)
                another = input("Do you want to choose another metric? (yes/no): ")
                if another.lower() == "yes":
                    repeat = True
                else:
                    repeat = False

   
    entity_to_id_path = Path(dataset_path) / pc.INDIVIDUAL_MAPPINGS
    relation_to_id_path = Path(dataset_path) / pc.OBJ_PROP_MAPPINGS
    with open(entity_to_id_path, "r") as f:
        entity_mapping = json.load(f)
    with open(relation_to_id_path, "r") as f:
        relation_mapping = json.load(f)
    ontology_path = dataset_path / "ontology.owl"
    train_path = dataset_path / "abox" / "splits" / "train.nt"
    output_kg_path = dataset_path / "ont_train_graph.owl"
    reasoner_path = Path().absolute() / "reasoners"
    if model_selected == "TransE":
        grid_hyperparameters = {
            "embedding_dim": [50, 100],     
            "lr":            [1e-3, 5e-4], 
            "margin":        [1.0, 2.0],
            "num_negs":      [32, 64],
        }
        keys, values = zip(*grid_hyperparameters.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
        #best_model_path = train_and_evaluate_TransE.train_TransE(dataset_path, entity_mapping, relation_mapping, experiments, output_directory)
        best_model_path = Path(r"C:\Users\bevia\Documents\GitHub\ST-KGs\results_DBPEDIA_50K_C_ROFF\TransE\Best") 
        train_and_evaluate_TransE.evaluate_inc_best_model_TransE(ontology_path, train_path, output_kg_path, reasoner_path, best_model_path, dataset_path, entity_to_id_path, relation_to_id_path, output_directory, kg, metrics)
    elif model_selected == "BoxE":
        rules = False
        choice = input("Do you want to use the rules? (yes/no) ")
        if choice.lower() == "yes":
            rules = True
            

        #grid_hyperparameters = {
            #"learningRate": [1e-3, 5e-4], 
            #"loss_margin": [3.0, 6.0, 9.0], 
            #"nbNegExp": [25, 50],  
            #"reg_lambda": [0.05, 0.01, 0.005]
        #}
        grid_hyperparameters = {
            "learningRate": [1e-3], 
            "loss_margin": [3.0], 
            "nbNegExp": [25],  
            "reg_lambda": [0.05]
        }

        keys, values = zip(*grid_hyperparameters.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
        temp_dir = Path(output_directory) / "temp_scoring"
        best_weights_dir_path, best_params = train_and_evaluate_BoxE.train_BoxE(dataset_path, dataset_name, experiments, output_directory, ontology_path, kg, rules)
        best_weights_dir_path = Path("BoxE") / f"weights_{dataset_name}"
        train_and_evaluate_BoxE.evaluate_inc_best_model_BoxE(ontology_path, train_path, output_kg_path, reasoner_path, best_weights_dir_path, dataset_name, output_directory, temp_dir, 64, entity_to_id_path, relation_to_id_path, kg, metrics)
    elif model_selected == "TransOWL":
        grid_hyperparameters = {
            "embedding_dim": [50, 100],
            "lr":            [1e-3, 5e-4],
            "margin":        [1.0, 2.0],
            "num_negs":      [32, 64],
            "reg_weight":    [1.0, 0.1],   # weight lambda of the axiom-based regularization
            "subprop_weight":[0.01],       # weight of the subPropertyOf term
            "subclass_weight":[0.01],      # weight of the subClassOf term (only on *_TYPE datasets)
            "beta":          [0.9],        # directional offset (1 - beta); beta=1 -> Option A
        }
        keys, values = zip(*grid_hyperparameters.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
        best_model_path = train_and_evaluate_TransOWL.train_TransOWL(
            dataset_path, entity_mapping, relation_mapping, experiments,
            output_directory, ontology_path, kg
        )
        best_dir = Path(output_directory) / "TransOWL" / "Best"
        train_and_evaluate_TransOWL.evaluate_inc_best_model_TransOWL(
            ontology_path, train_path, output_kg_path, reasoner_path, best_dir,
            dataset_path, entity_to_id_path, relation_to_id_path,
            output_directory, kg, metrics
        )


if __name__ == "__main__":
    main()


