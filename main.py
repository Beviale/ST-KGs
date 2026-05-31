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
import shutil
from pykeen.evaluation import InconsistencyMetric


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
    test_complete_nt_path = dataset_path / "abox" / "splits" / "test_complete.nt"
    test_complete_tsv_path = dataset_path / "abox" / "splits" / "test_complete.tsv"
    test_sem_nt_path = dataset_path / "abox" / "splits" / "test_sem_at_k.nt"
    test_sem_tsv_path = dataset_path / "abox" / "splits" / "test_sem_at_k.tsv"
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
        if os.path.exists(test_sem_nt_path):
            nt_to_tsv(test_sem_nt_path, test_sem_tsv_path)
        if os.path.exists(test_complete_nt_path):
            nt_to_tsv(test_complete_nt_path, test_complete_tsv_path)
        nt_to_tsv(valid_nt_path, valid_tsv_path)
    else:
        if not Path.exists(train_tsv_path):
            nt_to_tsv(train_nt_path, train_tsv_path)
        if not Path.exists(test_tsv_path):
            nt_to_tsv(test_nt_path, test_tsv_path)
        if not Path.exists(test_sem_tsv_path) and os.path.exists(test_sem_nt_path):
            nt_to_tsv(test_sem_nt_path, test_sem_tsv_path)
        if not Path.exists(test_complete_tsv_path) and os.path.exists(test_complete_nt_path):
            nt_to_tsv(test_complete_nt_path, test_complete_tsv_path)
        if not Path.exists(valid_tsv_path):
            nt_to_tsv(valid_nt_path, valid_tsv_path)

def select_dataset(base_path: str) -> str:
    datasets = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    print("Select a dataset:")
    for i, dataset in enumerate(datasets):
        print(f"  {i+1}. {dataset}")
    
    choice = int(input("Select a dataset: ")) - 1
    return Path(os.path.join(base_path, datasets[choice]))



def prepare_for_sem_at_k(dataset_path, kg):
    test_path_complete = Path(dataset_path) / "abox" / "splits" / "test_complete.nt"
    test_path_sem = Path(dataset_path) / "abox" / "splits" / "test_sem_at_k.nt"
    with open(test_path_complete, "r") as f_in, open(test_path_sem, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            head, relation, tail = parts[0].replace("<", "").replace(">", ""), parts[1].replace("<", "").replace(">", ""), parts[2].replace("<", "").replace(">", "")
            if not set(kg.individual_classes(kg.individual_to_id(head)).tolist()):
                continue
            if not set(kg.individual_classes(kg.individual_to_id(tail)).tolist()):
                continue
            if not set(kg.obj_prop_domain(kg.obj_prop_to_id(relation)).tolist()):
                continue
            if not set(kg.obj_prop_range(kg.obj_prop_to_id(relation)).tolist()):
                continue
            f_out.write(line + "\n")


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
    path_complete = Path(dataset_path) / "abox" / "splits" / "test_complete.nt"
    path_test = Path(dataset_path) / "abox" / "splits" / "test.nt"
    if not os.path.exists(path_complete):
        shutil.copy(path_test, path_complete)
    else:
        shutil.copy(path_complete, path_test)

    create_tsv_files(dataset_path, True)
    kg = KnowledgeGraph(
        path=dataset_path
    )
    prepare_for_sem_at_k(dataset_path, kg)
    create_tsv_files(dataset_path, True)
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    output_directory = f"results_{model_selected}_{dataset_name}"
    os.makedirs(output_directory, exist_ok=True)
    print(f"Dataset selected: {dataset_name}")
    create_tsv_files(dataset_path, True)
    test_sem = False
    repeat = True
    while(repeat):
        choice = input("Do you want to use the test set with untyped entities and relations filtered out? (yes/no): ")
        if choice.lower() == "no":
            test_sem = False
            repeat=False
            source = Path(dataset_path) / "abox" / "splits" / "test_complete.nt"
            destination = Path(dataset_path) / "abox" / "splits" / "test.nt"
            shutil.copy(source, destination)
            source = Path(dataset_path) / "abox" / "splits" / "test_complete.tsv"
            destination = Path(dataset_path) / "abox" / "splits" / "test.tsv"
            shutil.copy(source, destination)
        elif choice.lower() == "yes":
            test_sem = True
            repeat=False
            source = Path(dataset_path) / "abox" / "splits" / "test_sem_at_k.nt"
            destination = Path(dataset_path) / "abox" / "splits" / "test.nt"
            shutil.copy(source, destination)
            source = Path(dataset_path) / "abox" / "splits" / "test_sem_at_k.tsv"
            destination = Path(dataset_path) / "abox" / "splits" / "test.tsv"
            shutil.copy(source, destination)
        else:
            print("Incorrect choice!")
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
    print("\n")

    if test_sem:
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
                print("Repeat the insertion")
            if metric_selection is None:
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
                print("Repeat the insertion")
            if metric_selection is None:
                repeat = True
            else: 
                repeat = False

    
    entity_to_id_path = dataset_path / pc.INDIVIDUAL_MAPPINGS
    relation_to_id_path = dataset_path / pc.OBJ_PROP_MAPPINGS
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
        best_model_path = train_and_evaluate_TransE.train_TransE(dataset_path, entity_mapping, relation_mapping, experiments, output_directory)
        train_and_evaluate_TransE.evaluate_inc_best_model_TransE(ontology_path, train_path, output_kg_path, reasoner_path, best_model_path, dataset_path, entity_to_id_path, relation_to_id_path, output_directory, kg, metric_selection)
    elif model_selected == "BoxE":
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
        output_dir_scoring = Path(output_directory) / "temp_scoring"
        #best_weights_dir_path, best_params = train_and_evaluate_BoxE.train_BoxE(dataset_path, dataset_name, experiments, output_directory)
        best_weights_dir_path = Path("BoxE") / f"weights_{dataset_name}" 
        train_and_evaluate_BoxE.evaluate_inc_best_model_BoxE(ontology_path, train_path, output_kg_path, reasoner_path, best_weights_dir_path, dataset_name, output_dir_scoring, 64, entity_to_id_path, relation_to_id_path, kg, metric_selection)



if __name__ == "__main__":
    main()


