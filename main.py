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
import random


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
    
    print("Select a dataset:")
    for i, dataset in enumerate(datasets):
        print(f"  {i+1}. {dataset}")
    
    choice = int(input("Select a dataset: ")) - 1
    return Path(os.path.join(base_path, datasets[choice]))

def is_valid_triple(line, train_entities, train_relations):
    parts = line.split()
    head = parts[0].replace("<", "").replace(">", "")
    relation = parts[1].replace("<", "").replace(">", "")
    tail = parts[2].replace("<", "").replace(">", "")
    return head in train_entities and tail in train_entities and relation in train_relations

def prepare_for_sem_at_k(dataset_path, kg):
    test_path_complete = Path(dataset_path) / "abox" / "splits" / "test_complete.nt"
    train_path_complete = Path(dataset_path) / "abox" / "splits" / "train_complete.nt"
    valid_path_complete = Path(dataset_path) / "abox" / "splits" / "valid_complete.nt"

    test_path_sem = Path(dataset_path) / "abox" / "splits" / "sem_at_k.nt"
    train_path_sem = Path(dataset_path) / "abox" / "splits" / "train_sem_at_k.nt"
    valid_path_sem = Path(dataset_path) / "abox" / "splits" / "valid_sem_at_k.nt"

    sem_triples = []
    with open(test_path_complete, "r") as f_in:
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
            sem_triples.append(line)

    with open(train_path_complete, "r") as f_in:
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
            sem_triples.append(line)

    with open(valid_path_complete, "r") as f_in:
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
            sem_triples.append(line)


    random.seed(42)
    random.shuffle(sem_triples)

    n = len(sem_triples)
    train_end = int(n * 0.8)
    valid_end = int(n * 0.9)

    train_triples = sem_triples[:train_end]
    valid_triples = sem_triples[valid_end:]
    test_triples = sem_triples[train_end:valid_end]

    train_entities = set()
    train_relations = set()
    for line in train_triples:
        parts = line.split()
        head = parts[0].replace("<", "").replace(">", "")
        relation = parts[1].replace("<", "").replace(">", "")
        tail = parts[2].replace("<", "").replace(">", "")
        train_entities.add(head)
        train_entities.add(tail)
        train_relations.add(relation)

    # filtra valid e test
    valid_triples = [t for t in valid_triples if is_valid_triple(t, train_entities, train_relations)]
    test_triples = [t for t in test_triples if is_valid_triple(t, train_entities, train_relations)]

    # salva
    with open(train_path_sem, "w") as f:
        f.write("\n".join(train_triples))
    with open(valid_path_sem, "w") as f:
        f.write("\n".join(valid_triples))
    with open(test_path_sem, "w") as f:
        f.write("\n".join(test_triples))

    print(f"(For Sem@K) Train: {len(train_triples)}, Valid: {len(valid_triples)}, Test: {len(test_triples)}")

    


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
    output_directory = f"results_{dataset_name}"
    os.makedirs(output_directory, exist_ok=True)
    print(f"Dataset selected: {dataset_name}")
    # Test
    path_complete = Path(dataset_path) / "abox" / "splits" / "test_complete.nt"
    path_test = Path(dataset_path) / "abox" / "splits" / "test.nt"
    if not os.path.exists(path_complete):
        shutil.copy(path_test, path_complete)
    else:
        shutil.copy(path_complete, path_test)
    # Train
    path_complete = Path(dataset_path) / "abox" / "splits" / "train_complete.nt"
    path_test = Path(dataset_path) / "abox" / "splits" / "train.nt"
    if not os.path.exists(path_complete):
        shutil.copy(path_test, path_complete)
    else:
        shutil.copy(path_complete, path_test)
    # Val
    path_complete = Path(dataset_path) / "abox" / "splits" / "valid_complete.nt"
    path_test = Path(dataset_path) / "abox" / "splits" / "valid.nt"
    if not os.path.exists(path_complete):
        shutil.copy(path_test, path_complete)
    else:
        shutil.copy(path_complete, path_test)

    create_tsv_files(dataset_path, True)
    save_complete_mappings(dataset_path)
    kg = KnowledgeGraph(
        path=dataset_path
    )
    prepare_for_sem_at_k(dataset_path, kg)
    sem = False
    repeat = True
    while(repeat):
        choice = input("Do you want to use the test set with untyped entities and relations filtered out? (yes/no): ")
        if choice.lower() == "no":
            sem = False
            repeat=False
        elif choice.lower() == "yes":
            sem = True
            repeat=False
            source = Path(dataset_path) / "abox" / "splits" / "sem_at_k.nt"
            destination = Path(dataset_path) / "abox" / "splits" / "test.nt"
            shutil.copy(source, destination)
            source = Path(dataset_path) / "abox" / "splits" / "train_sem_at_k.nt"
            destination = Path(dataset_path) / "abox" / "splits" / "train.nt"
            shutil.copy(source, destination)
            source = Path(dataset_path) / "abox" / "splits" / "valid_sem_at_k.nt"
            destination = Path(dataset_path) / "abox" / "splits" / "valid.nt"
            shutil.copy(source, destination)

            rebuild_mappings(dataset_path)
            mappings_dir_path = Path(dataset_path) / "mappings"
            mappings = ["individual_to_id", "class_to_id", "object_property_to_id"]
            for mapping in mappings:
                source = mappings_dir_path / f"{mapping}_sem.json"
                destination = mappings_dir_path / f"{mapping}.json"
                if source.exists():
                    shutil.copy(source, destination)
                    print(f"Copied {source.name} → {destination.name}")
                else:
                    print(f"Warning: {source.name} not found, skipping...")
        else:
            print("Incorrect choice!")

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
        output_dir_scoring = Path(output_directory) / "temp_scoring"
        best_weights_dir_path, best_params = train_and_evaluate_BoxE.train_BoxE(dataset_path, dataset_name, experiments, output_directory, ontology_path, kg, rules)
        best_weights_dir_path = Path("BoxE") / f"weights_{dataset_name}" 
        train_and_evaluate_BoxE.evaluate_inc_best_model_BoxE(ontology_path, train_path, output_kg_path, reasoner_path, best_weights_dir_path, dataset_name, output_dir_scoring, 64, entity_to_id_path, relation_to_id_path, kg, metrics)


def save_complete_mappings(dataset_path):
    mappings_dir_path = Path(dataset_path) / "mappings"
    
    files = [
        "individual_to_id",
        "object_property_to_id"
    ]
    
    for file in files:
        complete_path = mappings_dir_path / f"{file}_complete.json"
        original_path = mappings_dir_path / f"{file}.json"
        
        if not complete_path.exists():
            if not original_path.exists():
                print(f"Warning: {original_path} not found, skipping...")
                continue
            with open(original_path, "r") as f:
                data = json.load(f)
            with open(complete_path, "w") as f:
                json.dump(data, f, indent=4)
        else:
            shutil.copy(complete_path, original_path)


def rebuild_mappings(dataset_path):
    mappings_dir_path = Path(dataset_path) / "mappings"
    train_path = Path(dataset_path) / "abox" / "splits" / "train.nt"
    test_path = Path(dataset_path) / "abox" / "splits" / "test.nt"
    valid_path = Path(dataset_path) / "abox" / "splits" / "valid.nt"

    all_entities = set()
    all_relations = set()

    for split_path in [train_path, valid_path, test_path]:
        with open(split_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                head = parts[0].replace("<", "").replace(">", "")
                relation = parts[1].replace("<", "").replace(">", "")
                tail = parts[2].replace("<", "").replace(">", "")
                all_entities.add(head)
                all_entities.add(tail)
                all_relations.add(relation)

    ent2id = {ent: idx for idx, ent in enumerate(sorted(all_entities))}
    rel2id = {rel: idx for idx, rel in enumerate(sorted(all_relations))}

    print(f"Number of Entities: {len(ent2id)}, Number of Relations: {len(rel2id)}")

    with open(mappings_dir_path / "individual_to_id_sem.json", "w") as f:
        json.dump(ent2id, f, indent=4)
    with open(mappings_dir_path / "object_property_to_id_sem.json", "w") as f:
        json.dump(rel2id, f, indent=4)

    return ent2id, rel2id

  
if __name__ == "__main__":
    main()


