import os
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import numpy as np
import jdex.utils.conventions.paths as pc
import json
import os
from tqdm import tqdm
from pykeen.evaluation import InconsistencyMetric
from pykeen.evaluation import InconsistentEvaluator
import shutil
import subprocess
import re
from datetime import datetime  
import sys
import os
import numpy as np
import msgpack
import msgpack_numpy as m
import torch
import msgpack
import convert_to_rules_BoxE

CONDA = os.environ.get("CONDA_EXE", "conda")
BOXE_PY = os.path.join(os.path.dirname(os.path.dirname(CONDA)), "envs", "boxe", "bin", "python")


def train_BoxE(dataset_path: str, dataset_name: str, experiments, output_dir, ontology_path, kg, rules=False):
    dataset_dir = Path("BoxE") / "Datasets" / dataset_name
    dataset_dir_multi = Path("BoxE") / "DatasetsMulti" / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir_multi.mkdir(parents=True, exist_ok=True)
    if rules:
        print("=== CREATING THE RULES ===")
        convert_to_rules_BoxE.convert_owl_to_boxe(ontology_path, kg, dataset_dir_multi / "rules.txt" )
        print("=== RULES Created SUCCESSFULLY ===")
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


    path_Ente2ID_dict = dataset_dir_multi / "Ent2ID.dict"
    path_Rel2ID_dict = dataset_dir_multi / "Rel2ID.dict"
    path_new_Ente2Id_dict = dataset_path / pc.INDIVIDUAL_MAPPINGS
    path_new_Rel2Id_dict = dataset_path / pc.OBJ_PROP_MAPPINGS
    with open(path_new_Ente2Id_dict, 'r', encoding='utf-8') as f:
        new_ente2Id_dict = json.load(f)
    with open(path_new_Rel2Id_dict, 'r', encoding='utf-8') as f:
        new_Rel2Id_dict = json.load(f)

    with open(path_Ente2ID_dict, 'wb') as f:
        msgpack.pack(new_ente2Id_dict, f)
    with open(path_Rel2ID_dict, 'wb') as f:
        msgpack.pack(new_Rel2Id_dict, f)

    #Pre-processing
    command = f"cd BoxE && {CONDA} run -n boxe {BOXE_PY} KBUtils.py"
    result = subprocess.run(
        command,
        shell=True, 
        text=True,
        capture_output=True, 
    )
    if result.returncode == 0:
        print("=== PREPROCESSING COMPLETED SUCCESSFULLY ===")
    else:
        print(
        f"=== ERROR DURING PREPROCESSING (Exit Code: {result.stderr}) ==="
    )

    best_mrr = -1.0  
    best_result_weights_path = None
    best_result_log_path = None
    best_params = None
    print(f"\n-----Number of hypeparameter combinations to test: {len(experiments)}-------\n")
    for i, params in enumerate(experiments):
        print(f"\n--- Experiment {i+1}/{len(experiments)} ---")
        print(f"Current hyparameter values: {params}")
        param_str_log = f"neg{params['nbNegExp']}_margin{params['loss_margin']}_reg{params['reg_lambda']}_lr{params['learningRate']}"
        boxe_path =  Path(output_dir) / ("BoxE_Rules" if rules else "BoxE")
        os.makedirs(boxe_path, exist_ok=True)
        dir_log_path = boxe_path / param_str_log
        os.makedirs(dir_log_path, exist_ok=True)
        log_file_path = os.path.join(dir_log_path, "training_log.txt")
        log_file_path_absolute = os.path.abspath(log_file_path)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file_path, "w", encoding="utf-8") as f:
            pass
        #-------------TRAINING
        #command = f"conda activate boxe && cd Boxe && python training.py {dataset_name} -validation True -validCkpt 5 -logFName BoxE_NoRule_{dataset_name}_{str(params)} -epochs 10 -nbNegExp {params['nbNegExp']} -lossMargn {params['loss_margin']} -regLambda {params['reg_lambda']} -learningRate {params['learningRate']}"

        if rules:
            command = (
                f"cd BoxE && "
                f"{CONDA} run -n boxe {BOXE_PY} Training.py {dataset_name} "
                f"-validation True "
                f"-validCkpt 10 "
                f"-logFName \"{os.path.normpath(log_file_path_absolute)}\" "  
                f"-epochs 1000 "
                f"-patience 3 "
                f"-batchSize 8192 "
                f"-nbNegExp {params['nbNegExp']} "
                f"-lossMargin {params['loss_margin']} "
                f"-regLambda {params['reg_lambda']} "
                f"-learningRate {params['learningRate']} "
                f"-ruleDir {os.path.normpath(os.path.abspath(dataset_dir_multi / "rules.txt"))}"
            )
        else:
            command = (
                f"cd BoxE && "
                f"{CONDA} run -n boxe {BOXE_PY} Training.py {dataset_name} "
                f"-validation True "
                f"-validCkpt 10 "
                f"-logFName \"{os.path.normpath(log_file_path_absolute)}\" "  
                f"-epochs 400 "
                f"-patience 3 "
                f"-batchSize 8192 "
                f"-nbNegExp {params['nbNegExp']} "
                f"-lossMargin {params['loss_margin']} "
                f"-regLambda {params['reg_lambda']} "
                f"-learningRate {params['learningRate']}"
            )

        result = subprocess.run(
            command,
            shell=True, 
            text=True,
            capture_output=True, 
        )
        if result.returncode == 0:
            print("=== TRAINING COMPLETED SUCCESSFULLY ===")
        else:
            print(
                f"=== ERROR DURING TRAINING (Exit Code: {result.stderr}) ==="
            )
            continue
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"Start training: {current_time}\n")

        #------------------TESTING the current hyperparameter configuration using the eval set
        command = (
            f"cd BoxE && "
            f"{CONDA} run -n boxe {BOXE_PY} Testing.py {dataset_name} rank "
            f"-testFile Valid "
            f"-testSetting filtered"
        )
        metrics_extracted = {}
        print("Executing testing command using validation set and capturing metrics...")
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8"
        )

        while True:
            output_line = process.stdout.readline()
            if output_line == '' and process.poll() is not None:
                break  
                
            if output_line:
                #print(output_line.strip())
                
                match = re.search(r'(MR|MRR|Hits@\d+):([\d\.]+)', output_line)
                if match:
                    metric_name = match.group(1)         
                    metric_value = float(match.group(2))  
                    metrics_extracted[metric_name] = metric_value

        process.wait()
        weights_path = Path("BoxE") / f"weights_{dataset_name}"
        specific_weights_path = os.path.join(weights_path, param_str_log)
        os.makedirs(specific_weights_path, exist_ok=True)
        for item in weights_path.iterdir():
            if str(item) == str(specific_weights_path):
                continue    
            try:
                if os.path.exists(os.path.join(specific_weights_path, os.path.basename(item))):
                    os.remove(os.path.join(specific_weights_path, os.path.basename(item)))
                shutil.move(str(item), str(specific_weights_path))
            except Exception as e:
                print(f"Error moving {item.name}: {e}")
        timestamp_file_path = Path(specific_weights_path) / "weights_log.txt"
        with open(timestamp_file_path, "w", encoding="utf-8") as f:
            f.write(f"Weights saved on: {current_time}\n")

        print("Extraction complete.")
        print("Captured metrics:", metrics_extracted)

        if metrics_extracted["MRR"] > best_mrr:
            print("New best hyparameter combination found!")
            best_mrr = metrics_extracted["MRR"]
            best_result_weights_path = specific_weights_path
            best_params = params
            best_result_log_path = dir_log_path
        valid_result_log_path = dir_log_path / "validation_results.txt"
        with open(valid_result_log_path, "w", encoding="utf-8") as f:
            for metric_name, metric_value in metrics_extracted.items():
                f.write(f"{metric_name}: {metric_value}\n")

    print("\n------------------------------------")
    print("Best hyperparameter configuration: " + str(best_params))
    print("Best validation MRR result: " + str(best_mrr))
    folder_name = os.path.basename(best_result_weights_path)
    parent_dir = os.path.dirname(best_result_weights_path)
    new_weights_path = os.path.join(parent_dir, f"{folder_name}_BEST")
    if os.path.exists(new_weights_path):
        shutil.rmtree(new_weights_path)
    os.rename(best_result_weights_path, new_weights_path)
    best_result_weights_path = new_weights_path
    print("Best weights saved on: " + str(best_result_weights_path))

    folder_name = os.path.basename(best_result_log_path)
    parent_dir = os.path.dirname(best_result_log_path)
    new_log_path = os.path.join(parent_dir, f"{folder_name}_BEST")
    if os.path.exists(new_log_path):
        shutil.rmtree(new_log_path)
    os.rename(best_result_log_path, new_log_path)
    best_result_log_path = new_log_path
    print("Best log saved on: " + str(best_result_weights_path))


    # ----------------Evaluate the best model on the test set
    for content_item in Path(best_result_weights_path).iterdir():
        destination = weights_path / content_item.name
        
        try:
            if not content_item.is_dir():
                shutil.copy2(str(content_item), str(destination))
                            
        except Exception as e:
            print(f"Error copying {content_item.name}: {e}")
            
    
    test_absolute_path = os.path.abspath(Path("BoxE") / "DatasetsMulti" / "testS.txt")
    command = (
        f"cd BoxE && "
        f"{CONDA} run -n boxe {BOXE_PY} Testing.py {dataset_name} "
        f"-verbosity False "
        f"rank "
        f"-testSetting filtered "
        f"-testFile Test"
    )
    # Dictionary to store parsed metrics
    metrics_extracted = {}
    print("Executing testing command using test set and capturing metrics...")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8"
    )

    while True:
        output_line = process.stdout.readline()
        if output_line == '' and process.poll() is not None:
            break  
            
        if output_line:
            #print(output_line.strip())
            
            match = re.search(r'(MR|MRR|Hits@\d+):([\d\.]+)', output_line)
            if match:
                metric_name = match.group(1)         
                metric_value = float(match.group(2))  
                metrics_extracted[metric_name] = metric_value

    process.wait()
    print("\nExtraction complete.")
    print("Captured metrics:", metrics_extracted)
    test_result_log_path = Path(best_result_log_path) / "test_results.txt"
    with open(test_result_log_path, "w", encoding="utf-8") as f:
        for metric_name, metric_value in metrics_extracted.items():
            f.write(f"{metric_name}: {metric_value}\n")
    return best_result_weights_path, best_params


def _filter_known_scores(scores: torch.Tensor, hrt: torch.Tensor, gold_col: int, known: dict) -> None:
    """Replica il setting 'filtered' di pykeen su score calcolati esternamente.

    Mette a -inf lo score di ogni entita' che forma una tripla gia' nota
    (train+valid+test), tranne la gold valutata su ciascuna riga. Cosi' il top-K
    di BoxE compete solo contro entita' non-vere, esattamente come pykeen fa per
    TransE via additional_filter_triples. Modifica `scores` in-place.

    Tail prediction: gold_col=2, chiave=(head, rel). Head: gold_col=0, chiave=(rel, tail).
    """
    num_cols = scores.shape[1]
    for i in range(hrt.shape[0]):
        h, r, t = int(hrt[i, 0]), int(hrt[i, 1]), int(hrt[i, 2])
        gold = int(hrt[i, gold_col])
        key = (h, r) if gold_col == 2 else (r, t)
        for e in known.get(key, ()):
            if e != gold and e < num_cols:
                scores[i, e] = float("-inf")


def _build_known_triple_maps(dataset_name, entity_to_id_path, relation_to_id_path):
    """Costruisce le mappe delle triple note (train+valid+test) nello spazio-ID jdex,
    leggendo gli stessi .txt usati per generare test.kb (h\\tr\\tt)."""
    from collections import defaultdict
    with open(entity_to_id_path, "r", encoding="utf-8") as f:
        ent_map = json.load(f)
    with open(relation_to_id_path, "r", encoding="utf-8") as f:
        rel_map = json.load(f)
    hr_to_tails = defaultdict(set)
    rt_to_heads = defaultdict(set)
    boxe_ds_dir = Path("BoxE") / "Datasets" / dataset_name
    for split in ("train.txt", "valid.txt", "test.txt"):
        split_path = boxe_ds_dir / split
        if not split_path.exists():
            print(f"[FILTER] ATTENZIONE: {split_path} non trovato, filtraggio parziale.")
            continue
        with open(split_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                h, rel, t = parts[0], parts[1], parts[2]
                if h in ent_map and rel in rel_map and t in ent_map:
                    hi, ri, ti = ent_map[h], rel_map[rel], ent_map[t]
                    hr_to_tails[(hi, ri)].add(ti)
                    rt_to_heads[(ri, ti)].add(hi)
    return hr_to_tails, rt_to_heads


def evaluate_inc_best_model_BoxE(ontology_path: str, train_path:str, output_kg_path: str, reasoner_path: str,  best_result_weights_path: str, dataset_name: str, output_directory: str, temp_dir:str, embedding_dim: int, entity_to_id_path, relation_to_id_path, kg, metrics: list[InconsistencyMetric], rules: bool = False):
    print("---- Evaluating using inconsistency metrics ----")
    boxE_outputDir = Path(output_directory) / ("BoxE_Rules" if rules else "BoxE")
    os.makedirs(boxE_outputDir, exist_ok=True)
    metrics_file_path = os.path.join(boxE_outputDir, "Inconsistent_Metrics.txt")
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    result = subprocess.run(
        [
            CONDA, "run", "-n", "boxe",
            BOXE_PY, "run_boxe_scoring.py",
            "--dataset", dataset_name,
            "--weights", os.path.normpath(os.path.abspath(os.path.join(best_result_weights_path, "values.ckpt"))),
            "--embedding_dim", str(embedding_dim),
            "--output", os.path.normpath(os.path.abspath(temp_dir)),
            "--type", "tail",
        ],
        cwd="BoxE",
        capture_output= True,
        text=True
    )
    if "DONE" not in result.stdout:
        print(result.stderr)
        raise RuntimeError("BoxE scoring failed!")

    scores_np_tail = np.load(os.path.join(temp_dir, "scores_tail.npy"))
    hrt_np_tail = np.load(os.path.join(temp_dir, "hrt_tail.npy"))
    scores_tensor_tail = torch.tensor(scores_np_tail, dtype=torch.float32)
    hrt_tensor_tail = torch.tensor(hrt_np_tail, dtype=torch.long)


    result = subprocess.run(
        [
            CONDA, "run", "-n", "boxe",
            BOXE_PY, "run_boxe_scoring.py",
            "--dataset", dataset_name,
            "--weights", os.path.normpath(os.path.abspath(os.path.join(best_result_weights_path, "values.ckpt"))),
            "--embedding_dim", str(embedding_dim),
            "--output", os.path.normpath(os.path.abspath(temp_dir)),
            "--type", "head",
        ],
        cwd="BoxE",
        capture_output= True,
        text=True
    )
    if "DONE" not in result.stdout:
        print(result.stderr)
        raise RuntimeError("BoxE scoring failed!")

    scores_np_head = np.load(os.path.join(temp_dir, "scores_head.npy"))
    hrt_np_head = np.load(os.path.join(temp_dir, "hrt_head.npy"))
    scores_tensor_head = torch.tensor(scores_np_head, dtype=torch.float32)
    hrt_tensor_head = torch.tensor(hrt_np_head, dtype=torch.long)

    # --- Filtering coerente con la valutazione di TransE (setting "filtered") ---
    # pykeen, per TransE, azzera (-inf) le entita' delle triple gia' note prima del top-K
    # (additional_filter_triples=[train, valid] + il filtro interno su test). Qui replichiamo
    # lo stesso comportamento sugli score grezzi di BoxE, altrimenti il confronto e' sbilanciato.
    hr_to_tails, rt_to_heads = _build_known_triple_maps(dataset_name, entity_to_id_path, relation_to_id_path)
    _filter_known_scores(scores_tensor_tail, hrt_tensor_tail, 2, hr_to_tails)
    _filter_known_scores(scores_tensor_head, hrt_tensor_head, 0, rt_to_heads)
    print(f"[FILTER] filtering applicato (filtered=True) su {hrt_tensor_tail.shape[0]} triple di test")

    for metric in metrics:
        print(f"------Metric={metric}------------------")
        for k in [1, 3, 10]:
            print(f"------K={k}------------------")
            inc_evaluator = InconsistentEvaluator(ontology_path, train_path, output_kg_path, reasoner_path, entity_to_id_path, relation_to_id_path, metric, kg, k, filtered=True)
            inc_evaluator.process_scores_(
                hrt_batch=hrt_tensor_tail,
                target="tail",
                scores=scores_tensor_tail,
            )
            inc_evaluator.process_scores_(
                hrt_batch=hrt_tensor_head,
                target="head",
                scores=scores_tensor_head,
            )
            inc_results = inc_evaluator.finalize()
            print(f"{metric.name}_{k}: {str(inc_results.data)}")
            with open(metrics_file_path, "a", encoding="utf-8") as f:
                f.write(f"{metric.name}_{k}: {str(inc_results.data)}\n")