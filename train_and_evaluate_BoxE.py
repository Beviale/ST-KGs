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


def train_BoxE(dataset_path: str, dataset_name: str, experiments, output_dir):
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
        capture_output=True, 
    )
    if result.returncode == 0:
        print("=== PREPROCESSING COMPLETED SUCCESSFULLY ===")
    else:
        print(
        f"=== ERROR DURING PREPROCESSING (Exit Code: {result.returncode}) ==="
    )
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

    best_mrr = -1.0  
    best_result_weights_path = None
    best_result_log_path = None
    best_params = None
    print(f"\n-----Number of hypeparameter combinations to test: {len(experiments)}-------\n")
    for i, params in enumerate(experiments):
        print(f"\n--- Experiment {i+1}/{len(experiments)} ---")
        print(f"Current hyparameter values: {params}")
        param_str_log = f"neg{params['nbNegExp']}_margin{params['loss_margin']}_reg{params['reg_lambda']}_lr{params['learningRate']}"
        boxe_path =  Path(output_dir) / "BoxE"
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
        command = (
            f"conda activate boxe && "
            f"cd BoxE && "
            f"python Training.py {dataset_name} "
            f"-validation True "
            f"-validCkpt 5 "
            f"-logFName \"{os.path.normpath(log_file_path_absolute)}\" "  
            f"-epochs 2 "
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
                f"=== ERROR DURING PREPROCESSING (Exit Code: {result.returncode}) ==="
            )
            continue
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"Start training: {current_time}\n")

        #------------------TESTING the current hyperparameter configuration using the eval set
        command = (
            f"conda activate boxe && "
            f"cd BoxE && "
            f"python Testing.py {dataset_name} rank "
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
            if item == specific_weights_path:
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
        f"conda activate boxe && "
        f"cd BoxE && "
        f"python Testing.py {dataset_name} "
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



def evaluate_inc_best_model_BoxE(ontology_path: str, train_path:str, output_kg_path: str, reasoner_path: str,  best_result_weights_path: str, dataset_name: str, output_directory: str, embedding_dim: int, entity_to_id_path, relation_to_id_path, kg, metric_selection: InconsistencyMetric):
    print("---- Evaluating using inconsistency metrics ----")
    os.makedirs(output_directory, exist_ok=True)
    result = subprocess.run(
        [
            "conda", "run", "-n", "boxe",
            "python", "run_boxe_scoring.py",
            "--dataset", dataset_name,
            "--weights", os.path.normpath(os.path.abspath(os.path.join(best_result_weights_path, "values.ckpt"))),
            "--embedding_dim", str(embedding_dim),
            "--output", os.path.normpath(os.path.abspath(output_directory)),
        ],
        cwd="BoxE",
        capture_output=True,
        text=True
    )
    if "DONE" not in result.stdout:
        print(result.stderr)
        raise RuntimeError("BoxE scoring failed!")

    scores_np = np.load(os.path.join(output_directory, "scores.npy"))
    hrt_np = np.load(os.path.join(output_directory, "hrt.npy"))
    scores_tensor = torch.tensor(scores_np, dtype=torch.float32)
    hrt_tensor = torch.tensor(hrt_np, dtype=torch.long)
    for k in [1, 3, 10]:
        inc_evaluator = InconsistentEvaluator(ontology_path, train_path, output_kg_path, reasoner_path, entity_to_id_path, relation_to_id_path, metric_selection, kg, k, filtered=True)
        inc_evaluator.process_scores_(
            hrt_batch=hrt_tensor,
            target="tail",
            scores=scores_tensor,
        )
        prova = inc_evaluator.finalize()
        print(str(prova))


