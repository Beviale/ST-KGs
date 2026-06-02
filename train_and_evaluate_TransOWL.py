import os
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import jdex.utils.conventions.paths as pc
import json
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.models import TransOWL
from pykeen.evaluation import InconsistentEvaluator, InconsistencyMetric
import torch

import convert_to_axioms_TransOWL


def train_TransOWL(dataset_path, entity_mapping, relation_mapping, experiments,
                   output_directory, ontology_path, kg):
    output_dir_path = Path(output_directory) / "TransOWL"
    os.makedirs(output_dir_path, exist_ok=True)

    # Extract the axioms (inverseOf / equivalentProperty / subPropertyOf) from the ontology
    inverse_pairs, equivalent_pairs, subproperty_pairs = convert_to_axioms_TransOWL.extract_axiom_pairs(
        ontology_path, kg
    )

    train_tf = TriplesFactory.from_path(
        dataset_path / "abox" / "splits" / "train.tsv",
        entity_to_id=entity_mapping, relation_to_id=relation_mapping,
    )
    valid_tf = TriplesFactory.from_path(
        dataset_path / "abox" / "splits" / "valid.tsv",
        entity_to_id=entity_mapping, relation_to_id=relation_mapping,
    )
    test_tf = TriplesFactory.from_path(
        dataset_path / "abox" / "splits" / "test.tsv",
        entity_to_id=entity_mapping, relation_to_id=relation_mapping,
    )

    best_metric = -1.0
    best_pipeline_result = None
    best_params = None
    print(f"Number of hyperparameter combinations to test: {len(experiments)}\n")
    for i, params in enumerate(experiments):
        print(f"--- Experiment {i+1}/{len(experiments)} ---")
        print(f"Current hyperparameter values: {params}")
        try:
            log_file_path = Path(output_dir_path) / (
                f"dim{params['embedding_dim']}_lr{params['lr']}_margin{params['margin']}"
                f"_numNegs{params['num_negs']}_reg{params['reg_weight']}.txt"
            )
            result = pipeline(
                training=train_tf,
                validation=valid_tf,
                testing=test_tf,
                # TransOWL: TransE scoring + axiom-based regularization on relations
                model=TransOWL,
                model_kwargs=dict(
                    embedding_dim=params["embedding_dim"],
                    scoring_fct_norm=1,
                    inverse_relations=inverse_pairs,
                    equivalent_relations=equivalent_pairs,
                    subproperty_relations=subproperty_pairs,
                    regularizer_weight=params["reg_weight"],
                    subproperty_weight=params["subprop_weight"],
                    beta=params["beta"],
                ),
                optimizer='Adam',
                optimizer_kwargs=dict(lr=params["lr"]),
                loss='marginranking',
                loss_kwargs=dict(margin=params["margin"]),
                training_loop='slcwa',
                device="cuda:0",
                training_kwargs=dict(num_epochs=1000, batch_size=128),
                negative_sampler='basic',
                negative_sampler_kwargs=dict(num_negs_per_pos=params["num_negs"]),
                evaluator_kwargs=dict(filtered=True),
                evaluation_kwargs=dict(batch_size=128),
                stopper='early',
                stopper_kwargs=dict(frequency=5, patience=3,
                                    relative_delta=0.002,
                                    metric='mean_reciprocal_rank'),
                result_tracker='csv',
                random_seed=42,
                result_tracker_kwargs=dict(path=str(log_file_path)),
            )
            current_metric = result.stopper.best_metric
            print(f"MRR obtained: {current_metric:.4f}")
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
    print(f"Best MRR: {best_metric:.4f}")
    print(f"Best hyperparameter combination: {best_params}")
    print("=" * 40)

    if best_pipeline_result is not None:
        best_dir = os.path.join(output_dir_path, "Best")
        os.makedirs(best_dir, exist_ok=True)
        best_pipeline_result.save_to_directory(best_dir)
        print(f"\nBest model successfully saved in directory: '{output_directory}'")
    return best_pipeline_result


def evaluate_inc_best_model_TransOWL(ontology_path, train_path, output_kg_path,
                                     reasoner_path, best_model_path, dataset_path,
                                     entity_to_id_path, relation_to_id_path,
                                     output_directory, kg, metrics):
    # Identical to evaluate_inc_best_model_TransE: scoring/eval is the same as TransE
    print("---- Evaluating using inconsistency metrics ----")
    out_dir = Path(output_directory) / "TransOWL"
    os.makedirs(out_dir, exist_ok=True)
    metrics_file_path = os.path.join(out_dir, "Inconsistent_Metrics.txt")
    best_model = torch.load(Path(best_model_path) / "trained_model.pkl", weights_only=False)
    with open(entity_to_id_path, "r") as f:
        entity_mapping = json.load(f)
    with open(relation_to_id_path, "r") as f:
        relation_mapping = json.load(f)
    train_tf = TriplesFactory.from_path(dataset_path / pc.TRAIN,
                                        entity_to_id=entity_mapping, relation_to_id=relation_mapping)
    valid_tf = TriplesFactory.from_path(dataset_path / pc.VALID,
                                        entity_to_id=entity_mapping, relation_to_id=relation_mapping)
    test_tf = TriplesFactory.from_path(dataset_path / pc.TEST,
                                       entity_to_id=entity_mapping, relation_to_id=relation_mapping)
    for metric in metrics:
        print(f"------Metric={metric.name}------------------")
        for k in [1, 3, 10]:
            print(f"-K={k}-")
            inc_evaluator = InconsistentEvaluator(ontology_path, train_path, output_kg_path,
                                                  reasoner_path, entity_to_id_path,
                                                  relation_to_id_path, metric, kg, k, filtered=True)
            inc_results = inc_evaluator.evaluate(
                model=best_model,
                mapped_triples=test_tf.mapped_triples,
                additional_filter_triples=[train_tf.mapped_triples, valid_tf.mapped_triples],
                batch_size=16, slice_size=256,
            )
            print(f"{metric.name}_{k}: {str(inc_results.data)}")
            with open(metrics_file_path, "a", encoding="utf-8") as f:
                f.write(f"{metric.name}_{k}: {str(inc_results.data)}\n")
