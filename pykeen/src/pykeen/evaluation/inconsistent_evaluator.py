from pykeen.evaluation.evaluator import Evaluator, MetricResults
from pykeen.typing import MappedTriples, Target
from pykeen.utils import FloatTensor
from typing import NamedTuple, ClassVar, Mapping
import torch
import numpy as np
import rdflib
import shutil
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from jdex.owl.reasoning import Reasoner
from pathlib import Path
import json
from rdflib import URIRef
from xml.etree.ElementTree import iterparse
from enum import Enum
from rdflib import OWL, RDFS, RDF
from jdex.loaders.torch import KnowledgeGraph

class InconsistencyMetric(Enum):
    INC_AT_K = "inc_at_k"
    SEM_AT_K = "sem_at_k"
    AC_AT_K_1 = "ac_at_k_using_reasoner"
    AC_AT_K_2 = "ac_at_k_using_domain_range_rel"



class InconsistentMetricKey(NamedTuple):
    side: str    # "head" o "tail" o "both"
    metric: str  # "inc_at_k"

class InconsistentMetricResults(MetricResults[InconsistentMetricKey]):
    metrics: ClassVar[Mapping] = {}

    @classmethod
    def key_from_string(cls, s):
        if s is None:
            return InconsistentMetricKey(side="both", metric="inc_at_k")
        parts = s.split(".")
        return InconsistentMetricKey(side=parts[0], metric=parts[1])

class InconsistentEvaluator(Evaluator[InconsistentMetricKey]):
    
    metric_result_cls = InconsistentMetricResults

    def __init__(self, ontology_path:str, train_path:str, output_kg_path:str, reasoner_path:str, entity_to_id_path:str, relation_to_id_path:str, metric:InconsistencyMetric, kg: KnowledgeGraph, k, num_workers:int=4, **kwargs):
        super().__init__(**kwargs)
        self.kg = kg
        # Number of parallel Konclude workers for reasoner-based metrics.
        # RAM is the bottleneck: each worker loads the full ontology in memory.
        # Tune as: (free_RAM_GB - 4) / RAM_per_Konclude_GB, rounded down.
        self.num_workers = num_workers
        self.ontology_path = ontology_path
        self.train_path = train_path
        self.output_kg_path = output_kg_path
        self.reasoner_path = reasoner_path
        self.metric = metric
        with open(entity_to_id_path, "r") as f:
            self.entity_to_id = json.load(f)  
        with open(relation_to_id_path, "r") as f:
            self.relation_to_id = json.load(f)  
        self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
        self.id_to_relation = {v: k for k, v in self.relation_to_id.items()}
        self.k = k
        self.inconsistencies = {"head": [], "tail": []}
        # Cache reasoner consistency results per unique triple (uri_h, uri_r, uri_t).
        # The base KG is fixed, so the outcome only depends on the added triple.
        self.consistency_cache = {}
        self.reasoner = Reasoner(
                            self.reasoner_path,
                            java8_path=Path(r"C:\Program Files\Java\jdk1.8.0_202"),
                            java11_path=Path(r"C:\Program Files\Java\jdk-11.0.26"),
                        )
        self._initialize_graph()
        self.ns_map = {}
        for event, elem in iterparse(str(self.output_kg_path), events=["start-ns"]):
            prefix, uri = elem
            self.ns_map[uri] = prefix
        if not self.reasoner.consistency(self.output_kg_path, verbose=0):
            raise ValueError("The given dataset is not consistent!")

    def _initialize_graph(self):
        if Path(self.output_kg_path).exists() == False:
            g_schema   = rdflib.Graph().parse(str(self.ontology_path), format="xml")
            g_abox     = rdflib.Graph().parse(str(self.train_path), format="nt")
            g_merged = rdflib.Graph()
            g_merged += g_schema
            g_merged += g_abox
            g_merged.serialize(destination=str(self.output_kg_path), format="xml") 


    def process_scores_(
        self,
        hrt_batch: MappedTriples,
        target: Target,
        scores: FloatTensor,
        true_scores=None,
        dense_positive_mask=None,
    ) -> None:
        top_k = torch.topk(scores, k=self.k, dim=1).indices

        if target == "tail":
            h = hrt_batch[:, 0].unsqueeze(1).expand(-1, self.k)  # (batch_size, k)
            r = hrt_batch[:, 1].unsqueeze(1).expand(-1, self.k)  # (batch_size, k)
            predictions = torch.stack([h, r, top_k], dim=2)         # (batch_size, k, 3)
        else:
            r = hrt_batch[:, 1].unsqueeze(1).expand(-1, self.k)  # (batch_size, k)
            t = hrt_batch[:, 2].unsqueeze(1).expand(-1, self.k)  # (batch_size, k)
            predictions = torch.stack([top_k, r, t], dim=2)     #(batch_size, k, 3) 

        # Pre-fill the consistency cache in parallel for reasoner-based metrics,
        # so the sequential metric loops below only hit the cache.
        if self.metric in (InconsistencyMetric.INC_AT_K, InconsistencyMetric.AC_AT_K_1):
            uris = [
                (self.id_to_entity[t[0].item()],
                 self.id_to_relation[t[1].item()],
                 self.id_to_entity[t[2].item()])
                for prediction in predictions for t in prediction
            ]
            self._prefetch_consistency(uris)

        if self.metric == InconsistencyMetric.INC_AT_K:
            for prediction in predictions:
                inconsistent_for_triple = 0
                for triple in prediction:
                    if self._is_consistent(triple) == False:
                        inconsistent_for_triple += 1
                inc_percentage = inconsistent_for_triple / self.k
                self.inconsistencies[target].append(inc_percentage)
        elif self.metric == InconsistencyMetric.AC_AT_K_1:
            for prediction in predictions:
                consistency_sum = 0.0
                consistent_count = 0
                for i, triple in enumerate(prediction):
                    if self._is_consistent(triple):
                        consistent_count += 1
                        consistency_sum += consistent_count / (i + 1)
                ac_at_k = consistency_sum / self.k
                self.inconsistencies[target].append(ac_at_k)
        elif self.metric == InconsistencyMetric.SEM_AT_K:
            for prediction in predictions:
                inconsistent_for_triple = 0
                for triple in prediction:
                    if self._is_consistent_relation(target, triple) == False:
                        inconsistent_for_triple += 1
                inc_percentage = inconsistent_for_triple / self.k
                self.inconsistencies[target].append(inc_percentage)
        elif self.metric == InconsistencyMetric.AC_AT_K_2:
            for prediction in predictions:
                consistency_sum = 0.0
                consistent_count = 0
                for i, triple in enumerate(prediction):
                    if self._is_consistent_relation(target, triple):
                        consistent_count += 1
                        consistency_sum += consistent_count / (i + 1)
                ac_at_k = consistency_sum / self.k
                self.inconsistencies[target].append(ac_at_k)



    def _is_consistent_relation(self, target, triple):
        head = triple[0]
        relation = triple[1].item()
        tail = triple[2].item()
        if target == "head":
            all_types_head = set(self.kg.individual_classes(head).tolist())
            relation_domains = set(self.kg.obj_prop_domain(relation).tolist())
            if not all_types_head:
                raise Exception(f"The head entity {head} does not have any class!")
            if not relation_domains:
                raise Exception(f"The relation {relation} does not have any domain!")
            return bool(all_types_head.intersection(relation_domains))
        elif target == "tail":
            all_types_tail = set(self.kg.individual_classes(tail).tolist())
            relation_ranges = set(self.kg.obj_prop_range(relation).tolist())
            if not all_types_tail:
                raise Exception(f"The tail entity {tail} does not have any class!")
            if not relation_ranges:
                raise Exception(f"The relation {relation} does not have any range!")
            return bool(all_types_tail.intersection(relation_ranges))



    def _consistency_on_file(self, kg_file, head, relation, tail) -> bool:
        """Check consistency of a single triple appended to a private KG copy.

        Same logic as :meth:`_is_consistent`, but operates on ``kg_file`` (a per-worker
        copy of the base KG) so it can run concurrently. ``Reasoner.consistency`` is
        stateless w.r.t. paths, so it is safe to call from multiple threads.
        """
        relation_tag = self._uri_to_prefix_tag(relation)
        triple_xml = f'    <rdf:Description rdf:about="{head}">\n        <{relation_tag} rdf:resource="{tail}"/>\n    </rdf:Description>\n'
        with open(kg_file, "r+b") as f:
            content = f.read()
            closing_tag = b"</rdf:RDF>"
            pos = content.rfind(closing_tag)
            if pos == -1:
                raise ValueError("Tag </rdf:RDF> non trovato")
            f.seek(pos)
            f.write(triple_xml.encode("utf-8") + closing_tag)
        result = self.reasoner.consistency(kg_file, verbose=0)
        with open(kg_file, "r+b") as f:  # restore the base KG copy
            content = f.read()
            triple_bytes = triple_xml.encode("utf-8")
            if triple_bytes in content:
                h, sep, t = content.rpartition(triple_bytes)
                content = h + t
            f.seek(0)
            f.write(content)
            f.truncate()
        return result

    def _prefetch_consistency(self, triples_uris, num_workers=None) -> None:
        """Populate ``self.consistency_cache`` in parallel using ``num_workers`` Konclude
        instances, each working on its own copy of the base KG file."""
        if num_workers is None:
            num_workers = self.num_workers
        todo = [t for t in set(triples_uris) if t not in self.consistency_cache]
        if not todo:
            return
        num_workers = max(1, min(num_workers, len(todo)))

        worker_files = [Path(f"{self.output_kg_path}.w{i}") for i in range(num_workers)]
        for wf in worker_files:
            shutil.copy(self.output_kg_path, wf)
        file_queue = queue.Queue()
        for wf in worker_files:
            file_queue.put(wf)
        lock = threading.Lock()

        def task(tr):
            wf = file_queue.get()
            try:
                res = self._consistency_on_file(wf, *tr)
            finally:
                file_queue.put(wf)
            with lock:
                self.consistency_cache[tr] = res

        try:
            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                list(ex.map(task, todo))
        finally:
            for wf in worker_files:
                wf.unlink(missing_ok=True)

    def _is_consistent(self, triple: tuple) -> bool:
        """Verify if the given triple is consistent"""
        head = self.id_to_entity[triple[0].item()]
        relation = self.id_to_relation[triple[1].item()]
        relation_tag = self._uri_to_prefix_tag(relation)
        tail = self.id_to_entity[triple[2].item()]     

        cache_key = (head, relation, tail)
        if cache_key in self.consistency_cache:
            return self.consistency_cache[cache_key]
        
        triple_xml = f'    <rdf:Description rdf:about="{head}">\n        <{relation_tag} rdf:resource="{tail}"/>\n    </rdf:Description>\n' 
        with open(self.output_kg_path, "r+b") as f:
            content = f.read()
            closing_tag = b"</rdf:RDF>"
            pos = content.rfind(closing_tag)
            if pos == -1:
                raise ValueError("Tag </rdf:RDF> non trovato")
            f.seek(pos)
            f.write(triple_xml.encode("utf-8") + closing_tag)

        result = self.reasoner.consistency(self.output_kg_path, verbose=0)
        with open(self.output_kg_path, "r+b") as f:
            content = f.read()
            triple_bytes = triple_xml.encode("utf-8")
        
            if triple_bytes in content:
                head, sep, tail = content.rpartition(triple_bytes)
                content = head + tail       
            f.seek(0)
            f.write(content)
            f.truncate()  
        self.consistency_cache[cache_key] = result
        return result

    def clear(self) -> None:
        self.inconsistencies = {"head": [], "tail": []}
        self.consistency_cache = {}

    def finalize(self) -> InconsistentMetricResults:
        data = {}
        for side in ["head", "tail"]:
            if self.inconsistencies[side]:
                inc_at_k = np.mean(self.inconsistencies[side])
            else:
                inc_at_k = 0.0
            data[InconsistentMetricKey(side=side, metric="inc_at_k")] = inc_at_k
        self.clear()
        return InconsistentMetricResults(data)
    

    def _uri_to_prefix_tag(self, uri: str) -> str:
        """Convert an URI in a XML tag with namespace."""
        if "#" in uri:
            ns, local = uri.rsplit("#", 1)
            ns += "#"
        else:
            ns, local = uri.rsplit("/", 1)
            ns += "/"
        prefix = self.ns_map.get(ns, None)
        if prefix is None:
            return None
        return f"{prefix}:{local}"

