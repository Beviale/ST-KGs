from pykeen.evaluation.evaluator import Evaluator, MetricResults
from pykeen.typing import MappedTriples, Target
from pykeen.utils import FloatTensor
from typing import NamedTuple, ClassVar, Mapping
import torch
import numpy as np
import rdflib
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
    SEM_AT_K_Base = "sem_at_k"
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

    def __init__(self, ontology_path:str, train_path:str, output_kg_path:str, reasoner_path:str, entity_to_id_path:str, relation_to_id_path:str, metric:InconsistencyMetric, kg: KnowledgeGraph, k, **kwargs):
        super().__init__(**kwargs)
        self.kg = kg
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
                self.consistencies[target].append(ac_at_k)
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
                self.consistencies[target].append(ac_at_k)



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



    def _is_consistent(self, triple: tuple) -> bool:
        """Verify if the given triple is consistent"""
        head = self.id_to_entity[triple[0].item()]
        relation = self.id_to_relation[triple[1].item()]
        relation_tag = self._uri_to_prefix_tag(relation)
        tail = self.id_to_entity[triple[2].item()]     
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
        return result

    def clear(self) -> None:
        self.inconsistencies = {"head": [], "tail": []}

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

