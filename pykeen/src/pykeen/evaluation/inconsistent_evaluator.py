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

class InconsistencyMetric(Enum):
    INC_AT_K = "inc_at_k"
    SEM_AT_K = "sem_at_k"




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

    def __init__(self, ontology_path:str, train_path:str, output_kg_path:str, reasoner_path:str, entity_to_id_path:str, relation_to_id_path:str, metric:InconsistencyMetric, class_to_id_path=None, k=5, **kwargs):
        super().__init__(**kwargs)
        if metric == InconsistencyMetric.SEM_AT_K and class_to_id_path is None:
            raise ValueError("The class_to_id path cannot be null with the metric " + str(InconsistencyMetric.SEM_AT_K) + ".")
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
        elif self.metric == InconsistencyMetric.SEM_AT_K:
            g = rdflib.Graph().parse(str(self.ontology_path), format="xml")
            for prediction in predictions:
                inconsistent_for_triple = 0
                for triple in prediction:
                    if self._is_consistent_relation(target, triple, g) == False:
                        inconsistent_for_triple += 1
                inc_percentage = inconsistent_for_triple / self.k
                self.inconsistencies[target].append(inc_percentage)

    def _is_consistent_relation(self, target, triple, g):
        head = self.id_to_entity[triple[0].item()]
        relation = self.id_to_relation[triple[1].item()]
        relation_uri = rdflib.URIRef(relation)
        tail = self.id_to_entity[triple[2].item()]
       
        if target == "head":
            head_uri = rdflib.URIRef(head)
            types_head = set(g.objects(subject=head_uri, predicate=RDF.type))
            all_types_head = set()
            for h in types_head:
                all_types_head.add(h)
                all_types_head |= self._get_all_superclasses(h, g)
            relation_ranges = set(g.objects(subject=relation_uri, predicate=RDFS.domain))
            return bool(all_types_head.intersection(relation_ranges))
        elif target == "tail":
            tail_uri = rdflib.URIRef(tail)
            types_tail = set(g.objects(subject=tail_uri, predicate=RDF.type))
            all_types_tail = set()
            for t in types_tail:
                all_types_tail.add(t)
                all_types_tail |= self._get_all_superclasses(t, g)
            relation_ranges = set(g.objects(subject=relation_uri, predicate=RDFS.range))
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
                # media delle percentuali per query
                inc_at_k = np.mean(self.inconsistencies[side])
            else:
                inc_at_k = 0.0
            data[InconsistentMetricKey(side=side, metric="inc_at_k")] = inc_at_k
        self.clear()
        return InconsistentMetricResults(data)
    

    def _uri_to_prefix_tag(self, uri: str) -> str:
        """Converte URI in tag XML con namespace."""
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

    def _get_all_superclasses(self, class_uri, g):
        """Risale la gerarchia delle classi."""
        superclasses = set()
        for superclass in g.objects(subject=class_uri, predicate=RDFS.subClassOf):
            superclasses.add(superclass)
            superclasses |= self._get_all_superclasses(superclass, g)  # ricorsivo
        return superclasses