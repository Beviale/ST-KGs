from owlready2 import get_ontology, ObjectPropertyClass, ThingClass

# Top classes excluded from class-axiom extraction: a constraint like "C ⊑ Thing"
# is trivially true for every class and would only add noise to the regularization.
TOP_CLASSES = {
    "http://www.w3.org/2002/07/owl#Thing",
    "http://schema.org/Thing",
}


def extract_axiom_pairs(owl_filepath: str, kg, entity_to_id=None):
    """Extract relation and (optionally) class axiom pairs from the OWL ontology.

    Relation axioms are mapped through object_property_to_id / kg.obj_prop_to_id.
    Class axioms are mapped through ``entity_to_id`` (classes must be entities, i.e.
    present in the training triples as objects of rdf:type). If ``entity_to_id`` is
    None or no class is found in it, the two class lists come back empty -> the model
    automatically falls back to relation-only regularization.

    Returns five lists of ID pairs:
      - inverse_pairs:        (r, q) with r == q^-  (owl:inverseOf,         symmetric)
      - equivalent_pairs:     (r, p) with r == p    (owl:equivalentProperty, symmetric)
      - subproperty_pairs:    (r, p) with r <= p    (rdfs:subPropertyOf,    directional)
      - equivalent_class_pairs:(C, D) with C == D   (owl:equivalentClass,   symmetric)
      - subclass_pairs:       (C, D) with C <= D    (rdfs:subClassOf,       directional)
    """
    print(f"Loading ontology for axiom extraction: {owl_filepath}...")
    onto = get_ontology(str(owl_filepath)).load()

    inverse_pairs = set()
    equivalent_pairs = set()
    subproperty_pairs = set()
    equivalent_class_pairs = set()
    subclass_pairs = set()

    # ---- Relation (object property) axioms ----
    for prop in onto.object_properties():
        prop_id = kg.obj_prop_to_id(prop.iri)
        if prop_id is None or prop_id == -1:
            continue

        for super_prop in prop.is_a:
            if isinstance(super_prop, ObjectPropertyClass):
                super_id = kg.obj_prop_to_id(super_prop.iri)
                if super_id is not None and super_id != -1 and super_id != prop_id:
                    subproperty_pairs.add((prop_id, super_id))

        for eq_prop in prop.equivalent_to:
            if hasattr(eq_prop, "iri"):
                eq_id = kg.obj_prop_to_id(eq_prop.iri)
                if eq_id is not None and eq_id != -1 and eq_id != prop_id:
                    equivalent_pairs.add(tuple(sorted((prop_id, eq_id))))

        if prop.inverse_property:
            inv_id = kg.obj_prop_to_id(prop.inverse_property.iri)
            if inv_id is not None and inv_id != -1 and inv_id != prop_id:
                inverse_pairs.add(tuple(sorted((prop_id, inv_id))))

    # ---- Class axioms (only if classes are part of the entity vocabulary) ----
    if entity_to_id:
        for cls in onto.classes():
            if cls.iri in TOP_CLASSES:
                continue
            c_id = entity_to_id.get(cls.iri)
            if c_id is None:
                continue

            for sup in cls.is_a:
                # named superclass, excluding the trivial top class (owl:Thing / schema:Thing)
                if isinstance(sup, ThingClass) and sup.iri not in TOP_CLASSES:
                    s_id = entity_to_id.get(sup.iri)
                    if s_id is not None and s_id != c_id:
                        subclass_pairs.add((c_id, s_id))

            for eq in cls.equivalent_to:
                if isinstance(eq, ThingClass) and eq.iri not in TOP_CLASSES:
                    e_id = entity_to_id.get(eq.iri)
                    if e_id is not None and e_id != c_id:
                        equivalent_class_pairs.add(tuple(sorted((c_id, e_id))))

    inverse_pairs = [list(p) for p in inverse_pairs]
    equivalent_pairs = [list(p) for p in equivalent_pairs]
    subproperty_pairs = [list(p) for p in subproperty_pairs]
    equivalent_class_pairs = [list(p) for p in equivalent_class_pairs]
    subclass_pairs = [list(p) for p in subclass_pairs]
    print(f"Extracted {len(inverse_pairs)} inverseOf, "
          f"{len(equivalent_pairs)} equivalentProperty, "
          f"{len(subproperty_pairs)} subPropertyOf, "
          f"{len(equivalent_class_pairs)} equivalentClass, "
          f"{len(subclass_pairs)} subClassOf pairs.")
    return (inverse_pairs, equivalent_pairs, subproperty_pairs,
            equivalent_class_pairs, subclass_pairs)
