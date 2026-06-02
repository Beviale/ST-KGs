from owlready2 import get_ontology


def extract_axiom_pairs(owl_filepath: str, kg):
    """Extract inverseOf and equivalentProperty pairs from the OWL ontology.

    Returns two lists of relation-ID pairs (consistent with
    object_property_to_id / kg.obj_prop_to_id):
      - inverse_pairs:     (r, q) with r == q^-  (owl:inverseOf)
      - equivalent_pairs:  (r, p) with r == p    (owl:equivalentProperty)
    """
    print(f"Loading ontology for axiom extraction: {owl_filepath}...")
    onto = get_ontology(str(owl_filepath)).load()

    inverse_pairs = set()
    equivalent_pairs = set()

    for prop in onto.object_properties():
        prop_id = kg.obj_prop_to_id(prop.iri)
        if prop_id is None or prop_id == -1:
            continue

        # equivalentProperty (symmetric -> sort the pair to deduplicate)
        for eq_prop in prop.equivalent_to:
            if hasattr(eq_prop, "iri"):
                eq_id = kg.obj_prop_to_id(eq_prop.iri)
                if eq_id is not None and eq_id != -1 and eq_id != prop_id:
                    equivalent_pairs.add(tuple(sorted((prop_id, eq_id))))

        # inverseOf
        if prop.inverse_property:
            inv_id = kg.obj_prop_to_id(prop.inverse_property.iri)
            if inv_id is not None and inv_id != -1 and inv_id != prop_id:
                inverse_pairs.add(tuple(sorted((prop_id, inv_id))))

    inverse_pairs = [list(p) for p in inverse_pairs]
    equivalent_pairs = [list(p) for p in equivalent_pairs]
    print(f"Extracted {len(inverse_pairs)} inverseOf and "
          f"{len(equivalent_pairs)} equivalentProperty pairs.")
    return inverse_pairs, equivalent_pairs
