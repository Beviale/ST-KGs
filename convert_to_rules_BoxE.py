from owlready2 import *

def convert_owl_to_boxe(owl_filepath: str, kg, output_filepath: str):
    """
        Converts hierarchies and equivalences of an OWL ontology into BoxE rules format.
    """
    print(f"Loading ontology: {owl_filepath}...")
    try:
        onto = get_ontology(str(owl_filepath)).load()
    except Exception as e:
        print(f"An error occured while loading the ontology: {e}")
        return

    rules = []
    
    ents = "[0,1]"
    ents_inv = "[1,0]" # For symmetric properties
    
    for prop in onto.object_properties():
        prop_id = kg.obj_prop_to_id(prop.iri) 
        
        if prop_id is None or prop_id == -1:
            continue 

        for super_prop in prop.is_a:
            if isinstance(super_prop, ObjectPropertyClass):
                super_id = kg.obj_prop_to_id(super_prop.iri)
                
                if super_id is not None and super_id != -1 and super_id != prop_id:
                    rule = f"{prop_id}{ents} > {super_id}{ents}\n"
                    rules.append(rule)

        for eq_prop in prop.equivalent_to:
            if hasattr(eq_prop, 'iri'):
                eq_id = kg.obj_prop_to_id(eq_prop.iri)
                if eq_id is not None and eq_id != -1 and prop_id != eq_id:
                    rules.append(f"{prop_id}{ents} = {eq_id}{ents}\n")
            
            elif isinstance(eq_prop, And):
                component_ids = [kg.obj_prop_to_id(p.iri) for p in eq_prop.Classes if kg.obj_prop_to_id(p.iri) is not None]
                component_ids = [idx for idx in component_ids if idx != -1]
                
                if len(component_ids) >= 2:
                    lhs_str = f"{component_ids[0]}{ents}&{component_ids[1]}{ents}"
                    for comp_id in component_ids[2:]:
                        lhs_str = f"({lhs_str})&{comp_id}{ents}"
                    
                    rules.append(f"{lhs_str} > {prop_id}{ents}\n")

        if SymmetricProperty in prop.is_a or any(issubclass(x, SymmetricProperty) for x in prop.is_a if isinstance(x, type)):
            rules.append(f"{prop_id}{ents} = {prop_id}{ents_inv}\n")

        if prop.inverse_property:
            inv_id = kg.obj_prop_to_id(prop.inverse_property.iri)
            if inv_id is not None and inv_id != -1 and prop_id != inv_id:
                rules.append(f"{prop_id}{ents} = {inv_id}{ents_inv}\n")

    rules = list(set(rules))
    with open(output_filepath, 'w') as f:
        f.writelines(rules)

        