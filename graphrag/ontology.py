import re
import json
import logging
from typing import List, Dict, Optional, Union
import traceback
import os 
from falkordb import Graph
from .entity import Entity
from .relation import Relation
from .base import AbstractSource
from .models import GenerativeModel
from .attribute import Attribute, AttributeType

logger = logging.getLogger(__name__)

def _process_attributes_from_graph(attributes: list[list[list[str]]]) -> list[Attribute]:
    """
    Processes the attributes extracted from the graph and converts them into the SDK convention.

    Args:
        attributes (list[list[list[str]]]): The attributes extracted from the graph.

    Returns:
        processed_attributes (list[Attribute]): The processed attributes.
    """
    processed_attributes = []
    for attr in attributes:
        attr_name, attr_type = attr[0]
        try:
            attr_type = AttributeType.from_string(attr_type)
        except:
            continue
        
        processed_attributes.append(Attribute(attr_name, attr_type))

    return processed_attributes
class Ontology(object):
    """
    Represents an ontology, which is a collection of entities and relations.

    Attributes:
        entities (list[Entity]): The list of entities in the ontology.
        relations (list[Relation]): The list of relations in the ontology.
    """

    def __init__(self, entities: Optional[list[Entity]] = None, relations: Optional[list[Relation]] = None):
        """
        Initialize the Ontology class.

        Args:
            entities (Optional[list[Entity]]): List of Entity objects. Defaults to None.
            relations (Optional[list[Relation]]): List of Relation objects. Defaults to None.
        """
        self.entities = entities or []
        self.relations = relations or []

    @staticmethod
    def from_sources(
        sources: list[AbstractSource],
        model: GenerativeModel,
        boundaries: Optional[str] = None,
        hide_progress: bool = False,
        auto_validate: bool = True,
        language: Optional[str] = "GERMAN",
    ) -> "Ontology":
        """
        Create an Ontology object from a list of sources with improved JSON handling.

        Args:
            sources (list[AbstractSource]): A list of AbstractSource objects representing the sources.
            model (GenerativeModel): The generative model to use.
            boundaries (Optional[str]): The boundaries for the ontology.
            hide_progress (bool): Whether to hide the progress bar.
            auto_validate (bool): Whether to automatically validate and optimize the ontology.
            language (Optional[str]): Language name for prompts ("German" or "English"). Defaults to "GERMAN".

        Returns:
            The created Ontology object.
        """
        # Import here to avoid circular imports
        from .steps.create_ontology_step import CreateOntologyStep
        
        step = CreateOntologyStep(
            sources=sources,
            ontology=Ontology(),
            model=model,
            hide_progress=hide_progress,
            language=language,
        )

        ontology = step.run(boundaries=boundaries)
        
        # Extra validation and optimization for automatically generated ontologies
        if auto_validate:
            # Remove duplicate entities
            entity_labels = set()
            unique_entities = []
            for entity in ontology.entities:
                if entity.label not in entity_labels:
                    entity_labels.add(entity.label)
                    unique_entities.append(entity)
            ontology.entities = unique_entities
            
            # Remove relations without valid entities
            ontology.discard_relations_without_entities()
            
            # Add default attributes for common entity types
            for entity in ontology.entities:
                # Add source attribute to track origin
                if not any(attr.name == "source" for attr in entity.attributes):
                    entity.attributes.append(Attribute("source", AttributeType.STRING))
        
        return ontology

    @staticmethod
    def from_ndjson(ndjson_input: Union[str, list]) -> "Ontology":
        """
        Create an Ontology object from NDJSON input.
        
        Args:
            ndjson_input (Union[str, list]): Either a string containing NDJSON or a list of dicts
            
        Returns:
            Ontology: The created Ontology object
            
        Raises:
            ValueError: If the input cannot be parsed or is invalid
        """
        try:
            if isinstance(ndjson_input, str):
                from .helpers import parse_ndjson
                data = parse_ndjson(ndjson_input)
            else:
                data = ndjson_input

            if not data:
                raise ValueError("Empty or invalid NDJSON input")

            entities = []
            relations = []
            for obj in data:
                if not isinstance(obj, dict):
                    continue
                obj_type = obj.get("type")
                if obj_type == "entity_definition":
                    entities.append(Entity.from_json(obj))
                elif obj_type == "relation_definition":
                    relations.append(Relation.from_json(obj))

            return Ontology(entities, relations)
            
        except Exception as e:
            logger.error(f"Error parsing {obj_type}: {obj}\nException: {e}")
            logger.error(f"Failed to create ontology from NDJSON: {str(e)}")
            raise ValueError(f"Failed to create ontology from NDJSON: {str(e)}")

    @staticmethod
    def from_schema_graph(graph: Graph) -> "Ontology":
        """
        Creates an Ontology object from a given schema graph.

        Args:
            graph (Graph): The graph object representing the ontology.

        Returns:
            The Ontology object created from the graph.
        """
        ontology = Ontology()

        entities = graph.query("MATCH (n) RETURN n").result_set
        for entity in entities:
            ontology.add_entity(Entity.from_graph(entity[0]))

        for relation in graph.query("MATCH ()-[r]->() RETURN r").result_set:
            ontology.add_relation(
                Relation.from_graph(relation[0], [x for xs in entities for x in xs])
            )

        return ontology
    
    @staticmethod
    def from_kg_graph(graph: Graph, sample_size: Optional[int] = 100) -> "Ontology":
        """
        Constructs an Ontology object from a given Knowledge Graph.

        This function queries the provided knowledge graph to extract:
        1. Entities and their attributes.
        2. Relationships between entities and their attributes.

        Args:
            graph (Graph): The graph object representing the knowledge graph.
            sample_size (Optional[int]): The maximum number of attributes to sample for each entity and relationship. Defaults to 100.

        Returns:
            Ontology: The Ontology object constructed from the Knowledge Graph.
        """
        ontology = Ontology()

        # Retrieve all node labels and edge types from the graph.
        n_labels = graph.call_procedure("db.labels").result_set
        e_types = graph.call_procedure("db.relationshipTypes").result_set
        
        # Extract attributes for each node label, limited by the specified sample size.
        for lbls in n_labels:
            l = lbls[0]
            attributes = graph.query(
                f"""MATCH (a:{l}) call {{ with a return [k in keys(a) | [k, typeof(a[k])]] as types }}
                WITH types limit {sample_size} unwind types as kt RETURN kt, count(1) ORDER BY kt[0]""").result_set
            attributes = _process_attributes_from_graph(attributes)
            ontology.add_entity(Entity(l, attributes))

        # Extract attributes for each edge type, limited by the specified sample size.
        for e_type in e_types:
            e_t = e_type[0]
            attributes = graph.query(
                            f"""MATCH ()-[a:{e_t}]->() call {{ with a return [k in keys(a) | [k, typeof(a[k])]] as types }}
                            WITH types limit {sample_size} unwind types as kt RETURN kt, count(1) ORDER BY kt[0]""").result_set
            attributes = _process_attributes_from_graph(attributes)
            for s_lbls in n_labels:
                for t_lbls in n_labels:
                    s_l = s_lbls[0]
                    t_l = t_lbls[0]
                    # Check if a relationship exists between the source and target entity labels
                    result_set = graph.query(f"MATCH (s:{s_l})-[a:{e_t}]->(t:{t_l}) return a limit 1").result_set
                    if len(result_set) > 0:
                        ontology.add_relation(Relation(e_t, s_l, t_l, attributes))
        
        return ontology
    
    @staticmethod
    def from_json(json_input: Union[str, dict, list]) -> "Ontology":
        """
        Create an Ontology object from regular JSON input.
        Args:
            json_input (Union[str, dict, list]): JSON string, dict, or list
        Returns:
            Ontology: The created Ontology object
        """
        import json as _json
        if isinstance(json_input, str):
            data = _json.loads(json_input)
        else:
            data = json_input

        # If it's a dict with 'entities' and 'relations' keys
        if isinstance(data, dict) and "entities" in data and "relations" in data:
            entities = [Entity.from_json(e) for e in data["entities"]]
            relations = [Relation.from_json(r) for r in data["relations"]]
            return Ontology(entities, relations)
        # If it's a list, try to split into entities/relations by type
        elif isinstance(data, list):
            entities = [Entity.from_json(e) for e in data if e.get("type") == "entity_definition"]
            relations = [Relation.from_json(r) for r in data if r.get("type") == "relation_definition"]
            return Ontology(entities, relations)
        else:
            raise ValueError("Invalid JSON format for Ontology")

    def add_entity(self, entity: Entity) -> None:
        """
        Adds an entity to the ontology.

        Args:
            entity: The entity object to be added.
        """
        self.entities.append(entity)

    def add_relation(self, relation: Relation) -> None:
        """
        Adds a relation to the ontology.

        Args:
            relation (Relation): The relation to be added.
        """
        self.relations.append(relation)

    def to_json(self) -> dict:
        """
        Converts the ontology object to a JSON representation.

        Returns:
            A dictionary representing the ontology object in JSON format.
        """
        return {
            "entities": [entity.to_json() for entity in self.entities],
            "relations": [relation.to_json() for relation in self.relations],
        }

    def merge_with(self, o: "Ontology"):
        """
        Merges the given ontology `o` with the current ontology.

        Args:
            o (Ontology): The ontology to merge with.

        Returns:
            The merged ontology.
        """
        # Merge entities
        for entity in o.entities:
            if entity.label not in [n.label for n in self.entities]:
                # Entity does not exist in self, add it
                self.entities.append(entity)
                logger.debug(f"Adding entity {entity.label}")
            else:
                # Entity exists in self, merge attributes
                entity1 = next(n for n in self.entities if n.label == entity.label)
                entity1.merge(entity)

        # Merge relations
        for relation in o.relations:
            if relation.label not in [e.label for e in self.relations]:
                # Relation does not exist in self, add it
                self.relations.append(relation)
                logger.debug(f"Adding relation {relation.label}")
            else:
                # Relation exists in self, merge attributes
                relation1 = next(e for e in self.relations if e.label == relation.label)
                relation1.combine(relation)

        return self

    def discard_entities_without_relations(self):
        """
        Discards entities that do not have any relations in the ontology.

        Returns:
            The updated ontology object after discarding entities without relations.
        """
        entities_to_discard = [
            entity.label
            for entity in self.entities
            if all(
                [
                    relation.source.label != entity.label
                    and relation.target.label != entity.label
                    for relation in self.relations
                ]
            )
        ]

        self.entities = [
            entity
            for entity in self.entities
            if entity.label not in entities_to_discard
        ]
        self.relations = [
            relation
            for relation in self.relations
            if relation.source.label not in entities_to_discard
            and relation.target.label not in entities_to_discard
        ]

        if len(entities_to_discard) > 0:
            logger.info(f"Discarded entities: {', '.join(entities_to_discard)}")

        return self

    def discard_relations_without_entities(self):
        """
        Discards relations that have entities not present in the ontology.

        Returns:
            The current instance of the Ontology class.
        """
        relations_to_discard = [
            relation.label
            for relation in self.relations
            if relation.source.label not in [entity.label for entity in self.entities]
            or relation.target.label not in [entity.label for entity in self.entities]
        ]

        self.relations = [
            relation
            for relation in self.relations
            if relation.label not in relations_to_discard
        ]

        if len(relations_to_discard) > 0:
            logger.info(f"Discarded relations: {', '.join(relations_to_discard)}")

        return self

    def validate_entities(self) -> bool:
        """
        Validates the entities in the ontology.

        This method checks for entities without unique attributes and logs a warning if any are found.

        Returns:
            True if all entities have unique attributes, False otherwise.
        """
        # Check for entities without unique attributes
        entities_without_unique_attributes = [
            entity.label
            for entity in self.entities
            if len(entity.get_unique_attributes()) == 0
        ]
        if len(entities_without_unique_attributes) > 0:
            logger.warn(
                f"""
*** WARNING ***
The following entities do not have unique attributes:
{', '.join(entities_without_unique_attributes)}
"""
            )
            return False
        return True

    def get_entity_with_label(self, label: str) -> Optional[Entity]:
        """
        Retrieves the entity with the specified label.

        Args:
            label (str): The label of the entity to retrieve.

        Returns:
            The entity with the specified label, or None if not found.
        """
        return next((n for n in self.entities if n.label == label), None)

    def get_relations_with_label(self, label: str) -> list[Relation]:
        """
        Returns a list of relations with the specified label.

        Args:
            label (str): The label to search for.

        Returns:
            A list of relations with the specified label.
        """
        return [e for e in self.relations if e.label == label]

    def has_entity_with_label(self, label: str) -> bool:
        """
        Checks if the ontology has an entity with the given label.

        Args:
            label (str): The label to search for.

        Returns:
            True if an entity with the given label exists, False otherwise.
        """
        return any(n.label == label for n in self.entities)

    def has_relation_with_label(self, label: str) -> bool:
        """
        Checks if the ontology has a relation with the given label.

        Args:
            label (str): The label of the relation to check.

        Returns:
            True if a relation with the given label exists, False otherwise.
        """
        return any(e.label == label for e in self.relations)

    def __str__(self) -> str:
        """
        Returns a string representation of the Ontology object.

        The string includes a list of entities and relations in the ontology.

        Returns:
            A string representation of the Ontology object.
        """
        return "Entities:\n\f- {entities}\n\nEdges:\n\f- {relations}".format(
            entities="\n- ".join([str(entity) for entity in self.entities]),
            relations="\n- ".join([str(relation) for relation in self.relations]),
        )

    def save_to_graph(self, graph: Graph) -> None:
        """
        Saves the entities and relations to the specified graph.

        Args:
            graph (Graph): The graph to save the entities and relations to.
        """
        for entity in self.entities:
            query = entity.to_graph_query()
            logger.debug(f"Query: {query}")
            graph.query(query)

        for relation in self.relations:
            query = relation.to_graph_query()
            logger.debug(f"Query: {query}")
            graph.query(query)
