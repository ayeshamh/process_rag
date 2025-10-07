import re
import logging
from typing import Dict, Set, List, Optional, Tuple, Union
from collections import defaultdict, Counter
from .entity import Entity
from .relation import Relation
from .attribute import Attribute, AttributeType

logger = logging.getLogger(__name__)

class LabelNormalizer:
    """
    Simplified German label normalizer for German text data.
    Focuses on basic cleaning, abbreviation expansion, and German title case.
    """
    
    def __init__(self):
        # German-specific abbreviation mappings
        self.ABBREVIATION_MAPPINGS = {
            # Organizations
            "org": "Organisation", "orgs": "Organisation",
            "gmbh": "GmbH", "ag": "AG", "ev": "e.V.",
            
            # Departments/Units
            "abt": "Abteilung", "abts": "Abteilung",
            "dept": "Abteilung", "depts": "Abteilung",
            "einr": "Einrichtung", "einrs": "Einrichtung",
            
            # Documents
            "dok": "Dokument", "doks": "Dokument",
            "dokument": "Dokument", "dokumente": "Dokument",
            "ber": "Bericht", "berichte": "Bericht",
            "prot": "Protokoll", "protokolle": "Protokoll",
            
            # Medical/Healthcare
            "med": "Medikament", "meds": "Medikament",
            "medikament": "Medikament", "medikamente": "Medikament",
            "pat": "Patient", "patienten": "Patient",
            "arzt": "Arzt", "ärzte": "Arzt",
            "pfleger": "Pfleger", "pflegerinnen": "Pfleger",
            
            # Technical
            "tech": "Techniker", "techs": "Techniker",
            "techniker": "Techniker", "technikerinnen": "Techniker",
            "admin": "Administrator", "admins": "Administrator",
            "koord": "Koordinator", "koords": "Koordinator",
            "koordinator": "Koordinator", "koordinatorinnen": "Koordinator",
            
            # Management
            "mgmt": "Management", "mgr": "Manager",
            "manager": "Manager", "managerinnen": "Manager",
            "dir": "Direktor", "dirs": "Direktor",
            "direktor": "Direktor", "direktorinnen": "Direktor",
            "sup": "Supervisor", "sups": "Supervisor",
            
            # General
            "pers": "Person", "personen": "Person",
            "ass": "Assistent", "asss": "Assistent",
            "assistent": "Assistent", "assistentinnen": "Assistent",
            "repr": "Repräsentant", "reprs": "Repräsentant",
            "info": "Information", "komm": "Kommunikation",
            "kommunikation": "Kommunikation",
        }
        
        # Common German words that should remain lowercase in title case
        self.LOWERCASE_WORDS = {
            'von', 'und', 'oder', 'der', 'die', 'das', 'ein', 'eine', 'eines', 'einer',
            'in', 'an', 'bei', 'für', 'mit', 'zu', 'auf', 'über', 'unter', 'durch',
            'ohne', 'gegen', 'während', 'trotz', 'wegen', 'statt', 'anstatt', 'außer',
            'innerhalb', 'außerhalb', 'oberhalb', 'unterhalb', 'diesseits', 'jenseits',
            'beiderseits', 'seitens', 'seit', 'bis', 'ab', 'aus', 'nach', 'vor', 'um',
            'gegenüber', 'entlang', 'längs', 'quer', 'zwischen', 'neben', 'hinter',
            'des', 'dem', 'den', 'im', 'am', 'zum', 'zur', 'beim', 'vom', 'zum', 'zur'
        }
    
    def normalize_entity_label(self, label: str, description: str = "") -> Tuple[str, Optional[str]]:
        """
        Normalize German entity label with basic cleaning, abbreviation expansion, and title case.
        
        Args:
            label (str): The entity label to normalize
            description (str): Optional description (kept for compatibility)
            
        Returns:
            Tuple of (normalized_label, None) - simplified for German data
        """
        if not label:
            return "", None
            
        # Step 1: Basic text cleaning
        normalized = self._preprocess_label(label)
        
        # Step 2: Expand German abbreviations
        normalized = self._expand_abbreviations(normalized)
        
        # Step 3: Apply German title case
        normalized = self._apply_german_title_case(normalized)
        
        return normalized, None  # No superclass detection for simplified approach
    
    def normalize_relation_label(self, label: str) -> str:
        """
        Normalize relation labels to UPPER_CASE format.
        
        Args:
            label (str): The relation label to normalize
            
        Returns:
            str: Normalized relation label in UPPER_CASE
        """
        if not label:
            return ""
            
        # Clean the label
        cleaned = re.sub(r'[^\w\s]', '', label.strip())
        
        # Convert to UPPER_CASE with underscores
        normalized = re.sub(r'\s+', '_', cleaned.upper())
        
        return normalized
    
    def _preprocess_label(self, label: str) -> str:
        """Basic preprocessing for German text."""
        # Remove special characters except spaces, hyphens, parentheses, slashes, and German umlauts
        cleaned = re.sub(r'[^\w\s\-äöüßÄÖÜ()/]', '', label.strip())
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()
    
    def _expand_abbreviations(self, label: str) -> str:
        """Expand known German abbreviations."""
        words = label.split()
        expanded_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.ABBREVIATION_MAPPINGS:
                # Special handling for company types that should stay as-is
                if word_lower in ['gmbh', 'ag', 'ev']:
                    expanded_words.append(self.ABBREVIATION_MAPPINGS[word_lower])
                else:
                    expanded_words.append(self.ABBREVIATION_MAPPINGS[word_lower])
            else:
                expanded_words.append(word)
                
        return ' '.join(expanded_words)
    
    def _apply_german_title_case(self, label: str) -> str:
        """Apply German title case rules (nouns are capitalized)."""
        words = label.split()
        title_words = []
        
        for i, word in enumerate(words):
            # Keep special company types as-is (GmbH, AG, e.V.)
            if word in ['GmbH', 'AG', 'e.V.']:
                title_words.append(word)
            # Keep abbreviations as-is if all caps or mixed case
            elif word.isupper() and len(word) <= 4:
                title_words.append(word)
            # Keep common German words lowercase (except first word)
            elif i > 0 and word.lower() in self.LOWERCASE_WORDS:
                title_words.append(word.lower())
            else:
                # Capitalize first letter (German nouns are capitalized)
                title_words.append(word.capitalize())
        
        return ' '.join(title_words)
