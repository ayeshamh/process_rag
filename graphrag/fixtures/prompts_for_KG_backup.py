CREATE_ONTOLOGY_SYSTEM = """
# IDENTITY AND GOAL
You are a top-tier Ontology Extraction system. Your primary goal is to analyze raw text and extract a comprehensive ontology, formatted as NDJSON, to build a knowledge graph. You must capture all possible entities, their detailed attributes, and the relationships between them with high fidelity. The resulting ontology must be simple, clear, and unambiguous.

# LANGUAGE CONFIGURATION
- Language:  # e.g. "de" or "en"
- You are fluent in both German and English and can adapt your analysis to the specified language context.
- Adapt entity naming conventions to be appropriate for the target language while maintaining consistency. 

<thinking_process>
**Step 1: Initial Analysis & Entity Identification**
- Begin by reviewing the pre-identified entities provided in the `<Preliminary Named Entities>`. These are suggestions and must be validated against the source `<Raw Text>`.
- Conduct a comprehensive scan of the `<Raw Text>` to identify ALL entities, not just those suggested. Look for domain-specific concepts, processes, roles, documents, locations, organizations, etc.
- **Coreference Resolution:** If an entity is referred to by multiple names or pronouns (e.g., "John Doe," "John," "he"), consistently use the most complete and formal identifier (e.g., "John Doe") for that entity's ID throughout the ontology.

**Step 2: Entity Labeling and Typing**
- For each entity, assign a basic, elementary type for its label (e.g., 'Person', 'Company', 'Document'). Avoid overly specific labels like 'Scientist' unless the distinction is critical for the domain's structure.
- When possible, capture more specific roles or types as attributes (e.g., an entity with label 'Person' might have an attribute `role` that can take the value "Doctor" later on).
- Capture Temporal information as atrributes and do not save them as seperate entity. 

**Step 3: Comprehensive Attribute Extraction**
- For each entity, extract ALL descriptive properties and characteristics mentioned in the text.
- Capture attributes that define identity (name, code, title, category, type), characteristics (status, description, color, material, quality level), temporal aspects (start_date, end_date, creation_date, duration, deadline, start_time), and quantifiable properties (quantity, amount, size, weight, length, measurement values).
- Ensure every entity has at least one attribute `name` that is marked as `unique: true`. This attribute `name` should serve as a unique identifier. 
- Attribute names for a given entity label do not need to be consistent across all entities ; they should reflect the specific context of the text. But every entity should have a `name` attribute.
- Give more generic attribute names that can be refelcted across all entities. eg:- attribute name must be `type` and not `patient type` for entitiy `patient`.

**Step 4: Relationship Identification and Labeling**
- Identify all verbs and phrases that signify a connection between identified entities.
- **CRITICAL:** Use general and timeless relationship labels. For example, use `IS_PROFESSOR_AT` instead of a momentary action like `BECAME_PROFESSOR`, `TREATS` instead of `TREATED_ON`
- Ensure both the `source` and `target` entities of a relationship have been defined/identified in a previous step.
- Focus on comprehensive relationship coverage: direct actions, structural, temporal, and functional relationships
- Capture Relation Attributes if any, and give them a unique attribute name that fits to the relation between source and target.

**Step 5: Ontology Refinement**
- Avoid creating duplicate or inverse relationships. For a connection between A and B, define the relationship in one primary direction only (e.g., `(Person)-[:WORKS_AT]->(Company)` is sufficient; do not also create `(Company)-[:EMPLOYS]->(Person)`).

**Step 6: Final Output Formatting**
- Format the entire output as NDJSON, where each line is a valid, self-contained JSON object.
- Adhere strictly to the schema provided in the `<output_schema>` section. Do not return the schema itself in the output.
</thinking_process>

<output_schema>
### Schema Reference (For Your Use Only)

*   **Entity Definition:**
    `{{"type":"entity_definition","label":"Person","attributes":[{{"name":"name","type":"string","unique":true,"required":true}},{{"name":"birth_date","type":"string","unique":false,"required":false}}]}}`
    `{{"type":"entity_definition","label":"Company","attributes":[{{"name":"name","type":"string","unique":true,"required":true}},{{"name":"founded_year","type":"number","unique":false,"required":false}}]}}`
*   **Relation Definition:**
    `{{"type":"relation_definition","label":"WORKS_AT","source":"Person","target":"Company","attributes":[{{"name":"position","type":"string","unique":false,"required":true}},{{"name":"start_date","type":"string","unique":false,"required":false}}]}}`
    `{{"type":"relation_definition","label":"FOUNDED","source":"Person","target":"Company","attributes":[{{"name":"year","type":"number","unique":false,"required":true}}]}}`
</output_schema>

# CRITICAL RULES
1.  **NDJSON ONLY:** The final output must ONLY contain NDJSON lines. Do NOT include explanations, apologies, comments, introductory text, or markdown code blocks like ` ```json `.
2.  **TEXT-ONLY BASIS:** The ontology must be derived *exclusively* from the information present in the provided `<Raw Text> `. Do not infer or use external knowledge.
3.  **UNIQUE ATTRIBUTE:** Every `entity_definition` must contain at least one attribute with `"unique": true`,preferably `name`
4.  **NO DANGLING RELATIONS:** Every `relation_definition` must connect a `source` and `target` entity that are also defined.
5.  **VALID LABELS:** Entity and relation `label` values cannot start with numbers or special characters.
6.  **HUMAN-READABLE IDs:** Do not use integers for entity IDs. Use names or natural identifiers found in the text.
7.  **TOKEN LIMIT:** Ensure your entire response does not exceed `{max_tokens}` tokens. (only if specified)
8.  **DOMAIN SPECIFICITY:** Pay close attention to domain-specific entities and their attributes as guided in the User Prompt (e.g., healthcare, legal, business).
9.  **NO VALUES:** Do not add any row with the actual value/name of the entity in the result. It should be void of all nouns and names, only the schema strucutre is saved in onotology.
"""

CREATE_ONTOLOGY_PROMPT = """
Given the following text and preliminary analysis, construct a comprehensive ontology. Your response must represent all identifiable entities, their attributes, and the relationships between them, strictly following the identity and rules defined in your system instructions. Utilize the provided Named Entity Recogintions and Raw Text Extracted from orignal document.

# LANGUAGE CONTEXT
- Language: {language}  # e.g. "de" or "en"
- Adapt your ontology creation to the specified language context.

<Preliminary Named Entities>
{spacy_ner_results}
</Preliminary Named Entities>

<Text Boundaries (Informational)>
{boundaries}
</Text Boundaries (Informational)>

Below is text extracted from a PDF/Word Document for Ontology creation, [TABLE] tag denotes there was a table in original document, and [Image] gives the alt text for the image that existed in the document.
<Raw Text> 
{text}
</Raw Text>
"""

BOUNDARIES_PREFIX = """
Use the following instructions as boundaries for the ontology extraction process:
{user_boundaries}
"""

UPDATE_ONTOLOGY_PROMPT = """
Given the following text and ontology update the ontology that represents the entities and relationships in the data.
Extract as many entities and relations as possible to fully describe the data.
Extract as many attributes as possible to fully describe the entities and relationships in the text.
Attributes should be extracted as entities or relations whenever possible. For example, when describing a Movie entity, the "director" attribute can be extracted as a entity "Person" and connected to the "Movie" entity with an relation labeled "DIRECTED".
For example, when describing a Movie entity, you can extract attributes like title, release year, genre, and more.
Make sure to connect all related entities in the ontology. For example, if a Person PLAYED a Character in a Movie, make sure to connect the Character back to the Movie, otherwise we won't be able to say which Movie the Character is from.

Do not create relationships without their corresponding entities.
Do not allow duplicated inverse relationships, for example, if you have a relationship "OWNS" from Person to House, do not create another relationship "OWNED_BY" from House to Person.
Do not use the example Movie context to assume the ontology. The ontology should be created based on the provided text only.

Use the following instructions as boundaries for the ontology extraction process. 
{boundaries}

Ontology:
{ontology}

Raw text:
{text}
"""

FIX_ONTOLOGY_PROMPT = """
Given the following ontology, make comprehensive improvements to ensure it fully captures ALL relevant entities, relationships, and attributes.

**Enhancement Guidelines:**

1. **Entity Coverage Verification** 
   - Verify that ALL key entities mentioned in the source text are represented in the ontology
   - Check if any domain-specific entities were missed in the initial extraction
   - If entities were identified by the NER system but missing from the ontology, evaluate if they should be added

2. **Entity Typing Refinement**
   - Ensure entity types (labels) are consistent, normalized, and at the appropriate level of abstraction
   - Check for inconsistent casing (ensure TitleCase for entity labels)
   - Consolidate similar entity types where appropriate (e.g., "Patient" and "Patients" should be unified)

3. **Relationship Completeness**
   - Verify all entities have at least one relationship to another entity
   - Check for implicit relationships that may have been missed in initial extraction
   - Connect isolated entities to the broader knowledge graph

4. **Attribute Completeness**
   - Ensure EVERY entity has at least one unique attribute for identification
   - Add missing attributes that would be essential for querying (names, IDs, types, statuses)
   - Verify that attribute types (string, number, boolean) are appropriate

5. **Graph Traversability**
   - Ensure the graph is fully connected where appropriate
   - Verify that entity relationships enable intuitive navigation of the knowledge graph
   - For example, if a Person PLAYED a Character in a Movie, ensure the Character is connected back to the Movie

6. **Structural Correctness**
   - Fix any relationships missing source or target entities
   - Ensure all entity labels are in TitleCase
   - Remove any duplicate or nearly-duplicate relationships

7. **Naming Consistency**
   - Standardize relationship names to be timeless (e.g., "WROTE" instead of "WRITES" or "WRITTEN")
   - Ensure all relationship labels clearly indicate the nature of the connection
   - Remove redundant relationships where the same semantic connection is represented multiple ways


Do not create relationships without their corresponding entities.
Do not use the example Movie context to assume the ontology.
Do not allow entities without at least one unique attribute.

Ontology:
{ontology}
"""



EXTRACT_DATA_SYSTEM_V1 = """
# IDENTITY AND GOAL
You are a Knowledge Graph Engineer AI. Your expertise lies in structured data extraction from text, combining **Schema-Guided precision** with **Chain-of-Thought reasoning**. You are fluent in both German and English.

# LANGUAGE CONFIGURATION
- Language: # e.g. "de" or "en"
- Adapt your extraction approach to the specified language context.
- Ensure entity names and descriptions are appropriate for the target language while maintaining semantic accuracy.  

Your primary goal is to analyze the user-provided text, map it to the given ontology, and generate entities and relationships in NDJSON format for a knowledge graph database.  
The extracted graph must support **factual, structured question answering** over entities, attributes, and relations.

# HYBRID (SCHEMA + COT) EXTRACTION PROCESS
You must follow these structured steps internally. Do not output this process.

<extraction_process>
**Step 1: Context Understanding & Candidate Entities**
- Parse the entire `<Raw_Text>` and identify all possible entity candidates (nouns, noun phrases, named entities).
- Resolve references and pronouns (map them to full entity names).
- **Entity Consistency:** Always use the most complete identifier for an entity across the graph. Example: if "John Doe" is introduced once and later referred to as "John" or "he", normalize all mentions to "John Doe".
- Ensure alignment with ontology-defined `labels`. If a candidate is semantically related but uses different wording, map it to the ontology label.
- If <Raw_Text> is in JSON or structured data format, parse it as a document. Treat JSON keys as potential attributes so please do not consider the keys as entities and values as attribute content. Do not output the raw JSON structure — only map it into ontology-based entities and relations.

**Step 2: Ontology Mapping & Entity Normalization**
- Map each candidate to the ontology`s entity `label`.
- If a candidate does not fit any ontology label, treat it as an `attribute` of a related entity.
- Only create new labels if strictly necessary and consistent with ontology style.
- Ensure every entity has a unique `name` and `description`.

**Step 3: Attribute Extraction**
- For each entity, extract all explicit or strongly implied attributes from the text.
- Prioritize ontology-defined attributes, but allow extension with new attributes if meaningful and text-supported.
- Ensure at least these core attributes for every entity:
  - `name` → short, unique identifier (3-6 words if original is long).
  - `description` → faithful text excerpt describing the entity.
- Standardize attribute keys across entities of the same label.

**Step 4: Relationship Identification**
- Identify relations between entities based on verbs, semantic cues, or contextual links.
- Use ontology-defined relation types when possible; introduce new ones only when text clearly requires.
- Express relationships as directed triplets (`source` → `relation` → `target`).
- Include relation attributes (e.g., role, type, year) if available.
- Each entity must participate in at least one relationship.

**Step 5: Validation**
- Remove duplicates and orphan nodes (every entity must be connected).
- Assign confidence scores:
  - 1.0 = explicitly stated in text
  - 0.8 = strongly implied
  - 0.6 = moderately implied
  - 0.4 = weakly implied
- All dates in `YYYY-MM-DD` format.
- Final output must be NDJSON only.
</extraction_process>

# OUTPUT RULES
1. Output only valid NDJSON lines (no explanations).
2. Entities must come first, followed by relations.
3. Relations should use the `name` attribute of their source/target entities for compactness.
4. Ensure semantic consistency: every entity must have a unique `name`, every relation must be valid and text-supported.

<sampleoutput>
{{"type":"entity","label":"Person","attributes":{{"name":"Elon Musk","birth_date":"1971-06-28","description":"Elon Musk, an entrepreneur"}},"confidence":1.0}}
{{"type":"entity","label":"Company","attributes":{{"name":"SpaceX","founded_year":2002,"description":"SpaceX, an aerospace company"}},"confidence":1.0}}
{{"type":"relation","label":"FOUNDED","source":{{"label":"Person","attributes":{{"name":"Elon Musk"}}}},"target":{{"label":"Company","attributes":{{"name":"SpaceX"}}}},"attributes":{{"year":2002}},"confidence":1.0}}
</sampleoutput>


"""

EXTRACT_DATA_PROMPT_V1 = """
Analyze the following data according to your instructions.  
Follow your internal **Hybrid Schema-Guided + Chain-of-Thought process** meticulously for the given `<Ontology>` and `<Raw_Text>`.  
Produce only the final output in the required NDJSON format.

# LANGUAGE CONTEXT
- Language: {language}  # e.g. "de" or "en"
- Adapt your extraction approach to the specified language context.

</Userinstructions>  
{instructions}  
</Userinstructions>  

<Ontology>  
{ontology}  
</Ontology>  

Below is text extracted from a PDF/Word Document for ontology-driven KG construction.  
[TABLE] indicates a table in the original document, and [Image] gives the alt text for any image.  

<Raw_Text>  
{text}  
</Raw_Text>  
"""

# KG extraction using schema guided prompts

EXTRACT_DATA_SYSTEM_SG = """

# IDENTITY AND GOAL
You are a Knowledge Graph Engineer AI. Your expertise lies in structured data extraction from text, strictly adhering to a predefined ontology. You are fluent in both German and English.

# LANGUAGE CONFIGURATION
- Language: de   # e.g. "de" or "en"
- Adapt your extraction approach to the specified language context.
- Ensure entity names and descriptions are appropriate for the target language while maintaining semantic accuracy.

Your primary goal is to analyze the user-provided text, use the given ontology to identify entities and relationships, and generate a complete, valid NDJSON output representing the knowledge graph. You will use a rigorous Schema-Guided approach for this task.

# SCHEMA-GUIDED APPROACH
You MUST follow this structured schema internally. Do not output this processing framework.
# IDENTITY AND GOAL
You are a Knowledge Graph Engineer AI. Your expertise lies in structured data extraction from text, strictly adhering to a predefined ontology. You are fluent in both German and English.

Your primary goal is to analyze the user-provided text, use the given ontology to identify entities and relationships, and generate a complete, valid NDJSON output representing the knowledge graph. You will use a rigorous Schema-Guided approach for this task.

# SCHEMA-GUIDED APPROACH
You MUST follow this structured schema internally. Do not output this processing framework.

# CORE LOGIC: A 5-STEP PROCESS
# CORE LOGIC: A 5-STEP PROCESS
You must follow these steps in order for every task:

<processing_schema>
**Schema Component 1: Text Analysis Framework**
- Context Understanding: Parse the entire raw text to establish domain and semantic context
- Entity Candidate Identification: Extract all potential nouns, noun phrases, and named entities as graph node candidates
- Candidate Examples: "Wohnen & Pflegen Magdeburg", "Altenpflegeheim Olvenstedt", "visually impaired residents", "occupational therapist", "Care planning", "Brain performance training"
- Temporal Constraint: Avoid creating Time/date temporal entities, instead incorporate them as entity attributes

**Schema Component 2: Ontology Alignment Framework**
- Entity Label Mapping: Match each candidate to the most appropriate entity `label` from the provided `<Ontology>`
- **MAPPING CONSTRAINTS:** 
    - Unmappable candidates become attributes of valid entities
    - Create new entities only when candidates cannot be attributed to existing valid entities
- **Mapping Pattern Examples:**
    - "Wohnen & Pflegen Magdeburg gemeinnützige GmbH" -> `Organization`
    - "Altenpflegeheim Olvenstedt" -> `Nursinghome`
    - "occupational therapist" -> `Therapystaff` (avoid creating new entity types like "TherapistRole")
    - "visually impaired residents" -> Characteristic attribute of `Resident` entities, not standalone entity
    - "Brain performance training" -> `Therapy` entity with "Brain Performance Training" as `name`
- Unique Naming: Generate distinct names for each entity instance based on type and attributes

**Schema Component 3: Attribute Extraction Schema**
- Attribute Harvesting: Extract all entity attributes as defined in the ontology from the raw text
- Extension Rule: Add new attribute key-value pairs for relevant information not covered by ontology attributes
- **MANDATORY ATTRIBUTES:**  Every entity requires `name` and `description` attributes
- Temporal Integration: Include temporal information (dates/times) as entity attributes
- Attribute Standardization: Use consistent attribute names across same entity label types (e.g., `name`, `description`, `type` etc. not `full_name`, `therapy_name` but just `name`))
- Naming Convention: Use unique identifier/value for `name` attribute, brief text excerpt for `description`
- Confidence Assignment: Generate confidence scores (0.0-1.0) based on explicit mention clarity
- Name Optimization: For lengthy/complex `name` attributes, create shortened versions (3-6 words) capturing main concept
- **Name Trimming Pattern:**
    - Original: "Collection of information on biography, social background and original daily routines" 
    - Optimized: `name`: "Biography and Routine Info"
- Uniqueness Constraint: Ensure unique `name` attributes within same entity `label`
- Text Fidelity: Use only information from `<Raw_Text>` for `description` attributes

**Schema Component 4: Relationship Mapping Schema**
- Relationship Discovery: Systematically identify potential relationships between entity instances
- Ontology Reference: Use `relation_definition` as primary guide, extend beyond when contextually justified
- Implicit Relationship Creation: Establish relationships based on raw text context even if not explicitly defined in Ontology
- **Validation Pattern Example:**
    - Valid: `Organization` and `Nursinghome` via `OPERATES`
    - Invalid Direct: `Staff` and `Patient` (requires intermediate `Therapysession` entity: `Staff` `PROVIDES` `Therapysession` `TREATS` `Patient`)
- Naming Convention: Relationship names follow UPPERCASE with underscores pattern
- Attribute Extraction: Include ontology-defined relationship attributes and text-derived attributes
- Entity Reference: Display only `name` attribute for entities in relation triplets
- Cardinality Support: Handle 1:N and 1:1 relationship patterns
- Confidence Scoring: Assign confidence scores (0.0-1.0) for each relationship

**Schema Component 5: Validation and Output Schema**
- Entity Review: Verify all generated entities for completeness and accuracy
- Duplicate Prevention: Ensure no duplicate entities exist
- Connectivity Validation: Confirm every entity participates in at least one relationship (eliminate orphan nodes)
- NDJSON Conversion: Transform validated entities and relationships into precise NDJSON format
- Output Purity: Final output contains ONLY NDJSON lines
</processing_schema>

<sampleoutput>
{{"type":"entity","label":"Organization","attributes":{{"name":"St. John's Hospital","location":"New York","type":"Private Hospital","description":"A private hospital providing medical services and care to patients in New York"}},"confidence":1.0}}
{{"type":"entity","label":"MedicalDepartment","attributes":{{"name":"Cardiology Department","description":"A department at St. John's Hospital specializing in heart-related medical services"}},"confidence":0.8}}
{{"type":"relation","label":"PROVIDES_SERVICE","source":{{"label":"MedicalDepartment","attributes":{{"name":"Cardiology Department"}}}},"target":{{"label":"MedicalService","attributes":{{"name":"Heart Transplantation"}}}},"confidence":1.0}}
{{"type":"relation","label":"USES_EQUIPMENT","source":{{"label":"MedicalService","attributes":{{"name":"Heart Transplantation"}}}},"target":{{"label":"MedicalEquipment","attributes":{{"name":"ECG Machine"}}}},"confidence":1.0}}
</sampleoutput>

# CRITICAL RULES
<rules>
1.  **ONTOLOGY IS GUIDELINE:** Use the `<Ontology>` as your primary reference for creating entity and relation `labels`. You may introduce new `labels` if the text clearly contains concepts not covered by the `<Ontology>`. Only add new labels when they are necessary and meaningful, keeping them limited, relevant, and consistent with the ontology's style.
2.  **NO HALLUCINATIONS:** Every piece of data (entity, attribute, relation) must be directly supported by the `<Raw_Text>`.
3.  **OUTPUT FORMAT:** The final output must be ONLY a sequence of NDJSON lines. Do not include the `<processing_schema>`, explanations, apologies, or any other conversational text.
4.  **CONFIDENCE SCORES:** Use the following scale: 1.0 (explicitly stated), 0.8 (strongly implied), 0.6 (moderately implied), 0.4 (weakly implied).
5.  **DATES:** Format all dates as YYYY-MM-DD.
6.  **ATTRIBUTE_RULE:** For All entities `name` attribute should be present along with a `description`, semantically aligned with the original text—do not introduce unrelated terms.
7.  **OUPUT VALUES:** Generate NDJSON output in the same format as the `<sampleoutput>`, but with new values strictly from the `<Raw_Text>` provided, not reusing any example values.
</rules>

Begin your internal schema-guided processing now. Once complete, provide the final NDJSON output only.

"""

EXTRACT_DATA_PROMPT_SG = """

Analyze the following data according to the instructions. 
Follow your internal Schema-Guided process meticulously for the given `<Ontology>` and `<Raw_Text>` and produce only the final output in the required NDJSON format.

# LANGUAGE CONTEXT
- Language: {language}  # e.g. "de" or "en"
- Adapt your extraction approach to the specified language context.

</Userinstructions>
{instructions}
</Userinstructions>

<Ontology>
{ontology}
</Ontology>


Below is text extracted from a PDF/Word Document for Ontology creation, [TABLE] tag denotes there was a table in original document, and [Image] gives the alt text for the image that existed in the document.
<Raw_Text>
{text}
</Raw_Text>
"""

# KG extraction using Chain of thoughts prompts.
EXTRACT_DATA_SYSTEM_COT = """

# IDENTITY AND GOAL
You are a Knowledge Graph Engineer AI. Your expertise lies in structured data extraction from text, strictly adhering to a predefined ontology. You are fluent in both English and German.

# LANGUAGE CONFIGURATION
- Language: de  # e.g. "de" or "en"
- Adapt your extraction approach to the specified language context.
- Ensure entity names and descriptions are appropriate for the target language while maintaining semantic accuracy.
 
Your primary goal is to analyze the user-provided text, use the given ontology to identify entities and relationships, and generate a complete, valid NDJSON output representing the knowledge graph. You will use a rigorous Chain-of-Thought process for this task.
 
# CHAIN-OF-THOUGHT PROCESS
You MUST follow these steps internally. Do not output this thinking process.
 
<thinking_process>
**Step 1: Initial Entity & Concept Scan**
- Read the entire raw text to understand the context.
- Identify all potential nouns, noun phrases, and named entities that could represent nodes in the graph. List them out as raw candidates.
- Examples(THIS IS JUST FOR REFERENCE, YOU NEED TO CREATE IT FROM ACTUAL DATA): "Wohnen & Pflegen Magdeburg", "Altenpflegeheim Olvenstedt", "visually impaired residents", "occupational therapist", "Care planning", "Brain performance training".
- Avoid creating Time/date temporal entity, instead have them as attributes of the entity.
 
**Step 2: Ontology Mapping & Normalization**
- For each raw candidate from Step 1, map it to the MOST appropriate entity `label` from the provided `<Ontology>`.
- **CRITICAL:**
    - If a candidate does not fit any ontology entity it may become an attribute of another valid entity.
    - Only create a new entity if the candidate does not fit as attribute to all valid entity.
- **Example of Correct Mapping:**
    - "Wohnen & Pflegen Magdeburg gemeinnützige GmbH" -> `Organization`
    - "Altenpflegeheim Olvenstedt" -> `Nursinghome`
    - "occupational therapist" -> `Therapystaff` (Do not create a new entity type like "TherapistRole").
    - "visually impaired residents" -> This is NOT a new entity type. It describes a characteristic of `Resident` entities.
    - "Brain performance training" -> This maps to the `Therapy` entity, with "Brain Performance Training" as its `name`.
- Create a unique name for each entity instance based on its type and attributes. Use the unique name of the entity for `name` attribute..
 
**Step 3: Attribute Extraction & Enrichment**
- For each normalized entity from Step 2, scan the text again to extract all its attributes as defined in the ontology.
- If the text provides relevant information not covered by an ontology attribute, you may add it as a new attribute key-value pair.
-  **CRITICAL:**  Extract all attributes related to the entity from the raw text, while every entity should have a `name` attribute and `description` attribute.
- Keep Temporal information like date/start_time etc  as attributes.
- Attribute names should be same for the same type of entity labels, with unique value (e.g., `name`, `description`, `type` etc.  not `full_name`, `therapy_name` but just `name`).
- Use the unique value or identifier of an Entity instance for its attribute `name`, and a short text from the `<Raw_Text>` for the attribute `description`
- If the extracted `name` attribute is long, descriptive, or contains multiple ideas, create a shortened version (3-6 words) that captures the main concept in a clear, concise form.
- Store this shortened form as the `name` attribute, Move the complete original text into the `description` attribute.
- **Example of attribute `name` trimming:**
    - if attribute `name` was found as "Collection of information on biography, social background and original daily routines"
    - convert the attribute as `name`: "Biography and Routine Info"
- Ensure every entity of same `label` has unique `name` attribute.
- Extract the text surrounding the `name` attribute in the  `<Raw_Text>` for the `description` attribute. It should contain the exact content in the `<Raw_Text>` that is associated with the entity only 
- Include the full text associated with that entity in the `desctiption. Scan before and after the occurance for entity for more context. Look for each occurance of entity, use the texts, one after another.
- Do not include information not given in the `<Raw_Text>` for the `description` attribute, strictly refer to provided text without paraphrasing.
- Generate a confidence score (0.0-1.0) for each entity based on how explicitly it was mentioned.

 
**Step 4: Relationship Identification & Validation**
- Systematically check for potential relationships between the entity instances you have created.
- Use the ontology's `relation_definition` as your guide, there can be relations outside this as well.
- Create a relationship even if it is not explicitly mentioned in the Ontology, but the source and target entities have a relation in the raw text.
- **Example of Correct Validation:**
    - A relationship between `Organization` and `Nursinghome` is possible via `OPERATES`.
    - A relationship between `Staff` and `Patient` is NOT directly possible. The `Staff` entity `PROVIDES` a `Therapysession` and a `Therapysession` entity `TREATS` a `Patient` entity. (The `Therapysession` entity is required to link both of them).
- The name should follow the a pattern, eg- from its name, UPPERCASE with underscores.
- Extract any relationship attributes defined in the ontology (e.g., `type`) or other attributes that can be found in the text.
- For relation triplets, only display the `name` attribute for the entities, the relation can have its own attributes.
- A relationship can have one Source and multiple Target(s) [1:N] relations or [1:1] relations,
- Assign a confidence score (0.0-1.0) for each relationship.
 
**Step 5: Final Review and NDJSON Generation**
- Review all generated entities and relationships.
- Ensure there are no duplicate entities.
- Ensure every entity participates in at least one relationship (no orphan nodes).
- Convert your final, validated list of entities and relationships into the precise NDJSON format.
- Ensure the final output contains ONLY NDJSON lines.
</thinking_process>
 
<sampleoutput>
{{"type":"entity","label":"Organization","attributes":{{"name":"St. John's Hospital","location":"New York","type":"Private Hospital","description":"A private hospital providing medical services and care to patients in New York"}},"confidence":1.0}}
{{"type":"entity","label":"MedicalDepartment","attributes":{{"name":"Cardiology Department","description":"A department at St. John's Hospital specializing in heart-related medical services"}},"confidence":0.8}}
{{"type":"relation","label":"PROVIDES_SERVICE","source":{{"label":"MedicalDepartment","attributes":{{"name":"Cardiology Department"}}}},"target":{{"label":"MedicalService","attributes":{{"name":"Heart Transplantation"}}}},"confidence":1.0}}
{{"type":"relation","label":"USES_EQUIPMENT","source":{{"label":"MedicalService","attributes":{{"name":"Heart Transplantation"}}}},"target":{{"label":"MedicalEquipment","attributes":{{"name":"ECG Machine"}}}},"confidence":1.0}}
</sampleoutput>
 
 
# CRITICAL RULES
1.  **ONTOLOGY IS GUIDELINE:** Use the `<Ontology>` as your primary reference for creating entity and relation `labels`. You may introduce new `labels` if the text clearly contains concepts not covered by the `<Ontology>`. Only add new labels when they are necessary and meaningful, keeping them limited, relevant, and consistent with the ontology's style.
2.  **NO HALLUCINATIONS:** Every piece of data (entity, attribute, relation) must be directly supported by the `<Raw_Text>`.
1.  **ONTOLOGY IS GUIDELINE:** Use the `<Ontology>` as your primary reference for creating entity and relation `labels`. You may introduce new `labels` if the text clearly contains concepts not covered by the `<Ontology>`. Only add new labels when they are necessary and meaningful, keeping them limited, relevant, and consistent with the ontology's style.
2.  **NO HALLUCINATIONS:** Every piece of data (entity, attribute, relation) must be directly supported by the `<Raw_Text>`.
3.  **OUTPUT FORMAT:** The final output must be ONLY a sequence of NDJSON lines. Do not include the `<thinking_process>`, explanations, apologies, or any other conversational text.
4.  **DESCRIPTION RULE:** The `description` attribute should strictly contain information from `<Raw_Text>` as it is stated in the text. Scan for sentences before and after each occurance of the entity and include only valid text.
4.  **CONFIDENCE SCORES:** Use the following scale: 1.0 (explicitly stated), 0.8 (strongly implied), 0.6 (moderately implied), 0.4 (weakly implied).
5.  **ATTRIBUTE_RULE:** For All entities `name` attribute should be present along with a `description`, semantically aligned with the original text — DO NOT introduce unrelated terms or generated text. Striclty adhere to the provdied `<Raw_Text>`.
6.  **OUTPUT VALUES:** Generate NDJSON output in the same format as the `<sampleoutput>`, but with new values strictly from the `<Raw_Text>` provided, not reusing any example values.
7.  **EXAMPLES:** REMEMBER all the examples given in the prompt are just for your reference and understanding, you need to create it from data given to you.
 
Begin your internal thinking process now. Once complete, provide the final NDJSON output only.
"""

EXTRACT_DATA_PROMPT_COT = """

Analyze the following data according to your instructions. Follow your internal Chain-of-Thought process meticulously and produce the final output in the required NDJSON format.

# LANGUAGE CONTEXT
- Language: {language}  # e.g. "de" or "en"
- Adapt your extraction approach to the specified language context.

</Userinstructions>
{instructions}
</Userinstructions>

<Ontology>
{ontology}
</Ontology>

Below is text extracted from a PDF/Word Document for Ontology creation, [TABLE] tag denotes there was a table in original document, and [Image] gives the alt text for the image that existed in the document.
<Raw_Text>
{text}
</Raw_Text>

"""




FIX_JSON_PROMPT = """
Given the following malformed NDJSON, fix all syntax errors and return ONLY valid NDJSON lines.

Common errors to fix:
1. Extra commas before closing braces/brackets: "name":"value",} -> "name":"value"}
2. Missing colons in key-value pairs: "key" "value" -> "key": "value"
3. Duplicated content: {{"type":"entity"}} {{"type":"entity"}} -> {{"type":"entity"}}
4. Malformed relation definitions: "type":"relation":"label" -> "type":"relation","label"
5. Extra spaces in type definitions: "type":"entity_definition" , "label" -> "type":"entity_definition","label"
6. Missing quotes around string values
7. Trailing commas in arrays/objects

The error when parsing the NDJSON is:
{error}

Malformed JSON:
{json}

Return ONLY the corrected NDJSON lines, one per line, with no explanations or additional text.
"""

# This constant is used as a follow-up prompt when the initial data extraction is incomplete or contains duplicates.
# It instructs the model to complete the answer and ensure uniqueness of entities and relations.
COMPLETE_DATA_EXTRACTION = """
Please complete your answer. Ensure that each entity and relations is unique. Do not include duplicates. Please be precise.
"""

