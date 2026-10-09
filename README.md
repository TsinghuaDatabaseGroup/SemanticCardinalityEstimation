# SemStats

SemStats estimates the cardinality of semantic predicates over a 
document corpus. It builds a semantic catalog and index, trains a lightweight judge, 
and estimates query cardinalities through stratified importance sampling.



## Requirements

- Python 3.10+ and `pip install -r requirements.txt` (`numpy`, `pyyaml`, `torch`, `xgboost`, `top2vec>=1.0.36`)
- Two OpenAI-compatible endpoints (e.g. vLLM): a chat model (`llm.base_url`) and an embedding model
  (`embedding.base_url`). Set their URLs and model names in configuration file. 


## Data

```
data/<dataset>/corpus/documents.jsonl        # {"doc_id": ..., "text": ...} per line
data/<dataset>/queries/{single,and,or}.json  # [{"query": ..., "predicates": [...]}]  (other fields are ignored)
data/<dataset>/evaluation/labels.jsonl       # {"doc_id": ..., "labels": [...]} per line (set-membership truth)
```

## Run

Supply a YAML configuration file containing your data paths, model endpoints, stage parameters, and dataset-specific prompt wording. 

```bash
python run_all.py --config "config path"
```

Add `--device cuda` to train on a GPU and `--minutes M` to change the labeling time budget (default 600 minutes).
Results are written to `workspace/<dataset>/runs/run1/estimates/eval/` (`metrics.json`, `per_query.json`, `report.txt`).

Stages can also be run one at a time with `run.py` (see its docstring):
`embed → build-catalog → build-index → gen-queries → label → snapshot → train → [calibrate] → estimate → evaluate`.

## Configuration

Configuration file holds the endpoints, data paths, stage parameters and a `dataset` section with the
wording used in the prompt templates. 



| Key | Values |
|---|---|
| `index.mode` | `single` (one node per document) or `multi` (multi-label teacher + per-node classifier) |
| `query_gen.fulltext.enabled` | also mine training predicates from full document text |
| `sample.and_weight` | `query` or `predicate_geomean` (sampling weights for AND queries) |
| `judge.and_judge` | `joint` or `per_predicate_calibrated` (AND verdicts; the latter runs `calibrate`) |



Configuration file example (the values below are configurable):

``` yaml
workspace:                 # Output directory

llm:
  base_url:                # Chat endpoint URL
  model:                   # Chat model name
  cache:                   # LLM response cache path

embedding:
  base_url:                # Embedding endpoint URL
  model:                   # Embedding model name
  query_vectors:           # Output path for query vectors

data:
  corpus:                  # Input documents.jsonl
  embeddings:              # Output document embeddings (.npy)
  # embedding_manifest:    # Optional output path

evaluation:
  queries:                 # Keep only the query sets you have
    single:                # Path to single-predicate queries
    and:                   # Path to AND queries
    or:                    # Path to OR queries
  labels:                  # Path to evaluation labels.jsonl

judge:
  chunk_chars: 40000
  # and_judge:              # Optional: joint | per_predicate_calibrated

select:
  votes: 1
  temperature: 0.0

sample:
  budget_ratio: 0.01
  seed: 42
  # and_weight:             # Optional: query | predicate_geomean

catalog:
  sample_ratio: 0.1
  min_sample_docs: 1000
  seed: 42
  top2vec:
    embedding_model:       # Top2Vec embedding model available in your environment
    chunk_chars: 1500
    min_count: 20
    min_cluster_size: 50
    min_samples: 10
    # smoothing_window: 5
    # topic_merge_delta: 0.1
    # num_topics: 100
  phrases_per_doc: 5
  max_root_phrases: 500
  max_phrase_len: 5
  stop_phrase_share: 0.1
  K_root: 16
  K: 5
  H: 3
  max_new_dims: 16
  min_freq_root: 0.01
  min_freq: 0.0025
  min_phrases: 8
  min_docs: 30
  min_assigned_share: 0.8
  merge_cos: 0.93
  tries: 3
  max_tokens_identify: 8192

index:
  mode:                    # Required: single | multi

  # Used when mode is single:
  sample_rate: 0.1
  root_boost: 2.0
  child_floor: 5
  floor_pool: 15
  tau: 0.5
  votes: 1
  seed: 42

  # Uncomment this section when mode is multi:
  # multi:
  #   sample_rate: 0.1
  #   root_boost: 2.0
  #   child_floor: 5
  #   floor_pool: 15
  #   xgb_trees: 200
  #   xgb_threshold: 0.5
  #   min_pos: 3
  #   min_neg: 3
  #   xgb_jobs: 8
  #   seed: 42

query_gen:
  seed: 2026
  doc_label:
    n_docs: 0
    batch: 20
    excerpt_chars: 500
    min_docs: 3
  catalog_axes:
    k_root: 20
    k_field: 10
    k_type: 10

  # fulltext:
  #   enabled: true
  #   min_docs: 3
  #   max_tokens: 1024

label:
  seed: 2026
  per_round: 15
  near_pool: 800
  chunk_pairs: 6000
  # seek:                   # Optional
  #   target: 10
  #   cap: 90
  #   min_rounds: 2

train:
  protocol: kfold          # kfold | split
  k: 5
  seed: 2026
  hidden: 1024
  layers: 2
  dropout: 0.1
  lr: 1.0e-4
  epochs: 80
  batch_size: 4096
  neg_weight: 0.333
  # val_frac: 0.15          # Required when protocol is split
  # device: cpu             # Optional; code default is cpu

dataset:
  name:                   # Short dataset identifier used in logs and metadata
  collection:             # Noun phrase for the corpus, completing "the category catalog of ..." (e.g. "a collection of ... posts")
  collection_kind:        # Kind of collection, completing "... sits in <whole_collection> of {collection_kind}" (e.g. "an encyclopedia")
  whole_collection:       # Phrase that stands in for the root node where a child node would read the "<node>" collection (e.g. "the encyclopedia's whole collection")
  catalog_owner:          # Possessive phrase preceding "classification catalog", (e.g. "an encyclopedia's")
  root_hint:              # Sentence appended to the dimension-identification prompt at the root, describing what top-level dimensions should look like
  unit:                   # Singular document noun, completing "one line per {unit}" in the document-labeling prompt
  doc_label_intro:        # Plural phrase completing "Below are N {doc_label_intro}."
  doc_label_task:         # Instructions for assigning three label tiers to each document
  tier_names:             # YAML list of exactly three label-tier names
  root_task:              # Instructions for root labels; may include {k}
  axes_task:              # Instructions for A/B axis labels; may include {node}, {k_field}, {k_type}
  judge_role:             # System-prompt role and dataset-specific judging context
  judge_unit:             # Phrase for one document, following "You will be shown"
  judge_ask_system:       # Phrase completing "asked whether {judge_ask_system} one or more specific <label_plural>" (e.g. "it would be filed under")
  judge_ask:              # Phrase before a quoted label in a single-predicate question (e.g. "Would this article be filed under")
  judge_ask_multi:        # Phrase before "AT LEAST ONE" or "ALL" in a multi-predicate question
  judge_and_rule:         # Extra rule for deciding whether all predicates apply
  judge_rule:             # Strict rule defining when a label does or does not apply
  label_noun:             # Singular noun for a predicate label
  label_plural:           # Plural noun for predicate labels

  # Required when index.mode is multi:
  # multi:
  #   collection:          # Description of the collection for the multi-label index teacher
  #   unit:                # Singular name for one document in that collection
  #   label_kind:          # What catalog category labels represent
  #   relevance:           # Rule for assigning a document to each category

  # Required when query_gen.fulltext.enabled is true:
  # fulltext:
  #   document:            # Phrase describing the document being read in full
  #   task:                # Instructions for extracting labels from its full text
  #   naming:              # Rules for naming and formatting extracted labels
  #   collection:          # Description of the collection for full-text extraction
  #   relevance:           # Rule for deciding which full-text labels apply
```
