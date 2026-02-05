# Local reranker model store

This directory is mounted into the `memmachine` container at `/models`.

## Cross-encoder reranker

To avoid downloading from Hugging Face at startup, place the SentenceTransformers
cross-encoder model files here:

```
models/
  cross-encoder/
    qnli-electra-base/
      config.json
      tokenizer.json
      tokenizer_config.json
      special_tokens_map.json
      vocab.txt / merges.txt / ...
      pytorch_model.bin or model.safetensors
      modules.json
      ...
```

Then set in `configuration.yml`:

- `resources.rerankers.ce_ranker_id.config.model_name: "/models/cross-encoder/qnli-electra-base"`

## Tip: pre-download on a machine with internet

You can use Python to download into a local folder:

```python
from sentence_transformers import CrossEncoder
CrossEncoder('cross-encoder/qnli-electra-base').save('/path/to/models/cross-encoder/qnli-electra-base')
```

Copy that folder into this `models/` directory.
