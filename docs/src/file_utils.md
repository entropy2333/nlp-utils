# File utils

## JsonFactory

```python
from nlp_utils.file_utils import JsonFactory

# write and load json
JsonFactory.write_json([1, 2, 3], "a.json")
JsonFactory.load_json("a.json")

# write and load jsonlines
JsonFactory.write_jsonl([1, 2, 3], "a.jsonl")
JsonFactory.load_jsonl("a.jsonl")

# write and load jsonlines with gzip
JsonFactory.write_jsonl([1, 2, 3], "a.jsonl.gz", gzip=True)
JsonFactory.load_jsonl("a.jsonl.gz", gzip=True)

# or directly call api
JsonFactory.write_jsonl_gzip([1, 2, 3], "a.jsonl.gz")
JsonFactory.load_jsonl_gzip("a.jsonl.gz")
```