`google_search.py`: script for calling search API for passage retrieval. Warning: not tested. only extract snippets instead of the all text from the webpage (sufficient for QA tasks).

### JSON field for each data item
- `"question"`: the RAG query
- `"correct answer"`: correct answer from the original dataset
- `"expanded answer"`: we also use GPT-4 to expand the answer set. You can choose to use it or not by setting the `add_expanded_answer` flag in the `process_data_item()` function in `src/dataset_utils.py`
- `"context"`: retrieved passages (via Google Search).
- `"incorrect answer"`: the target incorrect answer for empirical attacks (used for empirical robustness evaluation only)
- `"incorrect context"`: the malicious passages used for poison-based empirical attacks.
- other fields are not really used (inherited from the original dataset)
