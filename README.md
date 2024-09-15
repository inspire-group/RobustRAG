[Certifiably Robust RAG against Retrieval Corruption](https://arxiv.org/abs/2405.15556)

This project is under active development. There might be some (small) mismatches between this repository and the arXiv preprint.

## Files

```shell
├── README.md                        # this file 
| 
├── main.py                          # entry point.  
├── llm_eval.py                      # LLM-as-a-judge for long-form evaluation

| 
├── src
|   ├── dataset_utils.py              # tool for dataset -- load data; clean data; eval response
|   ├── model.py                      # LLM wrapper -- query; batched query; wrap_prompt 
|   ├── prompt_template.py            # prompt template
|   ├── defense.py                    # defense class
|   ├── attack.py                     # attack algorithm 
|   ├── helper.py                     # misc utils 
|
| 
├── data   
|   ├── realtimeqa.json               # a subset of realtimeqa
|   ├── open_nq.json                  # a (random) subset of the open nq dataset (we only use its first 100 queries)
|   ├── biogen.json                   # a subset of the biogen dataset
|   └── ...                 

```
## Dependency

Tested with `torch==2.2.1` and `transformers==4.40.1`. This repository should be compatible with newer version of packages. `requirements.txt` lists other required packages (with version numbers commented out).


## Notes
1. add your OpenAI keys to `models.py` if you want to run GPT as underlying models; add keys to `llm_eval.py` if you want to run LLM-as-a-judge. 
(Alternative: running `export OPENAI_API_KEY="YOUR-API-KEY"` in command line)
2. the `--eta` argument in this repo corresponds to $k\cdot\eta$ discussed in the paper.
3. `llm_eval.py` might crash occasionally due to GPT's randomness (not following the LLM-as-a-judge output format). Haven't implemented error handling logic. As a workaround, delete any generated files and rerun `llm_eval.py`.

## Usage
```
python main.py 
--model_name: mistral7b,llama7b,gpt3.5
--dataset_name: realtimeqa, realtimeqa-mc, open_nq, biogen
--top_k: 0, 5, 10, 20, etc.
--attack_method: none, Poison, PIA
--defense_method: none, voting, keyword, decoding
--alpha
--beta
--eta # NOTE!! the eta in this code is actually k\cdot\eta in the paper
--corruption_size
--subsample_iter: only used for some settings in biogen certification

--debug: add this flag to print some extra info for debugging
--save_response: add this flag to save the results(responses) for later analysis (currently more useful to bio_gen task)
--use_cache: add this flag to cache the results(responses) to avoid duplicate running 
```
see `run.sh` for commands to reproduce results in the main body of the paper.