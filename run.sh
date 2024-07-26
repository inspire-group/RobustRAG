# Table 1
# no RAG and vanilla RAG
#python main.py --model_name mistral7b --dataset_name realtimeqa-mc --top_k 0 --defense_method none --corruption_size 0 
#python main.py --model_name mistral7b --dataset_name realtimeqa-mc --top_k 10 --defense_method none --corruption_size 0 
#python main.py --model_name llama7b --dataset_name realtimeqa-mc --top_k 0 --defense_method none --corruption_size 0 
#python main.py --model_name llama7b --dataset_name realtimeqa-mc --top_k 10 --defense_method none --corruption_size 0 
#python main.py --model_name gpt3.5 --dataset_name realtimeqa-mc --top_k 0 --defense_method none --corruption_size 0 
#python main.py --model_name gpt3.5 --dataset_name realtimeqa-mc --top_k 10 --defense_method none --corruption_size 0 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 0 --defense_method none --corruption_size 0 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 10 --defense_method none --corruption_size 0 
#python main.py --model_name llama7b --dataset_name realtimeqa --top_k 0 --defense_method none --corruption_size 0 
#python main.py --model_name llama7b --dataset_name realtimeqa --top_k 10 --defense_method none --corruption_size 0 
#python main.py --model_name gpt3.5 --dataset_name realtimeqa --top_k 0 --defense_method none --corruption_size 0 
#python main.py --model_name gpt3.5 --dataset_name realtimeqa --top_k 10 --defense_method none --corruption_size 0 
#python main.py --model_name mistral7b --dataset_name open_nq --top_k 0 --defense_method none --corruption_size 0 
#python main.py --model_name mistral7b --dataset_name open_nq --top_k 10 --defense_method none --corruption_size 0 
#python main.py --model_name llama7b --dataset_name open_nq --top_k 0 --defense_method none --corruption_size 0 
#python main.py --model_name llama7b --dataset_name open_nq --top_k 10 --defense_method none --corruption_size 0 
#python main.py --model_name gpt3.5 --dataset_name open_nq --top_k 0 --defense_method none --corruption_size 0 
#python main.py --model_name gpt3.5 --dataset_name open_nq --top_k 10 --defense_method none --corruption_size 0 

# Voting for RealtimeQA-MC (multiple choice)
#python main.py --model_name mistral7b --dataset_name realtimeqa-mc --top_k 10 --defense_method voting --corruption_size 1 --no_vanilla 
#python main.py --model_name llama7b --dataset_name realtimeqa-mc --top_k 10 --defense_method voting --corruption_size 1 --no_vanilla 
#python main.py --model_name gpt3.5 --dataset_name realtimeqa-mc --top_k 10 --defense_method voting --corruption_size 1 --no_vanilla 

# decoding for RealtimeQA and NQ
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 10 --defense_method decoding --corruption_size 1 --eta 0.0 --no_vanilla 
#python main.py --model_name llama7b --dataset_name realtimeqa --top_k 10 --defense_method decoding --corruption_size 1 --eta 0.0 --no_vanilla 
#python main.py --model_name mistral7b --dataset_name open_nq --top_k 10 --defense_method decoding --corruption_size 1 --eta 0.0 --no_vanilla 
#python main.py --model_name llama7b --dataset_name open_nq --top_k 10 --defense_method decoding --corruption_size 1 --eta 0.0 --no_vanilla 

# keyword for RealtimeQA and NQ
##python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 1 --alpha 0.3 --beta 3 --no_vanilla 
#python main.py --model_name llama7b --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 1 --alpha 0.3 --beta 3 --no_vanilla 
#python main.py --model_name gpt3.5 --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 1 --alpha 0.3 --beta 3 --no_vanilla 
#python main.py --model_name mistral7b --dataset_name open_nq --top_k 10 --defense_method keyword --corruption_size 1 --alpha 0.3 --beta 3 --no_vanilla 
#python main.py --model_name llama7b --dataset_name open_nq --top_k 10 --defense_method keyword --corruption_size 1 --alpha 0.3 --beta 3 --no_vanilla 
#python main.py --model_name gpt3.5 --dataset_name open_nq --top_k 10 --defense_method keyword --corruption_size 1 --alpha 0.3 --beta 3 --no_vanilla 

# biogen keyword
#python main.py --model_name mistral7b --dataset_name biogen --top_k 10 --defense_method keyword --alpha 0.4 --beta 4 --save_response 
#python llm_eval.py --model_name mistral7b --dataset_name biogen --top_k 10 --defense_method keyword --alpha 0.4 --beta 4 --type pred
#python llm_eval.py --model_name mistral7b --dataset_name biogen --top_k 10 --defense_method keyword --alpha 0.4 --beta 4 --type certify

#python main.py --model_name llama7b --dataset_name biogen --top_k 10 --defense_method keyword --alpha 0.4 --beta 4 --save_response 
#python llm_eval.py --model_name llama7b --dataset_name biogen --top_k 10 --defense_method keyword --alpha 0.4 --beta 4 --type pred
#python llm_eval.py --model_name llama7b --dataset_name biogen --top_k 10 --defense_method keyword --alpha 0.4 --beta 4 --type certify

# biogen decoding eta= 4
#python main.py --model_name mistral7b --dataset_name biogen --top_k 10 --defense_method decoding --eta 4  --save_response
#python llm_eval.py --model_name mistral7b --dataset_name biogen --top_k 10 --defense_method decoding --eta 4  --type pred
python llm_eval.py --model_name mistral7b --dataset_name biogen --top_k 10 --defense_method decoding --eta 4  --type certify

#python main.py --model_name llama7b --dataset_name biogen --top_k 10 --defense_method decoding --eta 4 --save_response
#python llm_eval.py --model_name llama7b --dataset_name biogen --top_k 10 --defense_method decoding --eta 4 --type pred
python llm_eval.py --model_name llama7b --dataset_name biogen --top_k 10 --defense_method decoding --eta 4 --type certify

# biogen decoding eta=1 with subsampling
#python main.py --model_name mistral7b --dataset_name biogen --top_k 10 --defense_method decoding --eta 1 --subsample_iter 100 --save_response
#python llm_eval.py --model_name mistral7b --dataset_name biogen --top_k 10 --defense_method decoding --eta 1 --subsample_iter 100 --type pred
python llm_eval.py --model_name mistral7b --dataset_name biogen --top_k 10 --defense_method decoding --eta 1 --subsample_iter 100 --type certify

#python main.py --model_name llama7b --dataset_name biogen --top_k 10 --defense_method decoding --eta 1 --subsample_iter 100 --save_response
#python llm_eval.py --model_name llama7b --dataset_name biogen --top_k 10 --defense_method decoding --eta 1 --subsample_iter 100 --type pred
#python llm_eval.py --model_name llama7b --dataset_name biogen --top_k 10 --defense_method decoding --eta 1 --subsample_iter 100 --type certify


# Figure 3 (top k analysis)
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 2 --defense_method keyword --corruption_size 1 --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 4 --defense_method keyword --corruption_size 1 --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 6 --defense_method keyword --corruption_size 1 --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 8 --defense_method keyword --corruption_size 1 --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 1 --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 12 --defense_method keyword --corruption_size 1 --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 14 --defense_method keyword --corruption_size 1 --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 16 --defense_method keyword --corruption_size 1 --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 18 --defense_method keyword --corruption_size 1 --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 20 --defense_method keyword --corruption_size 1 --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 2 --defense_method decoding --corruption_size 1 --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 4 --defense_method decoding --corruption_size 1 --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 6 --defense_method decoding --corruption_size 1 --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 8 --defense_method decoding --corruption_size 1 --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 10 --defense_method decoding --corruption_size 1 --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 12 --defense_method decoding --corruption_size 1 --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 14 --defense_method decoding --corruption_size 1 --no_vanilla --use_cache 
##python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 16 --defense_method decoding --corruption_size 1 --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 18 --defense_method decoding --corruption_size 1 --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 20 --defense_method decoding --corruption_size 1 --no_vanilla --use_cache 

# Figure 4 (corruption size analysis)
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 1 --no_vanilla --use_cache --alpha 0.3 --beta 3 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 2 --no_vanilla --use_cache --alpha 0.3 --beta 3 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 3 --no_vanilla --use_cache --alpha 0.3 --beta 3 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 4 --no_vanilla --use_cache --alpha 0.3 --beta 3 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 5 --no_vanilla --use_cache --alpha 0.3 --beta 3 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 1 --no_vanilla --use_cache --alpha 0.4 --beta 4 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 2 --no_vanilla --use_cache --alpha 0.4 --beta 4 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 3 --no_vanilla --use_cache --alpha 0.4 --beta 4 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 4 --no_vanilla --use_cache --alpha 0.4 --beta 4 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 5 --no_vanilla --use_cache --alpha 0.4 --beta 4 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 1 --no_vanilla --use_cache --alpha 0.5 --beta 5 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 2 --no_vanilla --use_cache --alpha 0.5 --beta 5 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 3 --no_vanilla --use_cache --alpha 0.5 --beta 5 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 4 --no_vanilla --use_cache --alpha 0.5 --beta 5 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 5 --no_vanilla --use_cache --alpha 0.5 --beta 5

# Table 2 (empirical attack) 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 10 --defense_method decoding --corruption_size 1 --eta 0.0 --attack_method PIA 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 10 --defense_method decoding --corruption_size 1 --eta 0.0 --attack_method Poison 
#python main.py --model_name llama7b --dataset_name realtimeqa --top_k 10 --defense_method decoding --corruption_size 1 --eta 0.0 --attack_method PIA 
#python main.py --model_name llama7b --dataset_name realtimeqa --top_k 10 --defense_method decoding --corruption_size 1 --eta 0.0 --attack_method Poison 
#python main.py --model_name mistral7b --dataset_name open_nq --top_k 10 --defense_method decoding --corruption_size 1 --eta 0.0 --attack_method PIA 
#python main.py --model_name mistral7b --dataset_name open_nq --top_k 10 --defense_method decoding --corruption_size 1 --eta 0.0 --attack_method Poison 
#python main.py --model_name llama7b --dataset_name open_nq --top_k 10 --defense_method decoding --corruption_size 1 --eta 0.0 --attack_method PIA 
#python main.py --model_name llama7b --dataset_name open_nq --top_k 10 --defense_method decoding --corruption_size 1 --eta 0.0 --attack_method Poison 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 1 --alpha 0.3 --beta 3 --attack_method PIA 
#python main.py --model_name mistral7b --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 1 --alpha 0.3 --beta 3 --attack_method Poison 
#python main.py --model_name llama7b --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 1 --alpha 0.3 --beta 3 --attack_method PIA 
#python main.py --model_name llama7b --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 1 --alpha 0.3 --beta 3 --attack_method Poison 
#python main.py --model_name gpt3.5 --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 1 --alpha 0.3 --beta 3 --attack_method PIA 
#python main.py --model_name gpt3.5 --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 1 --alpha 0.3 --beta 3 --attack_method Poison 
#python main.py --model_name mistral7b --dataset_name open_nq --top_k 10 --defense_method keyword --corruption_size 1 --alpha 0.3 --beta 3 --attack_method PIA 
#python main.py --model_name mistral7b --dataset_name open_nq --top_k 10 --defense_method keyword --corruption_size 1 --alpha 0.3 --beta 3 --attack_method Poison 
#python main.py --model_name llama7b --dataset_name open_nq --top_k 10 --defense_method keyword --corruption_size 1 --alpha 0.3 --beta 3 --attack_method PIA 
#python main.py --model_name llama7b --dataset_name open_nq --top_k 10 --defense_method keyword --corruption_size 1 --alpha 0.3 --beta 3 --attack_method Poison 
#python main.py --model_name gpt3.5 --dataset_name open_nq --top_k 10 --defense_method keyword --corruption_size 1 --alpha 0.3 --beta 3 --attack_method PIA 
#python main.py --model_name gpt3.5 --dataset_name open_nq --top_k 10 --defense_method keyword --corruption_size 1 --alpha 0.3 --beta 3 --attack_method Poison 


# biogen with more parameters (Figure 4)
#python main.py --model_name mistral7b --dataset_name biogen --top_k 10 --defense_method decoding --eta 5 --save_response 
#python main.py --model_name mistral7b --dataset_name biogen --top_k 10 --defense_method decoding --eta 6 --save_response 
# omitting similar usage for eval.py