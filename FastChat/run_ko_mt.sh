cd fastchat/llm_judge/

# Generating model answers
CUDA_VISIBLE_DEVICES=0 python gen_model_answer.py \
		--model-path LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct \
		--model-id exaone-3.0-7.8b-instruct \
		--dtype bfloat16 


# Assessing the model answer through LLM-as-a-judge (here, "gpt-4-0613")
python gen_judgment.py \
    --model-list exaone-3.0-7.8b-instruct


# Giving a penalty to the score of non-Korean responses
cd data/mt_bench/model_judgment
python detector.py \
    --model_id exaone-3.0-7.8b-instruct


# Showing the evaluation results
cd ../../..
python show_result.py \
    --mode single \
    --input-file ./data/mt_bench/model_judgment/exaone-3.0-7.8b-instruct_single_final.jsonl