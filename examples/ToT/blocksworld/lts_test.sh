#export CUDA_VISIBLE_DEVICES=0
export llama_path="your/path/to/llama3" # or "your/path/to/llama2"
export llama_size="8B" # or "7B" of llama2
export base_lm="openai" # or "llama2"
export VAL="/root/Search-LLM-inference/LLMs-Planning/planner_tools/VAL"
export OPENAI_API_KEY="sk-UZDt7u7q-4PVd5lmJGh8kjsrDQTARspr7xDReOvPuDT3BlbkFJiuZMmmFmtO5BvBnYc1EidjNTv4LybuQ1C_ZQbekJ8A"
export model_dir="meta-llama/Llama-3.2-3B-Instruct"

time_budgets=(4 8 12 16 20)
temps=(0.01 1 2)
for t in "${temps[@]}"; do
	for budget in "${time_budgets[@]}"; do
	    echo "$budget"
	    python examples/ToT/blocksworld/tot_inference.py \
		--base_lm hf \
		--data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json' \
		--model_dir "$model_dir" \
		--prompt_path examples/CoT/blocksworld/prompts/pool_prompt_v1.json \
		--log_dir "logs/lts_v1_step4_${budget}_${t}" \
		--temperature 0.8 \
		--search_algo lts \
		--total_calls 1000 \
		--max-time "$budget" \
		--depth_limit 4 \
		--lts_temp "$t" \
		--max_per_state 3 \
		--llama_size "$llama_size" \
		| tee "outputs/lts_step4_${budget}_${t}.log"
	done
done
