#!/bin/bash

mkdir -p $HOME/.cache/huggingface
echo -n "$HF_TOKEN" > $HOME/.cache/huggingface/token

export MODEL_CONFIG_NAME=${MODEL_CONFIG_NAME:-"OA_SFT_Pythia_12B"}

num_shards=${NUM_SHARDS:-1}
load_sleep=${LOAD_SLEEP:-0}

export MODEL_ID=$(python /worker/get_model_config_prop.py model_id)

#quantize=${QUANTIZE:-"false"}
quantize_args=""
if [ "$quantize" = "true" ]; then
    quantize_args="--quantize"
fi

export HF_HUB_ENABLE_HF_TRANSFER=
export HF_HOME=$HOME/.cache/huggingface
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

echo "Downloading model $MODEL_ID"
python /worker/download_model.py

# Usage: text-generation-launcher <--model-id <MODEL_ID>|--revision <REVISION>|--validation-workers <VALIDATION_WORKERS>|--sharded <SHARDED>|
#       --num-shard <NUM_SHARD>|--quantize <QUANTIZE>|--speculate <SPECULATE>|--dtype <DTYPE>|--trust-remote-code|--max-concurrent-requests <MAX_CONCURRENT_REQUESTS>|
#       --max-best-of <MAX_BEST_OF>|--max-stop-sequences <MAX_STOP_SEQUENCES>|--max-top-n-tokens <MAX_TOP_N_TOKENS>|--max-input-length <MAX_INPUT_LENGTH>|
#       --max-total-tokens <MAX_TOTAL_TOKENS>|--waiting-served-ratio <WAITING_SERVED_RATIO>|--max-batch-prefill-tokens <MAX_BATCH_PREFILL_TOKENS>|
#       --max-batch-total-tokens <MAX_BATCH_TOTAL_TOKENS>|--max-waiting-tokens <MAX_WAITING_TOKENS>|--max-batch-size <MAX_BATCH_SIZE>|--enable-cuda-graphs|
#       --hostname <HOSTNAME>|--port <PORT>|--shard-uds-path <SHARD_UDS_PATH>|--master-addr <MASTER_ADDR>|--master-port <MASTER_PORT>|
#       --huggingface-hub-cache <HUGGINGFACE_HUB_CACHE>|--weights-cache-override <WEIGHTS_CACHE_OVERRIDE>|--disable-custom-kernels|--cuda-memory-fraction <CUDA_MEMORY_FRACTION>|
#       --rope-scaling <ROPE_SCALING>|--rope-factor <ROPE_FACTOR>|--json-output|--otlp-endpoint <OTLP_ENDPOINT>|--cors-allow-origin <CORS_ALLOW_ORIGIN>|
#       --watermark-gamma <WATERMARK_GAMMA>|--watermark-delta <WATERMARK_DELTA>|--ngrok|--ngrok-authtoken <NGROK_AUTHTOKEN>|--ngrok-edge <NGROK_EDGE>|
#       --tokenizer-config-path <TOKENIZER_CONFIG_PATH>|--disable-grammar-support|--env>
#
# Example:
# Args { model_id: "HuggingFaceH4/zephyr-7b-beta", revision: None, validation_workers: 2, sharded: None, num_shard: Some(1), quantize: Some(Awq), speculate: None, 
# dtype: None, trust_remote_code: false, max_concurrent_requests: 128, max_best_of: 2, max_stop_sequences: 4, max_top_n_tokens: 5, max_input_length: 4096, 
# max_total_tokens: 8192, waiting_served_ratio: 1.2, max_batch_prefill_tokens: 4096, max_batch_total_tokens: None, max_waiting_tokens: 20, max_batch_size: None, 
# enable_cuda_graphs: false, hostname: "c724784395be", port: 8300, shard_uds_path: "/tmp/text-generation-server", master_addr: "localhost", master_port: 29500, 
# huggingface_hub_cache: Some("/data"), weights_cache_override: None, disable_custom_kernels: false, cuda_memory_fraction: 1.0, rope_scaling: None, rope_factor: None, 
# json_output: false, otlp_endpoint: None, cors_allow_origin: [], watermark_gamma: None, watermark_delta: None, ngrok: false, ngrok_authtoken: None, ngrok_edge: None, 
# tokenizer_config_path: None, disable_grammar_support: false, env: false }

# if cuda devices is empty
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    worker_port=8300
    echo "Starting worker server on port $worker_port"
    text-generation-launcher --model-id $MODEL_ID --num-shard $num_shards $quantize_args --port $worker_port &
    export INFERENCE_SERVER_URL="http://localhost:$worker_port"
    echo "Starting worker"
    python /worker &

else
    # split cuda devices and loop over them
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    IFS=',' read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
    for i in "${!devices[@]}"; do
        device="${devices[$i]}"
        worker_port=$((8300 + $i))
        master_port=$((29500 + $i))
        shard_uds_path="/tmp/text-generation-server-$i"
        echo "Starting worker server $i on port $worker_port on device $device"
        #CUDA_VISIBLE_DEVICES=$device text-generation-launcher --model-id $MODEL_ID --num-shard $num_shards $quantize_args --port $worker_port --master-port $master_port --shard-uds-path $shard_uds_path &
        CUDA_VISIBLE_DEVICES=$device text-generation-launcher --model-id $MODEL_ID --num-shard $num_shards --port $worker_port --master-port $master_port --shard-uds-path $shard_uds_path &
        echo "Starting worker $i"
        CUDA_VISIBLE_DEVICES="" INFERENCE_SERVER_URL="http://localhost:$worker_port" python /worker &
        sleep $load_sleep
    done
fi

wait -n

exit $?
