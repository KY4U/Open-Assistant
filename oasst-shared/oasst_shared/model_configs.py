import pydantic


class ModelConfig(pydantic.BaseModel):
    model_id: str
    max_input_length: int = 512
    max_total_length: int = 1024
    quantized: bool = False
    # quantize_args: str = "Awq"  # possible values: awq, eetq, gptq, bitsandbytes, bitsandbytes-nf4, bitsandbytes-fp4
    trust_remote_code: bool = False

    @property
    def is_llama(self) -> bool:
        return "llama" in self.model_id.lower()

    @property
    def is_lorem(self) -> bool:
        return self.model_id == "_lorem"

    @property
    def compat_hash(self) -> str:
        return f"{self.model_id}-{self.max_total_length}-{self.max_input_length}-{'q' if self.quantized else 'f'}"



"""     
    "_lorem": ModelConfig(
        model_id="_lorem",
        max_input_length=128,
        max_total_length=256,
    ),
    "distilgpt2": ModelConfig(
        model_id="distilgpt2",
        max_input_length=512,
        max_total_length=1024,
    ),
    "OA_SFT_Pythia_12B": ModelConfig(
        model_id="OpenAssistant/oasst-sft-1-pythia-12b",
        max_input_length=1024,
        max_total_length=2048,
        # This is the first iteration English supervised-fine-tuning (SFT) model of the Open-Assistant project. 
        # Seems to work well with no gpu.
        # Fails with CUDA_VISIBLE_DEVICES: 0,1
    ),
    "OA_SFT_Pythia_12Bq": ModelConfig(
        model_id="OpenAssistant/oasst-sft-1-pythia-12b",
        max_input_length=1024,
        max_total_length=2048,
        quantized=True,
        # ValueError: quantization is not available on CPU
    ),   
    "OA_SFT_Pythia_12Bq_4": ModelConfig(
        model_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
        max_input_length=1024,
        max_total_length=2048,
        quantized=True,
        # ValueError: quantization is not available on CPU
    ),
  
    "OA_SFT_Llama2_70B_10": ModelConfig(
        model_id="OpenAssistant/llama2-70b-oasst-sft-v10",
        max_input_length=3072,
        max_total_length=4096,
        # Requires model_prompt_format: str = "chatml" in inference/worker/settings.py
        # Downloading model Unknown model config name: OA_SFT_Llama2_70B_10
    ),
 
    "OA_SFT_CodeLlama_13B_10": ModelConfig(
        model_id="OpenAssistant/codellama-13b-oasst-sft-v10",
        max_input_length=8192,
        max_total_length=12288,
        trust_remote_code = True
        # Requires model_prompt_format: str = "chatml" in inference/worker/settings.py
        # Bitsandbytes error
    ),

    # From here are models in Open-Assistant on Huggingface
    "OA_codellama_13b_oasst_sft_v10": ModelConfig(
        model_id="OpenAssistant/codellama-13b-oasst-sft-v10",
        max_input_length=1024,
        max_total_length=2048,
        # Requires model_prompt_format: str = "chatml" in inference/worker/settings.py
        # This model is an Open-Assistant fine-tuning of Meta's CodeLlama 13B LLM.
        # raise AttributeError(f"module {self.__name__} has no attribute {name}")
    ),
    "OA_llama2_70b_oasst_sft_v10": ModelConfig(
        model_id="OpenAssistant/llama2-70b-oasst-sft-v10",
        max_input_length=1024,
        max_total_length=2048,
        # Requires model_prompt_format: str = "chatml" in inference/worker/settings.py
        # This model is an Open-Assistant fine-tuning of Meta's Llama2 70B LLM.
        # raise AttributeError(f"module {self.__name__} has no attribute {name}")
    ),
    "OA_llama2_13b_megacode2": ModelConfig(
        model_id="OpenAssistant/llama2-13b-megacode2-oasst",
        max_input_length=1024,
        max_total_length=2048,
        # Requires model_prompt_format: str = "chatml" in inference/worker/settings.py
        # This model is an Open-Assistant fine-tuning of Meta's Llama2 70B LLM.
        # raise AttributeError(f"module {self.__name__} has no attribute {name}")
    ),
    "OA_falcon_40b_megacode2": ModelConfig(
        model_id="OpenAssistant/falcon-40b-megacode2-oasst",
        max_input_length=1024,
        max_total_length=2048,
        # Requires model_prompt_format: str = "chatml" in inference/worker/settings.py
        # This model is an Open-Assistant fine-tuning of Meta's Llama2 70B LLM.
        # raise AttributeError(f"module {self.__name__} has no attribute {name}")
    ), 
    meta-llama/Llama-2-7b-chat

    "OA_SFT_Pythia_12B_4": ModelConfig(
        model_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
        max_input_length=1024,
        max_total_length=2048,
        # This is the 4th iteration English supervised-fine-tuning (SFT) model of the Open-Assistant project.
    ),

    """    
MODEL_CONFIGS = {
    "_lorem": ModelConfig(
        model_id="_lorem",
        max_input_length=128,
        max_total_length=256,
    ),
    # "distilgpt2": ModelConfig(
    #     model_id="distilgpt2",
    #     max_input_length=512,
    #     max_total_length=1024,
    # ),
    # "OA_SFT_Pythia_12B": ModelConfig(
    #     model_id="OpenAssistant/oasst-sft-1-pythia-12b",
    #     max_input_length=1024,
    #     max_total_length=2048,
    #     # This is the first iteration English supervised-fine-tuning (SFT) model of the Open-Assistant project. 
    #     # Seems to work well with no gpu.
    #     # torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 200.00 MiB. GPU 
    # ),
    # "OA_SFT_Pythia_12B_4": ModelConfig(
    #     model_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
    #     max_input_length=1024,
    #     max_total_length=2048,
    #     trust_remote_code=True,
    #     # This is the 4th iteration English supervised-fine-tuning (SFT) model of the Open-Assistant project.
    #     # torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 200.00 MiB. GPU 
    # ),
    # "OA_SFT_Pythia_12Bq": ModelConfig(
    #     model_id="OpenAssistant/oasst-sft-1-pythia-12b",
    #     max_input_length=1024,
    #     max_total_length=2048,
    #     quantized=True,
    #     # torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 200.00 MiB. GPU 
    # ),   
    "Zephyr_7b_beta": ModelConfig(
        model_id="HuggingFaceH4/zephyr-7b-beta",
        max_input_length=512,   # It's actually 514 but I'm putting in a small buffer.
        max_total_length=8192,
        trust_remote_code=True,
        # Requires model_prompt_format: str = "chatHF" in inference/worker/settings.py
        # Works great with GPU.
    ),
    # "Starchat2_15b_v01": ModelConfig(
    #     model_id="HuggingFaceH4/starchat2-15b-v0.1",
    #     max_input_length=1024,
    #     max_total_length=2048,
    #     trust_remote_code=True,
    #     # Requires model_prompt_format: str = "chatHF" in inference/worker/settings.py
    #     # torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU
    # ),
    # "latest_model_test": ModelConfig(
    #     model_id="Qwen/Qwen2-1.5B",
    #     max_input_length=512,
    #     max_total_length=8192,
    #     trust_remote_code=True,
    #     # Requires model_prompt_format: str = "chatml" in inference/worker/settings.py
    #     # Very fast with GPU
    # ),
}
