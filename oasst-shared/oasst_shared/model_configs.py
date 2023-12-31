import pydantic


class ModelConfig(pydantic.BaseModel):
    model_id: str
    max_input_length: int = 512
    max_total_length: int = 1024
    quantized: bool = False
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


MODEL_CONFIGS = {
    "OA_SFT_Pythia_12B_4": ModelConfig(
        model_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
        max_input_length=1024,
        max_total_length=2048,
        trust_remote_code = True
    ),
}
