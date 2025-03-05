# %%
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# %%
model_name = "meta-llama/Llama-3.2-3B"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    # bnb_4bit_compute_dtype=torch.float16,  # Ensure computation type matches input type
    bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for better performance
    bnb_4bit_use_double_quant=True,  # Double quantization for memory efficiency
)
loaded_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    # device_map=torch.device("cuda"),
    device_map="auto"
)
# loaded_model = loaded_model.half()
# loaded_tokenizer = AutoTokenizer.from_pretrained(finetuned_models[model])
# tokenizer_name = finetuned_models[model]
