from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from unsloth import is_bfloat16_supported
import verifiers as vf
from alfworld_grpo.tools import alfworld_tools
from alfworld_grpo.envs.alfworld_env import AlfworldEnv
from alfworld_grpo.utils.config_utils import get_default_grpo_config
import torch



model_name = "Qwen/Qwen2.5-7B-Instruct"
model_kwargs = dict(
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    use_cache=False,
    load_in_4bit=True
)

lora_rank = 64
max_seq_length = 512

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

vf_env = AlfworldEnv(
    dataset="alfworld",
    tools=alfworld_tools
)
trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    reward_funcs=vf_env.get_rubric(),
    args=get_default_grpo_config(run_name="alfworld", num_gpus=2, hub_repo_id=f'crislmfroes/AlfWorld-{model_name.split("/")[1]}'),
    train_dataset=vf_env.get_dataset(),
)
trainer.train()
trainer.push_to_hub()