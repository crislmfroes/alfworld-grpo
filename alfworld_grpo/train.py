import verifiers as vf
from alfworld_grpo.tools import alfworld_tools
from alfworld_grpo.envs.alfworld_env import AlfworldEnv
from alfworld_grpo.utils.config_utils import get_default_grpo_config
import torch
from peft import LoraConfig, get_peft_model


model_name = "Qwen/Qwen2.5-7B-Instruct"
model_kwargs = dict(
    torch_dtype=torch.bfloat16,
    attn_implementation="xformers",
    use_cache=False,
    load_in_4bit=True
)

model, tokenizer = vf.get_model_and_tokenizer(model_name, model_kwargs=model_kwargs)

peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj"]
)

model = get_peft_model(model, peft_config)

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
    peft_config=peft_config
)
trainer.train()
#model.push_to_hub_merged(f"crislmfroes/Alfworld-{model_name}", tokenizer, save_method = "merged_16bit")
trainer.push_to_hub()