import verifiers as vf
from alfworld_grpo.tools import alfworld_tools
from alfworld_grpo.envs.alfworld_env import AlfworldEnv
from alfworld_grpo.utils.config_utils import get_default_grpo_config
import torch

model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
model_kwargs = dict(
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    use_cache=False,
    load_in_4bit=True
)
model, tokenizer = vf.get_model_and_tokenizer(model_name=model_name, model_kwargs=model_kwargs)

vf_env = AlfworldEnv(
    dataset="alfworld",
    tools=alfworld_tools
)
trainer = vf.GRPOEnvTrainer(
    model=model,
    env=vf_env,
    reward_funcs=vf_env.get_rubric(),
    args=get_default_grpo_config(run_name="alfworld", num_gpus=1, hub_repo_id=f'crislmfroes/AlfWorld-{model_name.split("/")[1]}'),
    train_dataset=vf_env.get_dataset()
)
trainer.train()
trainer.push_to_hub()