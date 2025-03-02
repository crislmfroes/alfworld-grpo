from verifiers.rubrics.tool_rubric import ToolRubric
import json
import difflib

class AlfworldRubric(ToolRubric):
    def __init__(self):
        super().__init__()
        self.reward_funcs = [
            self.success_reward_func,
            self.difflib_task_score_reward_func,
            self.xml_reward_func,
            self.format_reward_func,
            self.tool_execution_reward_func
        ]

    def difflib_task_score_reward_func(self, completions: list[list[dict[str, str]]], **kwargs)->list[float]:
        rewards = []
        for c in completions:
            reset_idxs = []
            trajectories = []
            reward = 0.0
            for idx, msg in enumerate(c):
                if msg['role'] == 'assistant':
                    parsed = self.parser.parse(msg['content'])
                    if hasattr(parsed, 'tool') and parsed.tool is not None:
                        try:
                            data = json.loads(parsed.tool)
                        except:
                            continue
                    if 'name' not in data.keys():
                        continue
                    if data['name'] != 'reset':
                        continue
                    reset_idxs.append(idx)
            for i in range(len(reset_idxs)):
                current_reset = reset_idxs[i]
                if i < len(reset_idxs) - 1:
                    next_reset = reset_idxs[i+1]
                else:
                    next_reset = None
                trajectories.append(c[current_reset:next_reset])
            for trajectory in trajectories:
                for msg in trajectory:
                    if msg['role'] == 'user' and 'Error:' not in msg['content']:
                        initial_obs = msg['content'].split('<result>')[1].split('</result>')[0]
                        task = initial_obs.split('Your task is to:')[1].split('\n')[0].strip()
                    elif msg['role'] == 'user':
                        obs = msg['content'].split('<result>')[1].split('</result>')[0]
                        matcher = difflib.SequenceMatcher(None, task, obs)
                        reward += matcher.ratio()
            reward = reward / (len(c)//2)
            rewards.append(reward)
        return rewards

    def success_reward_func(self, completions: list[list[dict[str, str]]], **kwargs)->list[float]:
        rewards = []
        for c in completions:
            reward = 0.0
            for msg in c:
                if msg['role'] == 'user' and 'SUCCESS!' in msg['content']:
                    reward += 1.0
                    break
            rewards.append(reward)
        return rewards