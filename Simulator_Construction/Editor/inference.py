from openai import OpenAI
import json
import time
with open("../prompts.json", "r", encoding="utf-8") as f_prompts:
    prompts = json.load(f_prompts)
    editor_prompts = prompts["Editor"]

edit_client = OpenAI(api_key="0",base_url="http://localhost:8001/v1")

class EditorAgent:
    def __init__(self, new_body, new_head):
        self.new_body = new_body
        self.new_head = new_head
    def get_prompt(self):
        instruction = editor_prompts["train_prompt"]["user"]
        instruction = instruction.format(new_body=self.new_body,new_head=self.new_head)
        system = editor_prompts["train_prompt"]["system"]
        return system, instruction

    def output(self):
        for attempt in range(5):
            try:
                system, instruction = self.get_prompt()
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": instruction},
                ]
                result = edit_client.chat.completions.create(messages=messages, model="Qwen/Qwen3-4B")
                pred_edit = result.choices[0].message.content
                res = json.loads(pred_edit)
                pred_editor_decision = res["Decision"]
                pred_editor_feedback = res["Analysis"]
                return pred_editor_decision, pred_editor_feedback
            except:
                time.sleep(30)
                print(f"Attempt {attempt + 1} timed out. Retrying...")
        return "None", "None"

