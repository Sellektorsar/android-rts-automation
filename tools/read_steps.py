import json

with open(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\.system_generated\logs\transcript_full.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data.get('step_index') in [4134, 4135, 4450, 4451]:
            print(f"Step {data.get('step_index')}:", data.get('tool_calls'))
