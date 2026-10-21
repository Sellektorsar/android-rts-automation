import re

with open(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if 'выбор навыка' in line.lower() or 'bbl6op habblka' in line.lower():
            coords = re.findall(r'\(\s*\d+,\s*\d+\)', line)
            print(line[:250], 'coords:', coords[:5])
