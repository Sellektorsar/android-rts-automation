import re

with open(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if 'казарм' in line.lower() or 'barrack' in line.lower():
            taps = re.findall(r'tap\(\d+,\s*\d+\)', line)
            if taps:
                print(line[:120], taps)
