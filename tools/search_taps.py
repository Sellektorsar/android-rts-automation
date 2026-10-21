import re

with open(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print('Searching for stage5 or stage 5 clicks in transcript...')
for i, line in enumerate(lines):
    if 'stage5' in line.lower() or 'этап 5' in line.lower() or 'этап 6' in line.lower():
        # find taps
        taps = re.findall(r'tap\(\d+,\s*\d+\)', line)
        if taps:
            print(f'Line {i}: {taps}')
