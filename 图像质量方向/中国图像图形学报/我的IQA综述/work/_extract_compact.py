import json
from collections import Counter

lib = json.load(open(r'work/final_library.json', encoding='utf-8'))
papers = sorted(lib['papers'], key=lambda p: p['final_id'])
lines = []
for p in papers:
    kf = (p.get('key_finding') or '')[:130].replace('\n', ' ')
    mn = (p.get('methods_notes') or '')[:90].replace('\n', ' ')
    themes = ','.join(p.get('themes', []))
    lines.append('{} | {} | {} | Q={} | {} | {} | {}'.format(
        p['final_id'], p['citation_key'], p['year'], p['source_quality'],
        p.get('fr_type', ''), p.get('role', ''), themes))
    lines.append('    F: ' + kf)
    if mn.strip() and mn.strip() != '—':
        lines.append('    M: ' + mn)

open(r'work/_library_compact.txt', 'w', encoding='utf-8').write('\n'.join(lines))
print('written lines:', len(lines), 'papers:', len(papers))
print('quality dist:', Counter(p['source_quality'] for p in papers))
print('year>=2023:', sum(1 for p in papers if p['year'] >= 2023))
print('branches:', Counter(b for p in papers for b in p.get('branches', [])))
print('read_status:', Counter(p.get('read_status', '?') for p in papers))
