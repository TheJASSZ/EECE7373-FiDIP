#!/usr/bin/env python3
"""Re-verify every DOI in refs.bib against the Crossref API (title, author family names, year, pages).

Usage: python3 report/check_bib.py report/refs.bib
Entries without a DOI are listed so they can be checked against their URL by hand.
"""
import json, re, sys, time, unicodedata, urllib.request, difflib

def parse_bib(text):
    entries = []
    for m in re.finditer(r'@(\w+)\{([^,]+),(.*?)\n\}', text, re.S):
        typ, key, body = m.group(1), m.group(2).strip(), m.group(3)
        fields = {}
        for fm in re.finditer(r'(\w+)\s*=\s*(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}|\d+|\w+)', body):
            v = fm.group(2)
            if v.startswith('{'): v = v[1:-1]
            fields[fm.group(1).lower()] = v
        entries.append((typ, key, fields))
    return entries

def norm(s):
    s = re.sub(r'\\[a-zA-Z]+\s*', '', s)           # drop latex commands
    s = s.replace('{', '').replace('}', '')
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9 ]', ' ', s.lower()).split()

def fam(bibauthors):
    out = []
    for a in re.split(r'\s+and\s+', bibauthors):
        a = a.strip()
        last = a.split(',')[0] if ',' in a else a.split()[-1]
        out.append(' '.join(norm(last)))
    return out

def crossref(doi):
    req = urllib.request.Request('https://api.crossref.org/works/' + doi,
                                 headers={'User-Agent': 'EECE7373-report-check (mailto:none@example.com)'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)['message']

bib = open(sys.argv[1]).read()
for typ, key, f in parse_bib(bib):
    doi = f.get('doi')
    if not doi:
        print(f'{key:24s} NO-DOI  ({typ}; url={f.get("url") or f.get("howpublished","")[:60]})')
        continue
    try:
        m = crossref(doi)
    except Exception as e:
        print(f'{key:24s} DOI-ERR {doi} {e}'); continue
    t_cr = ' '.join(norm((m.get('title') or [''])[0]))
    t_b = ' '.join(norm(f.get('title', '')))
    tr = difflib.SequenceMatcher(None, t_cr, t_b).ratio()
    fams_cr = [' '.join(norm(a.get('family', a.get('name', '')))) for a in m.get('author', [])]
    fams_b = fam(f.get('author', '')) if f.get('author') else []
    auth_ok = fams_cr == fams_b
    y_cr = None
    for k in ('published-print', 'published', 'issued', 'published-online'):
        if k in m and m[k].get('date-parts') and m[k]['date-parts'][0][0]:
            y_cr = m[k]['date-parts'][0][0]; break
    cont = (m.get('container-title') or [''])[0]
    pages = m.get('page', '')
    flag = 'OK ' if (tr > 0.95 and auth_ok and str(y_cr) == f.get('year')) else 'CHK'
    print(f'{key:24s} {flag} title={tr:.2f} authors={"same" if auth_ok else "DIFF"} year bib={f.get("year")} cr={y_cr} '
          f'pages bib={f.get("pages","-")} cr={pages or "-"}\n{"":28s}container: {cont[:95]}')
    if not auth_ok:
        print(f'{"":28s}cr  : {fams_cr}\n{"":28s}bib : {fams_b}')
    time.sleep(0.3)
