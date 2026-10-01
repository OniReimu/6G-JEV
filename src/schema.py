"""Shared public schema; truth is evaluation-only."""
FIELDS = {
    'service_type': ['count', 'detection', 'ocr', 'unsupported'],
    'locality': ['site_only', 'remote_allowed', 'unspecified'],
    'quality_floor': ['standard', 'high', 'unspecified'],
    'urgency': ['normal', 'urgent', 'unspecified'],
}
LABELS = 'ABCD'
POLICY = '''Extract the explicit requirements in the request. Do not infer missing requirements.
service_type: A=count objects (number only); B=detect/localize objects (boxes); C=OCR/read text; D=unsupported service.
locality: A=data must stay at the originating site; B=remote processing explicitly allowed; C=not specified.
quality_floor: A=standard quality explicitly sufficient; B=high quality explicitly required; C=not specified.
urgency: A=normal priority explicitly requested; B=urgent/priority requested; C=not specified.
Negation matters. A ban on remote processing means site_only. Not urgent means normal.
For ALL=, return exactly four uppercase label letters in the field order above, without separators or explanation.
For a single field followed by =, return exactly its one label letter.'''
JSON_POLICY = POLICY.split('For ALL=')[0] + '''If locality, quality, or urgency is not explicitly stated, choose C for that field, even if a normal default seems reasonable.
Return only a compact JSON object. Use the exact field names above and one uppercase letter as each value. Do not provide explanations.'''

def decode_labels(raw):
    if len(raw) != 4 or any(c not in LABELS[:len(v)] for c, v in zip(raw, FIELDS.values())):
        raise ValueError('invalid four-field labels')
    return {f: values[LABELS.index(c)] for (f, values), c in zip(FIELDS.items(), raw)}

def encode_labels(labels):
    return ''.join(LABELS[v.index(labels[f])] for f, v in FIELDS.items())
