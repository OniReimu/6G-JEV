"""Transparent keyword baseline; authored from task schema, no truth access."""
import re

def parse(text):
    s=text.lower()
    def has(pattern):return re.search(pattern,s) is not None
    service='unsupported'
    if has(r'\b(count|tally)\b|how many|number of (cars|vehicles)'):service='count'
    elif has(r'bounding box|\ba box\b|\bboxes\b|\blocate\b|\blocations\b'):service='detection'
    elif has(r'\bread\b.*\b(text|sign)\b|\btranscribe\b|printed characters|writing.*(notice|text)|extract.*text'):service='ocr'
    locality='unspecified'
    if has(r'no off.site|only this site|remain on premises|keep.*this site|do not transmit.*outside|do not send.*elsewhere'):locality='site_only'
    elif has(r'remote.*(permitted|allowed|acceptable)|offload|fine to send.*another site|no locality restriction'):locality='remote_allowed'
    quality='unspecified'
    if has(r'no need for high|standard.*(sufficient|enough|meets)|use standard|accept.*standard'):quality='standard'
    elif has(r'high.quality|high quality|quality is high'):quality='high'
    urgency='unspecified'
    if has(r'not urgent|normal.priority|normal priority|ordinary priority|no priority escalation'):urgency='normal'
    elif has(r'urgent|priority alert|priority over routine'):urgency='urgent'
    return dict(service_type=service,locality=locality,quality_floor=quality,urgency=urgency)
