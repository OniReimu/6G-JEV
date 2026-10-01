"""Exact-intent semantic-result cache. No embedding similarity or truth lookup."""
import hashlib
from src.schema import JSON_POLICY

class IntentCache:
    def __init__(self):self.entries={};self.version=hashlib.sha256(JSON_POLICY.encode()).hexdigest()
    def key(self,text):return (self.version,' '.join(text.split()))
    def lookup(self,text,now):
        entry=self.entries.get(self.key(text))
        return None if entry is None or entry['ready_at']>now else dict(entry['labels'])
    def store(self,text,labels,ready_at):
        self.entries[self.key(text)]=dict(labels=dict(labels),ready_at=ready_at)
