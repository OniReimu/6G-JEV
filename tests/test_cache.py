import unittest
from src.intent_cache import IntentCache
class CacheTests(unittest.TestCase):
 def test_inflight_is_not_a_hit(self):
  c=IntentCache();c.store('count vehicles',{'service_type':'count'},ready_at=2.)
  self.assertIsNone(c.lookup('count vehicles',1.99));self.assertIsNotNone(c.lookup('count vehicles',2.))
 def test_negation_and_version_separate_keys(self):
  c=IntentCache();c.store('remote allowed',{'locality':'remote_allowed'},0)
  self.assertIsNone(c.lookup('remote not allowed',1));c.version='new-policy'
  self.assertIsNone(c.lookup('remote allowed',1))
if __name__=='__main__':unittest.main()
