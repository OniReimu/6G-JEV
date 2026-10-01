import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
from src.credentials import load_openrouter_key, save_openrouter_key


class CredentialTests(unittest.TestCase):
    def test_private_round_trip_without_export(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{},clear=True):
            path=Path(tmp)/'key.env'
            save_openrouter_key('fake-test-value',path)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode),0o600)
            self.assertEqual(load_openrouter_key(path),'fake-test-value')
            self.assertNotIn('OPENROUTER_API_KEY',os.environ)

    def test_environment_precedence_does_not_read_file(self):
        with patch.dict(os.environ,{'OPENROUTER_API_KEY':'environment-test'}):
            self.assertEqual(load_openrouter_key('/does/not/exist'),'environment-test')

    def test_comments_and_quotes_are_data(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{},clear=True):
            path=Path(tmp)/'key.env'
            path.write_text('# comment\nexport OPENROUTER_API_KEY="literal-$VARIABLE"\n')
            path.chmod(0o600)
            self.assertEqual(load_openrouter_key(path),'literal-$VARIABLE')

    def test_bad_file_does_not_leak_value(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{},clear=True):
            path=Path(tmp)/'key.env'
            path.write_text('OPENROUTER_API_KEY=private-test-token\nBAD=private-test-token\n');path.chmod(0o600)
            with self.assertRaises(RuntimeError) as caught:load_openrouter_key(path)
            self.assertNotIn('private-test-token',str(caught.exception))
            path.chmod(0o644)
            with self.assertRaises(RuntimeError):load_openrouter_key(path)

    def test_missing_or_empty_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{},clear=True):
            path=Path(tmp)/'key.env'
            self.assertIsNone(load_openrouter_key(path))
            path.write_text('OPENROUTER_API_KEY=\n');path.chmod(0o600)
            self.assertIsNone(load_openrouter_key(path))

    def test_refuses_symlink_on_save_and_read(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{},clear=True):
            target=Path(tmp)/'real.env';save_openrouter_key('unchanged-test',target)
            link=Path(tmp)/'link.env';link.symlink_to(target)
            with self.assertRaises(RuntimeError):save_openrouter_key('replacement-test',link)
            with self.assertRaises(OSError):load_openrouter_key(link)
            self.assertEqual(load_openrouter_key(target),'unchanged-test')


if __name__=='__main__':unittest.main()
