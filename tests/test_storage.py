from .common import make_temp_path

from pipeline.storage.state import StateStorageEmpty, StateStorageFile
from pipeline.core import PipelineError

import pytest


class TestStateStorageEmpty:
    def test_set_value(self):
        state_storage = StateStorageEmpty()
        state_storage.set_value("key_name", 123)

    def test_get_value(self):
        state_storage = StateStorageEmpty()

        with pytest.raises(PipelineError):
            state_storage.get_value("some_key")

        state_storage.set_value("some_key", 123)
        with pytest.raises(PipelineError):
            state_storage.get_value("some_key")

    def test_has_key(self):
        state_storage = StateStorageEmpty()

        assert not state_storage.has_key("key")
        state_storage.set_value("key", "abacaba")

        assert not state_storage.has_key("key")

    def test_remove_key(self):
        state_storage = StateStorageEmpty()

        with pytest.raises(PipelineError):
            state_storage.remove_key("abacaba")

        state_storage.set_value("abacaba", 9.23)
        with pytest.raises(PipelineError):
            state_storage.remove_key("abacaba")


class TestStateStorageFile:
    def test_basic(self):
        path = make_temp_path()
        state_storage = StateStorageFile(path)

        assert not state_storage.has_key("key")

        with pytest.raises(PipelineError):
            state_storage.remove_key("abacaba")

        with pytest.raises(PipelineError):
            state_storage.get_value("some_key")

    def test_save_load(self):
        path = make_temp_path()
        state_storage = StateStorageFile(path)

        state_storage.set_value("aba", 123)
        assert state_storage.get_value("aba") == 123
        assert state_storage.has_key("aba")

        state_storage = StateStorageFile(path)
        assert state_storage.get_value("aba") == 123
        assert state_storage.has_key("aba")

        state_storage.remove_key("aba")
        assert not state_storage.has_key("aba")

        state_storage = StateStorageFile(path)
        assert not state_storage.has_key("aba")
