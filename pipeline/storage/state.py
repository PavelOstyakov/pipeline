from ..core import PipelineError

import abc
import pickle
import os


class StateStorageBase(abc.ABC):
    @abc.abstractmethod
    def has_key(self, key: str):
        pass

    @abc.abstractmethod
    def get_value(self, key: str):
        pass

    @abc.abstractmethod
    def remove_key(self, key: str):
        pass

    @abc.abstractmethod
    def set_value(self, key: str, value: object):
        pass


class StateStorageEmpty(StateStorageBase):
    def set_value(self, key: str, value: object):
        pass

    def get_value(self, key: str):
        raise PipelineError("Key error: {}".format(key))

    def has_key(self, key: str):
        return False

    def remove_key(self, key: str):
        raise PipelineError("Key error: {}".format(key))


class StateStorageFile(StateStorageBase):
    def __init__(self, path: str):
        self._path = path

        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fout:
                pickle.dump({}, fout)

        with open(path, "rb") as fin:
            self._state = pickle.load(fin)

    def _save(self):
        with open(self._path, "wb") as fout:
            pickle.dump(self._state, fout)

    def has_key(self, key: str):
        return key in self._state

    def get_value(self, key: str):
        if key not in self._state:
            raise PipelineError("Key error: {}".format(key))

        return self._state[key]

    def set_value(self, key: str, value: object):
        self._state[key] = value

        self._save()

    def remove_key(self, key: str):
        if key not in self._state:
            raise PipelineError("Key error: {}".format(key))

        del self._state[key]

        self._save()

