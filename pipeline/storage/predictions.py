from ..core import PipelineError

import abc
import torch
import os


class PredictionsStorageBase(abc.ABC):
    @abc.abstractmethod
    def add(self, identifier, prediction):
        pass

    def add_batch(self, identifiers, predictions):
        for identifier, prediction in zip(identifiers, predictions):
            self.add(identifier, prediction)

    @abc.abstractmethod
    def flush(self):
        pass

    @abc.abstractmethod
    def get_all(self):
        pass

    @abc.abstractmethod
    def get_by_id(self, identifier):
        pass

    def get_by_id_batch(self, identifiers):
        result = []
        for identifier in identifiers:
            result.append(self.get_by_id(identifier))

        return torch.stack(result)

    @abc.abstractmethod
    def sort_by_id(self):
        pass


class PredictionsStorageFiles(PredictionsStorageBase):
    def __init__(self, path):
        if os.path.exists(path) and not os.path.isdir(path):
            raise PipelineError("{} should be a directory".format(path))

        os.makedirs(path, exist_ok=True)

        self._path = path

        self._identifiers = []
        self._predictions = []

        self._identifier_to_element_id = {}

        if os.path.exists(os.path.join(self._path, "identifiers")):
            self._load_predictions()

    def _load_predictions(self):
        self._identifiers = torch.load(os.path.join(self._path, "identifiers"))
        self._predictions = torch.load(os.path.join(self._path, "predictions"))

        assert len(self._identifiers) == len(self._predictions)

        for i, identifier in enumerate(self._identifiers):
            self._identifier_to_element_id[identifier] = i

    def _save_predictions(self):
        assert len(self._identifiers) == len(self._predictions)

        with open(os.path.join(self._path, "identifiers"), "wb") as fout:
            torch.save(self._identifiers, fout)

        with open(os.path.join(self._path, "predictions"), "wb") as fout:
            torch.save(self._predictions, fout)

    def add(self, identifier, prediction):
        self._identifiers.append(identifier)
        self._predictions.append(prediction)
        self._identifier_to_element_id[identifier] = len(self._identifiers)

    def flush(self):
        self._save_predictions()

    def get_all(self):
        return self._identifiers, self._predictions

    def get_by_id(self, identifier):
        if identifier not in self._identifier_to_element_id:
            raise PipelineError("Key error: {}".format(identifier))

        element_id = self._identifier_to_element_id[identifier]
        return self._predictions[element_id]

    def sort_by_id(self):
        result = sorted(zip(self._identifiers, self._predictions), key=lambda x: x[0])
        self._identifiers, self._predictions = list(zip(*result))
        self.flush()
