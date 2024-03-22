from typing import Union, Tuple, Callable
import numpy as np
import abc
import json
import os
from collections import ChainMap
from typing import Any, ClassVar, Union
import logging
from functools import partial
import pprint
import dill
import copy
from pathlib import Path
import warnings


def perturb(X: np.ndarray,
            eps: Union[float, np.ndarray] = 1e-08,
            proba: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply perturbation to instance or prediction probabilities. Used for numerical calculation of gradients.
    -------
    X: Array to be perturbed
    eps: Size of perturbation
    proba: If True, the net effect of the perturbation needs to be 0 to keep the sum of the probabilities equal to 1
    -------
    Instances where a positive and negative perturbation is applied.
    """
    # N = batch size; F = nb of features in X
    shape = X.shape
    X = np.reshape(X, (shape[0], -1))  # NxF
    dim = X.shape[1]  # F
    pert = np.tile(np.eye(dim) * eps, (shape[0], 1))  # (N*F)xF
    if proba:
        eps_n = eps / (dim - 1)
        pert += np.tile((np.eye(dim) - np.ones((dim, dim))) * eps_n, (shape[0], 1))  # (N*F)xF
    X_rep = np.repeat(X, dim, axis=0)  # (N*F)xF
    X_pert_pos, X_pert_neg = X_rep + pert, X_rep - pert
    shape = (dim * shape[0],) + shape[1:]
    X_pert_pos = np.reshape(X_pert_pos, shape)  # (N*F)x(shape of X[0])
    X_pert_neg = np.reshape(X_pert_neg, shape)  # (N*F)x(shape of X[0])
    return X_pert_pos, X_pert_neg


def num_grad_batch(func: Callable,
                   X: np.ndarray,
                   args: Tuple = (),
                   eps: Union[float, np.ndarray] = 1e-08) -> np.ndarray:
    """
    Calculate the numerical gradients of a vector-valued function (typically a prediction function in classification)
    with respect to a batch of arrays X.
    Parameters
    ----------
    func
        Function to be differentiated
    X
        A batch of vectors at which to evaluate the gradient of the function
    args
        Any additional arguments to pass to the function
    eps
        Gradient step to use in the numerical calculation, can be a single float or one for each feature
    Returns
    -------
    An array of gradients at each point in the batch X
    """
    # N = gradient batch size; F = nb of features in X, P = nb of prediction classes, B = instance batch size
    batch_size = X.shape[0]
    data_shape = X[0].shape
    preds = func(X, *args)
    X_pert_pos, X_pert_neg = perturb(X, eps)  # (N*F)x(shape of X[0])
    X_pert = np.concatenate([X_pert_pos, X_pert_neg], axis=0)
    preds_concat = func(X_pert, *args)  # make predictions
    n_pert = X_pert_pos.shape[0]

    grad_numerator = preds_concat[:n_pert] - preds_concat[n_pert:]  # (N*F)*P
    grad_numerator = np.reshape(np.reshape(grad_numerator, (batch_size, -1)),
                                (batch_size, preds.shape[1], -1), order='F')  # NxPxF

    grad = grad_numerator / (2 * eps)  # NxPxF
    grad = grad.reshape(preds.shape + data_shape)  # BxPx(shape of X[0])

    return grad

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(
                obj,
                (
                        np.int_,
                        np.intc,
                        np.intp,
                        np.int8,
                        np.int16,
                        np.int32,
                        np.int64,
                        np.uint8,
                        np.uint16,
                        np.uint32,
                        np.uint64,
                ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def load_explainer(path: Union[str, os.PathLike], predictor) -> 'Explainer':
    """
    Load an explainer from disk.
    Parameters
    ----------
    path
        Path to a directory containing the saved explainer.
    predictor
        Model or prediction function used to originally initialize the explainer.
    Returns
    -------
    An explainer instance.
    """
    # load metadata
    with open(Path(path, 'meta.dill'), 'rb') as f:
        meta = dill.load(f)

    # check version
    if meta['version'] != __version__:
        warnings.warn(f'Trying to load explainer from version {meta["version"]} when using version {__version__}. '
                      f'This may lead to breaking code or invalid results.')

    name = meta['name']
    try:
        # get the explainer specific load function
        load_fn = getattr(thismodule, '_load_' + name)
    except AttributeError:
        load_fn = _simple_load
    return load_fn(path, predictor, meta)


def save_explainer(explainer: 'Explainer', path: Union[str, os.PathLike]) -> None:
    """
    Save an explainer to disk. Uses the `dill` module.
    Parameters
    ----------
    explainer
        Explainer instance to save to disk.
    path
        Path to a directory. A new directory will be created if one does not exist.
    """
    name = explainer.meta['name']
    if name in NOT_SUPPORTED:
        raise NotImplementedError(f'Saving for {name} not yet supported')

    path = Path(path)

    # create directory
    path.mkdir(parents=True, exist_ok=True)

    # save metadata
    meta = copy.deepcopy(explainer.meta)
    meta['version'] = explainer._version
    with open(Path(path, 'meta.dill'), 'wb') as f:
        dill.dump(meta, f)

    try:
        # get explainer specific save function
        save_fn = getattr(thismodule, '_save_' + name)
    except AttributeError:
        # no explainer specific functionality required, just set predictor to `None` and dump
        save_fn = _simple_save
    save_fn(explainer, path)

import abc
import json
import os
from collections import ChainMap
from typing import Any, ClassVar, Union
import logging
from functools import partial
import pprint

import attr

# from alibi.saving import load_explainer, save_explainer, NumpyEncoder
# from alibi.version import __version__

logger = logging.getLogger(__name__)


# default metadata
def default_meta() -> dict:
    return {
        "name": None,
        "type": [],
        "explanations": [],
        "params": {},
    }


class AlibiPrettyPrinter(pprint.PrettyPrinter):
    """
    Overrides the built in dictionary pretty representation to look more similar to the external
    prettyprinter libary.
    """
    _dispatch = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # `sort_dicts` kwarg was only introduced in Python 3.8 so we just override it here.
        # Before Python 3.8 the printing was done in insertion order by default.
        self._sort_dicts = False

    def _pprint_dict(self, object, stream, indent, allowance, context, level):
        # Add a few newlines and the appropriate indentation to dictionary printing
        # compare with https://github.com/python/cpython/blob/3.9/Lib/pprint.py
        write = stream.write
        indent += self._indent_per_level
        write('{\n' + ' ' * (indent + 1))
        if self._indent_per_level > 1:
            write((self._indent_per_level - 1) * ' ')
        length = len(object)
        if length:
            if self._sort_dicts:
                items = sorted(object.items(), key=pprint._safe_tuple)
            else:
                items = object.items()
            self._format_dict_items(items, stream, indent, allowance + 1,
                                    context, level)
        write('}\n' + ' ' * (indent - 1))

    _dispatch[dict.__repr__] = _pprint_dict


alibi_pformat = partial(AlibiPrettyPrinter().pformat)

__version__ = '0.5.9dev'

@attr.s
class Explainer(abc.ABC):
    """
    Base class for explainer algorithms
    """
    _version: ClassVar[str] = __version__
    meta = attr.ib(default=attr.Factory(default_meta), repr=alibi_pformat)  # type: dict

    def __attrs_post_init__(self):
        # add a name to the metadata dictionary
        self.meta["name"] = self.__class__.__name__

        # expose keys stored in self.meta as attributes of the class.
        for key, value in self.meta.items():
            setattr(self, key, value)

    @abc.abstractmethod
    def explain(self, X: Any) -> "Explanation":
        pass

    @classmethod
    def load(cls, path: Union[str, os.PathLike], predictor: Any) -> "Explainer":
        """
        Load an explainer from disk.
        Parameters
        ----------
        path
            Path to a directory containing the saved explainer.
        predictor
            Model or prediction function used to originally initialize the explainer.
        Returns
        -------
        An explainer instance.
        """
        return load_explainer(path, predictor)

    def reset_predictor(self, predictor: Any) -> None:
        raise NotImplementedError

    def save(self, path: Union[str, os.PathLike]) -> None:
        """
        Save an explainer to disk. Uses the `dill` module.
        Parameters
        ----------
        path
            Path to a directory. A new directory will be created if one does not exist.
        """
        save_explainer(self, path)

    def _update_metadata(self, data_dict: dict, params: bool = False) -> None:
        """
        Updates the metadata of the explainer using the data from the `data_dict`. If the params option
        is specified, then each key-value pair is added to the metadata `'params'` dictionary.
        Parameters
        ----------
        data_dict
            Contains the data to be stored in the metadata.
        params
            If True, the method updates the `'params'` attribute of the metatadata.
        """

        if params:
            for key in data_dict.keys():
                self.meta['params'].update([(key, data_dict[key])])
        else:
            self.meta.update(data_dict)


class FitMixin(abc.ABC):
    @abc.abstractmethod
    def fit(self, X: Any) -> "Explainer":
        pass


@attr.s
class Explanation:
    """
    Explanation class returned by explainers.
    """
    meta = attr.ib(repr=alibi_pformat)  # type: dict
    data = attr.ib(repr=alibi_pformat)  # type: dict

    def __attrs_post_init__(self):
        """
        Expose keys stored in self.meta and self.data as attributes of the class.
        """
        for key, value in ChainMap(self.meta, self.data).items():
            setattr(self, key, value)

    def to_json(self) -> str:
        """
        Serialize the explanation data and metadata into a json format.
        Returns
        -------
        String containing json representation of the explanation
        """
        return json.dumps(attr.asdict(self), cls=NumpyEncoder)

    @classmethod
    def from_json(cls, jsonrepr) -> "Explanation":
        """
        Create an instance of an Explanation class using a json representation of the Explanation.
        Parameters
        ----------
        jsonrepr
            json representation of an explanation
        Returns
        -------
            An Explanation object
        """
        dictrepr = json.loads(jsonrepr)
        try:
            meta = dictrepr['meta']
            data = dictrepr['data']
        except KeyError:
            logger.exception("Invalid explanation representation")
        return cls(meta=meta, data=data)

    def __getitem__(self, item):
        """
        This method is purely for deprecating previous behaviour of accessing explanation
        data via items in the returned dictionary.
        """
        import warnings
        msg = "The Explanation object is not a dictionary anymore and accessing elements should " \
              "be done via attribute access. Accessing via item will stop working in a future version."
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return getattr(self, item)

# Counterfactuals
"""
Default counterfactual metadata.
"""
DEFAULT_META_CF = {"name": None,
                   "type": ["blackbox", "tensorflow", "keras"],
                   "explanations": ["local"],
                   "params": {}}

"""
Default counterfactual data.
"""
DEFAULT_DATA_CF = {"cf": None,
                   "all": [],
                   "orig_class": None,
                   "orig_proba": None,
                   "success": None}  # type: dict