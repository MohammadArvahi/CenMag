"""Guarded Datasets 2.20 array-copy compatibility for diagnostic NumPy 2 runs.

The conversion below is adapted from Hugging Face Datasets 2.20.0
datasets/formatting/formatting.py (Apache-2.0). Only the final construction
changes from np.array(copy=False) to np.asarray. NumPy 1 canonical execution
does not activate this shim. No label, dtype, split, order or RNG is changed.
"""
from importlib.metadata import version
import numpy as np

def enable_numpy2_dataset_compat():
    if int(np.__version__.split('.')[0]) < 2:
        return {'status': 'NOT_REQUIRED'}
    if version('datasets') != '2.20.0':
        raise RuntimeError('CenMag NumPy 2 conversion shim requires exactly Datasets 2.20.0')
    from datasets.formatting.formatting import NumpyArrowExtractor
    if not getattr(NumpyArrowExtractor._arrow_array_to_numpy, '_cenmag_copy_compat', False):
        NumpyArrowExtractor._arrow_array_to_numpy = arrow_array_to_numpy
    return {'status': 'ENABLED', 'change': 'copy-permitting array construction only'}

def arrow_array_to_numpy(self, pa_array):
    import pyarrow as pa
    from datasets.features.features import _ArrayXDExtensionType, _is_zero_copy_only
    from datasets.formatting.formatting import _is_array_with_nulls
    if isinstance(pa_array, pa.ChunkedArray):
        if isinstance(pa_array.type, _ArrayXDExtensionType):
            zero_copy_only = _is_zero_copy_only(pa_array.type.storage_dtype, unnest=True)
            array = [row for chunk in pa_array.chunks for row in chunk.to_numpy(zero_copy_only=zero_copy_only)]
        else:
            zero_copy_only = _is_zero_copy_only(pa_array.type) and all(
                not _is_array_with_nulls(chunk) for chunk in pa_array.chunks)
            array = [row for chunk in pa_array.chunks for row in chunk.to_numpy(zero_copy_only=zero_copy_only)]
    else:
        if isinstance(pa_array.type, _ArrayXDExtensionType):
            zero_copy_only = _is_zero_copy_only(pa_array.type.storage_dtype, unnest=True)
            array = pa_array.to_numpy(zero_copy_only=zero_copy_only)
        else:
            zero_copy_only = _is_zero_copy_only(pa_array.type) and not _is_array_with_nulls(pa_array)
            array = pa_array.to_numpy(zero_copy_only=zero_copy_only).tolist()
    if len(array) > 0:
        if any((isinstance(x, np.ndarray) and (x.dtype == object or x.shape != array[0].shape))
               or (isinstance(x, float) and np.isnan(x)) for x in array):
            return np.asarray(array, dtype=object)
    return np.asarray(array)

arrow_array_to_numpy._cenmag_copy_compat = True
