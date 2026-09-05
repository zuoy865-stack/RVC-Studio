"""Public Python API for pymss.

pymss provides model catalog helpers, model downloading, audio I/O,
logging helpers, and the ``MSSeparator`` runtime for music source
separation. Most users can import from this top-level package instead of
importing submodules directly.

Exports:
    MSSeparator: Main runtime class for loading separation models and producing
        stems. Prefer ``MSSeparator.from_model_name(...)`` for catalog models.
    get_separation_logger: Create or reuse the package logger.
    create_separator: Create ``MSSeparator`` from a catalog model name.
    get_model_entry: Resolve catalog metadata for one model name or alias.
    list_models: List model catalog entries.
    resolve_model: Resolve catalog model paths without constructing a
        separator.
    download_model: Download all files required by one catalog model.
    load_audio: Load an audio file into a NumPy array.
    save_audio: Save a NumPy audio array to wav/flac/mp3/m4a.

Example:
    >>> from pymss import MSSeparator
    >>> separator = MSSeparator.from_model_name(
    ...     "bs_roformer_voc_hyperacev2",
    ...     download=True,
    ...     model_dir="models",
    ... )
    >>> separator.process_folder("song.wav")

Example:
    >>> from pymss import download_model, list_models
    >>> models = list_models(supported=True)
    >>> download_model(models[0].name, model_dir="models")

"""

from .separator import MSSeparator
from .logger import get_separation_logger
from .model_registry import create_separator, get_model_entry, list_models, resolve_model
from .model_download import download_model
from .audio_io import load_audio, save_audio

__all__ = (
    "MSSeparator",
    "get_separation_logger",
    "create_separator",
    "get_model_entry",
    "list_models",
    "resolve_model",
    "download_model",
    "load_audio",
    "save_audio",
)
