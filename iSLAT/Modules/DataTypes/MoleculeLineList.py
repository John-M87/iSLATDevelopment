from pathlib import Path
import numpy as np
import time
import os
import hashlib
from collections import namedtuple
from typing import ClassVar, Dict, Optional, List, Any, NamedTuple, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import pandas

# Performance logging
from iSLAT.Modules.Debug.PerformanceLogger import perf_log, log_timing, PerformanceSection
from iSLAT.Modules.Debug.DebugConfig import debug_config

#from .MoleculeLine import MoleculeLine

# Lazy imports for performance
from ._pandas_import import get_pandas as _get_pandas

# Cache version - increment when cache format changes
# v2: Switched from compressed npz to separate uncompressed npy files for faster loading
# v3: Added source_format tag and extra_columns support for multi-format readers
_CACHE_VERSION = 3

class LineTuple(NamedTuple):
    """Named tuple for line data"""
    nr: np.ndarray[int]
    '''Line number'''
    lev_up: np.ndarray[str]
    '''Upper energy level (quantum state label)'''
    lev_low: np.ndarray[str]
    '''Lower energy level (quantum state label)'''
    lam: np.ndarray[float]
    '''Wavelength in microns'''
    freq: np.ndarray[float]
    '''Frequency in Hz'''
    a_stein: np.ndarray[float]
    '''Einstein A coefficient'''
    e_up: np.ndarray[float]
    '''Upper state energy'''
    e_low: np.ndarray[float]
    '''Lower state energy'''
    g_up: np.ndarray[int]
    '''Upper state degeneracy'''
    g_low: np.ndarray[int]
    '''Lower state degeneracy'''

# Structured array dtype for efficient line data storage
# Note: lev_up and lev_low are quantum state labels (strings like '0_0_0|10_2_9')
_LINE_DTYPE = np.dtype([
    ('nr', np.int32),
    ('lev_up', 'U64'),  # Unicode string up to 64 chars for quantum state labels
    ('lev_low', 'U64'), # Unicode string up to 64 chars for quantum state labels
    ('lam', np.float64),
    ('freq', np.float64),
    ('a_stein', np.float64),
    ('e_up', np.float64),
    ('e_low', np.float64),
    ('g_up', np.int32),
    ('g_low', np.int32)
])

from ._mixins import WavelengthRangeMixin

class MoleculeLineList(WavelengthRangeMixin):
    """
    Efficient molecular line list with lazy loading and caching.
    """
    __slots__ = ('molecule_id', 'lines', 'partition_function', '_partition_type', 
                 '_lines_type', '_lines_cache', '_lines_cache_valid', '_wavelengths_cache',
                 '_frequencies_cache', '_a_stein_cache', '_e_up_cache', '_e_low_cache',
                 '_g_up_cache', '_g_low_cache', '_data_loaded', '_filename', '_raw_lines_data',
                 '_pandas_df_cache', '_molar_mass', '_wavelength_range', '_filtered_raw_data',
                 '_lev_up_cache', '_lev_low_cache', '_extra_fields', '_source_format',
                 '_quantum_cache')
    
    # Class-level reader registry — populated lazily to avoid circular imports
    _reader_registry: ClassVar[Dict[str, Any]] = {}
    
    def __init__(self, molecule_id: Optional[str] = None, filename: Optional[str | Path] = None, 
                 lines_data: Optional[List[dict]] = None,
                 wavelength_range: Optional[tuple] = None,
                 format: Optional[str] = None,
                 extra_fields: Optional[Dict[str, list]] = None):
        """
        Initialize a MoleculeLineList object.

        Parameters
        ----------
        molecule_id : str
            Identifier for the molecule.
        lines_data : list of dict, optional
            List of dictionaries containing line data with keys like 'frequency', 'wavelength', 'intensity', etc.
        filename : str | Path, optional
            Path to a .par file to read molecular data from.
        wavelength_range : tuple of (float, float), optional
            Wavelength range ``(lam_min, lam_max)`` in microns.  When set,
            all data accessors return only lines within this range.
        format : str, optional
            File format hint (``"hitran"``, ``"csv"``, ``"saved"``, or
            ``"auto"``).  When *None* or ``"auto"`` the format is detected
            from the file extension / header.  Only used when *filename*
            is provided.
        extra_fields : dict[str, list], optional
            Additional per-line columns beyond the 10 core fields.
            Keys are column names, values are lists aligned with
            *lines_data*.  Preserved through round-trips.
        """
        self.molecule_id = molecule_id
        self.lines: List[Any] = []
        self.partition_function = None
        self._data_loaded = False
        self._filename = filename
        self._raw_lines_data = None
        self._pandas_df_cache = None
        self._molar_mass = None
        self._wavelength_range = wavelength_range
        self._filtered_raw_data = None
        self._extra_fields: Dict[str, list] = extra_fields or {}
        self._source_format: Optional[str] = format
        
        # Define namedtuple types for data structure
        self._partition_type = namedtuple('partition', ['t', 'q'])
        self._lines_type = LineTuple
        #namedtuple('lines', ['nr', 'lev_up', 'lev_low', 'lam', 'freq', 'a_stein',
                           #                    'e_up', 'e_low', 'g_up', 'g_low'])
        
        # Cache for performance optimization
        self._lines_cache = None
        self._lines_cache_valid = False
        self._wavelengths_cache = None
        self._frequencies_cache = None
        self._a_stein_cache = None
        self._e_up_cache = None
        self._e_low_cache = None
        self._g_up_cache = None
        self._g_low_cache = None
        self._lev_up_cache = None
        self._lev_low_cache = None
        self._quantum_cache: dict = {}
        
        # Lazy loading - only load data when needed
        if filename:
            self._filename = filename
        elif lines_data:
            self._load_from_data(lines_data)

    # ================================
    # Class-level reader registry API
    # ================================

    @classmethod
    def _ensure_registry(cls) -> None:
        """Populate the reader registry on first access (avoids circular imports)."""
        if not cls._reader_registry:
            from iSLAT.Modules.FileHandling.line_list_readers import DEFAULT_READERS
            cls._reader_registry.update(DEFAULT_READERS)

    @classmethod
    def register_reader(cls, format_name: str, reader: Any) -> None:
        """Register a new reader for *format_name*.

        Parameters
        ----------
        format_name : str
            Short tag (e.g. ``"parquet"``, ``"fits"``).
        reader : LineListReader
            Any object with a ``read(filepath) -> ReadResult`` method.
        """
        cls._ensure_registry()
        cls._reader_registry[format_name] = reader

    @classmethod
    def from_file(
        cls,
        filepath: Union[str, Path],
        molecule_id: Optional[str] = None,
        format: Optional[str] = None,
        wavelength_range: Optional[tuple] = None,
    ) -> "MoleculeLineList":
        """Construct a :class:`MoleculeLineList` from any supported file.

        This is the recommended entry point for creating a line list from
        a file.  The *format* is auto-detected when omitted.

        Parameters
        ----------
        filepath : str or Path
            Path to the line-list file.
        molecule_id : str, optional
            Molecule identifier.  If *None*, attempts to derive it from
            the file metadata.
        format : str, optional
            ``"hitran"``, ``"csv"``, ``"saved"``, or *None* for auto.
        wavelength_range : tuple, optional
            ``(lam_min, lam_max)`` filter.

        Returns
        -------
        MoleculeLineList
        """
        cls._ensure_registry()

        from iSLAT.Modules.FileHandling.line_list_readers import detect_format, ReadResult

        filepath = Path(filepath)
        fmt = format or detect_format(filepath)

        reader = cls._reader_registry.get(fmt)
        if reader is None:
            raise ValueError(
                f"No reader registered for format {fmt!r}.  "
                f"Available: {sorted(cls._reader_registry)}"
            )

        result: ReadResult = reader.read(filepath)

        # Derive molecule_id from metadata when not explicitly given
        mid = molecule_id
        if mid is None:
            species = result.metadata.get("species")
            if species and len(species) == 1:
                mid = species[0]

        obj = cls(
            molecule_id=mid,
            lines_data=result.lines_data,
            wavelength_range=wavelength_range,
            format=fmt,
            extra_fields=result.extra_columns,
        )

        # Attach partition function and metadata
        obj.partition_function = result.partition
        obj._molar_mass = result.metadata.get("molar_mass")

        return obj

    # ================================
    # Extra-fields / source-format properties
    # ================================

    @property
    def extra_fields(self) -> Dict[str, list]:
        """Additional per-line columns beyond the 10-core dtype.

        Returns an empty dict when the line list was loaded from HITRAN
        (which has no extras).  For CSV / saved-lines sources this may
        contain ``xmin``, ``xmax``, ``Flux_data``, etc.
        """
        return self._extra_fields

    @property
    def source_format(self) -> Optional[str]:
        """Format tag of the original file (``"hitran"``, ``"csv"``, ``"saved"``, or *None*)."""
        return self._source_format

    def _ensure_data_loaded(self):
        """Ensure molecular data is loaded (lazy loading)."""
        if not self._data_loaded and self._filename is not None:
            self._load_from_file(self._filename)
            self._data_loaded = True

    # ================================
    # Wavelength Range Filtering
    # ================================
    # wavelength_range property provided by WavelengthRangeMixin

    def _on_wavelength_range_changed(self, old, new):
        """Hook called by WavelengthRangeMixin when wavelength_range changes."""
        self._filtered_raw_data = None
        self.lines = None  # Force recreation of line objects
        self._invalidate_caches()

    def _get_active_raw_data(self):
        """Return raw line data filtered by *wavelength_range* (if set).

        When ``wavelength_range`` is ``None`` the full array is returned.
        The filtered view is cached and rebuilt only when the range
        changes (via the setter which invalidates all caches).
        """
        self._ensure_data_loaded()
        if self._raw_lines_data is None or len(self._raw_lines_data) == 0:
            return self._raw_lines_data

        if self._wavelength_range is None:
            return self._raw_lines_data

        if self._filtered_raw_data is not None:
            return self._filtered_raw_data

        lam_min, lam_max = self._wavelength_range
        lam_values = self._raw_lines_data['lam']
        mask = (lam_values >= lam_min) & (lam_values <= lam_max)
        self._filtered_raw_data = self._raw_lines_data[mask]
        return self._filtered_raw_data

    def without_wavelength_range(self) -> "MoleculeLineList":
        """Return this line list with no *wavelength_range* applied.

        The copy shares the already-loaded line data rather than re-reading
        the source file, so it also covers lists built in memory (e.g. a
        filtered result) that have no file behind them.  Returns ``self``
        when no range is set.
        """
        self._ensure_data_loaded()
        if self._wavelength_range is None or self._raw_lines_data is None:
            return self
        full = MoleculeLineList(molecule_id=self.molecule_id,
                                format=self._source_format,
                                extra_fields=dict(self._extra_fields))
        full._filename = self._filename
        full._raw_lines_data = self._raw_lines_data
        full.partition_function = self.partition_function
        full._molar_mass = self._molar_mass
        full.lines = None  # Created on demand
        full._data_loaded = True
        return full

    def _load_from_data(self, lines_data: List[dict]):
        """Load from provided line data with optimized batch processing."""
        # Convert list of dicts to structured numpy array for fast column access
        if lines_data:
            n_lines = len(lines_data)
            self._raw_lines_data = np.empty(n_lines, dtype=_LINE_DTYPE)
            
            # Fill structured array from list of dicts
            for i, d in enumerate(lines_data):
                self._raw_lines_data[i] = (
                    d.get('nr', 0),
                    str(d.get('lev_up', '')),
                    str(d.get('lev_low', '')),
                    d.get('lam', 0.0),
                    d.get('freq', 0.0),
                    d.get('a_stein', 0.0),
                    d.get('e_up', 0.0),
                    d.get('e_low', 0.0),
                    d.get('g_up', 0),
                    d.get('g_low', 0)
                )
        else:
            self._raw_lines_data = np.empty(0, dtype=_LINE_DTYPE)
        
        self.lines = None  # Will be created on demand
        self._data_loaded = True
        self._invalidate_caches()

    # ================================
    # Binary Cache System for HITRAN Files
    # ================================
    def _get_cache_path(self, source_filepath: str) -> str:
        """
        Get the cache file path for a given source file.
        
        Parameters
        ----------
        source_filepath : str
            Path to the original source file (.par, .csv, etc.)
            
        Returns
        -------
        str
            Path to the cache directory for this molecule/file
        """
        from iSLAT.Modules.FileHandling import line_cache_folder_path
        
        # Create cache folder if it doesn't exist
        os.makedirs(line_cache_folder_path, exist_ok=True)
        
        # Generate cache directory name based on source file name and molecule_id
        source_basename = os.path.basename(source_filepath)
        # Use a directory for v3 cache (multiple .npy files)
        cache_dir = os.path.join(line_cache_folder_path, f"{self.molecule_id}_{source_basename}")
        return cache_dir
    
    def _is_cache_valid(self, source_filepath: str, cache_filepath: str) -> bool:
        """
        Check if the cache file is valid (exists and is newer than source).
        
        Parameters
        ----------
        source_filepath : str
            Path to the original .par file
        cache_filepath : str
            Path to the cache directory
            
        Returns
        -------
        bool
            True if cache is valid and can be used
        """
        # For v2 cache, check if the directory and metadata file exist
        metadata_file = os.path.join(cache_filepath, "metadata.npy")
        lines_file = os.path.join(cache_filepath, "lines.npy")
        
        if not os.path.exists(metadata_file) or not os.path.exists(lines_file):
            return False
        
        if not os.path.exists(source_filepath):
            return False
        
        # Check if cache is newer than source file
        cache_mtime = os.path.getmtime(metadata_file)
        source_mtime = os.path.getmtime(source_filepath)
        
        return cache_mtime > source_mtime
    
    def _load_from_cache(self, cache_filepath: str) -> bool:
        """
        Load molecular data from fast binary cache files.
        
        Uses separate .npy files without compression.
        Restores source_format and extra_columns when available (v3+).
        
        Parameters
        ----------
        cache_filepath : str
            Path to the cache directory
            
        Returns
        -------
        bool
            True if cache was loaded successfully, False otherwise
        """
        try:
            metadata_file = os.path.join(cache_filepath, "metadata.npy")
            lines_file = os.path.join(cache_filepath, "lines.npy")
            partition_t_file = os.path.join(cache_filepath, "partition_t.npy")
            partition_q_file = os.path.join(cache_filepath, "partition_q.npy")
            
            # Load metadata (small, fast)
            metadata = np.load(metadata_file, allow_pickle=False)
            cache_version = int(metadata[0])
            
            if cache_version != _CACHE_VERSION:
                print(f"Cache version mismatch (got {cache_version}, expected {_CACHE_VERSION}), rebuilding...")
                return False
            
            self._molar_mass = float(metadata[1])
            
            # Load partition function (small arrays, fast)
            partition_t = np.load(partition_t_file, allow_pickle=False)
            partition_q = np.load(partition_q_file, allow_pickle=False)
            self.partition_function = self._partition_type(t=partition_t, q=partition_q)
            
            # Use memory mapping for very large files, direct load for smaller ones
            file_size = os.path.getsize(lines_file)
            if file_size > 25_000_000:  # > 25MB: use memory mapping
                self._raw_lines_data = np.load(lines_file, mmap_mode='r', allow_pickle=False)
            else:
                self._raw_lines_data = np.load(lines_file, allow_pickle=False)
            
            # v3+: restore source_format
            fmt_file = os.path.join(cache_filepath, "source_format.npy")
            if os.path.exists(fmt_file):
                fmt_arr = np.load(fmt_file, allow_pickle=True)
                self._source_format = str(fmt_arr.item()) if fmt_arr.size else None

            # v3+: restore extra_columns (pickled dict)
            extras_file = os.path.join(cache_filepath, "extra_columns.npy")
            if os.path.exists(extras_file):
                extras = np.load(extras_file, allow_pickle=True)
                self._extra_fields = extras.item() if extras.size else {}

            self.lines = None  # Will be created on demand
            self._data_loaded = True
            self._invalidate_caches()
            
            return True
            
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return False
    
    def _save_to_cache(self, cache_filepath: str) -> bool:
        """
        Save molecular data to fast binary cache files.
        
        Uses separate uncompressed .npy files for maximum load speed.
        v3: also persists source_format and extra_columns.
        
        Parameters
        ----------
        cache_filepath : str
            Path to the cache directory
            
        Returns
        -------
        bool
            True if cache was saved successfully, False otherwise
        """
        try:
            # Create cache directory
            os.makedirs(cache_filepath, exist_ok=True)
            
            # Prepare partition function arrays
            partition_t = np.array(self.partition_function.t, dtype=np.float64) if self.partition_function else np.array([], dtype=np.float64)
            partition_q = np.array(self.partition_function.q, dtype=np.float64) if self.partition_function else np.array([], dtype=np.float64)
            
            # Save metadata (version + molar mass)
            metadata = np.array([_CACHE_VERSION, self._molar_mass if self._molar_mass else 0.0], dtype=np.float64)
            np.save(os.path.join(cache_filepath, "metadata.npy"), metadata)
            
            # Save partition function arrays
            np.save(os.path.join(cache_filepath, "partition_t.npy"), partition_t)
            np.save(os.path.join(cache_filepath, "partition_q.npy"), partition_q)
            
            # Save lines data - the main payload
            np.save(os.path.join(cache_filepath, "lines.npy"), self._raw_lines_data)
            
            # v3: Save source format tag
            np.save(
                os.path.join(cache_filepath, "source_format.npy"),
                np.array(self._source_format or "", dtype=object),
            )

            # v3: Save extra columns (pickled dict of lists)
            if self._extra_fields:
                np.save(
                    os.path.join(cache_filepath, "extra_columns.npy"),
                    np.array(self._extra_fields, dtype=object),
                )

            return True
            
        except Exception as e:
            print(f"Failed to save cache: {e}")
            return False

    def _load_from_file(self, filename: str | Path):
        """
        Load molecular data from a file, using binary cache if available.

        When ``_source_format`` is set (or auto-detected as non-HITRAN),
        the reader registry is consulted.  Otherwise the legacy HITRAN
        path is used for full backward compatibility.
        
        Parameters
        ----------
        filename : str | Path
            Path to the data file.
        """
        section = PerformanceSection(f"MoleculeLineList._load_from_file({self.molecule_id})")
        section.start()
        
        # Resolve full path for cache lookup
        from iSLAT.Modules.FileHandling import hitran_data_folder_path, absolute_data_files_path
        
        if os.path.isabs(filename):
            source_filepath = str(filename)
        else:
            # Try relative to data files path
            source_filepath = os.path.join(absolute_data_files_path, str(filename))
            if not os.path.exists(source_filepath):
                source_filepath = str(filename)

        # --- Determine format --------------------------------------------------
        fmt = self._source_format
        if fmt is None or fmt == "auto":
            try:
                from iSLAT.Modules.FileHandling.line_list_readers import detect_format
                fmt = detect_format(source_filepath)
                self._source_format = fmt
            except (ValueError, ImportError):
                # Fallback: assume HITRAN for backward compat
                fmt = "hitran"
                self._source_format = fmt

        # --- Cache check (applies to all formats) ----------------------------
        cache_filepath = self._get_cache_path(source_filepath)
        
        section.mark("check_cache")
        if self._is_cache_valid(source_filepath, cache_filepath):
            section.mark("load_from_cache")
            if self._load_from_cache(cache_filepath):
                debug_config.info('molecule_dict', f"[CACHE HIT] Loaded {self.molecule_id} from binary cache")
                section.mark("cache_load_complete")
                section.end()
                section.get_breakdown(print_output=True)
                return
        
        # --- Cache miss — read via the appropriate reader ---------------------
        section.mark("read_data")
        print(f"[CACHE MISS] Parsing {self.molecule_id} from source file...")

        if fmt == "hitran":
            # Legacy HITRAN path — keeps full backward compatibility
            from iSLAT.Modules.FileHandling.molecular_data_reader import read_molecular_data
            
            partition_function, lines_data, other_fields = read_molecular_data(self.molecule_id, filename)
            section.mark("parse_complete")
            
            self.partition_function = partition_function
            self._molar_mass = other_fields[0][1]
            print(f'Molar_mass: {self._molar_mass}')
        else:
            # Use reader registry for CSV / saved / custom formats
            self.__class__._ensure_registry()
            reader = self.__class__._reader_registry.get(fmt)
            if reader is None:
                raise ValueError(
                    f"No reader registered for format {fmt!r}.  "
                    f"Available: {sorted(self.__class__._reader_registry)}"
                )

            from iSLAT.Modules.FileHandling.line_list_readers import ReadResult
            result: ReadResult = reader.read(source_filepath)
            section.mark("parse_complete")

            self.partition_function = result.partition
            self._molar_mass = result.metadata.get("molar_mass")
            self._extra_fields = result.extra_columns
            lines_data = result.lines_data

        # Convert list of dicts to structured numpy array for fast column access
        section.mark("convert_to_structured_array")
        if lines_data:
            n_lines = len(lines_data)
            self._raw_lines_data = np.empty(n_lines, dtype=_LINE_DTYPE)
            
            # Fill structured array from list of dicts
            for i, d in enumerate(lines_data):
                self._raw_lines_data[i] = (
                    d.get('nr', 0),
                    str(d.get('lev_up', '')),
                    str(d.get('lev_low', '')),
                    d.get('lam', 0.0),
                    d.get('freq', 0.0),
                    d.get('a_stein', 0.0),
                    d.get('e_up', 0.0),
                    d.get('e_low', 0.0),
                    d.get('g_up', 0),
                    d.get('g_low', 0)
                )
        else:
            self._raw_lines_data = np.empty(0, dtype=_LINE_DTYPE)
        
        self.lines = None  # Will be created on demand
        self._data_loaded = True
        self._invalidate_caches()
        
        # Save to cache for next time
        section.mark("save_to_cache")
        if self._save_to_cache(cache_filepath):
            print(f"[CACHE SAVED] {self.molecule_id} cached for faster loading")
        
        section.end()
        section.get_breakdown(print_output=True)
        
    def _ensure_lines_created(self):
        """Ensure MoleculeLine objects are created from (filtered) raw data."""
        active_data = self._get_active_raw_data()
        if self.lines is None and active_data is not None and len(active_data) > 0:
            from iSLAT.Modules.DataTypes.MoleculeLine import MoleculeLine
            # Create MoleculeLine objects from structured array
            self.lines = [
                MoleculeLine(self.molecule_id, {
                    'nr': int(row['nr']),
                    'lev_up': str(row['lev_up']),
                    'lev_low': str(row['lev_low']),
                    'lam': float(row['lam']),
                    'freq': float(row['freq']),
                    'a_stein': float(row['a_stein']),
                    'e_up': float(row['e_up']),
                    'e_low': float(row['e_low']),
                    'g_up': int(row['g_up']),
                    'g_low': int(row['g_low'])
                })
                for row in active_data
            ]

    def _invalidate_caches(self):
        """Invalidate all cached data."""
        self._lines_cache_valid = False
        self._lines_cache = None
        self._wavelengths_cache = None
        self._frequencies_cache = None
        self._a_stein_cache = None
        self._e_up_cache = None
        self._e_low_cache = None
        self._g_up_cache = None
        self._g_low_cache = None
        self._lev_up_cache = None
        self._lev_low_cache = None
        self._pandas_df_cache = None  # Invalidate DataFrame cache too
        self._filtered_raw_data = None  # Force re-filter on next access
        self._quantum_cache.clear()

    # ─────────────────────────────────────────────────────────────────
    # Quantum-state access
    # ─────────────────────────────────────────────────────────────────

    @property
    def quantum_schema(self):
        """Return the :class:`~.QuantumStateSchema.QuantumStateSchema` for this molecule.

        The schema is looked up in :class:`~.QuantumStateSchema.QuantumStateRegistry`
        using :attr:`molecule_id`.  Importing :mod:`~.HITRANQuantumSchemas`
        here (on first access) registers all HITRAN schemas as a side-effect
        while keeping startup cost zero when the feature is not used.
        """
        import iSLAT.Modules.DataTypes.HITRANQuantumSchemas  # noqa: F401 — register side-effect
        from .QuantumStateSchema import QuantumStateRegistry
        return QuantumStateRegistry.get_schema(self.molecule_id)

    def get_quantum_numbers(self, state: str = 'upper') -> dict:
        """Return a dict of named quantum-number arrays for all lines.

        Parsing results are cached per *state* and invalidated together
        with all other caches when line data changes.

        Parameters
        ----------
        state : ``'upper'`` or ``'lower'``
            Which level labels to parse.

        Returns
        -------
        dict[str, numpy.ndarray]
            Mapping of field name → 1-D array aligned with the line data.

        Raises
        ------
        ValueError
            If *state* is not ``'upper'`` or ``'lower'``.
        """
        if state not in ('upper', 'lower'):
            raise ValueError(f"state must be 'upper' or 'lower', got {state!r}")

        if state in self._quantum_cache:
            return self._quantum_cache[state]

        self._ensure_data_loaded()
        col = 'lev_up' if state == 'upper' else 'lev_low'
        if self._raw_lines_data is not None:
            labels = self._raw_lines_data[col]
        else:
            labels = np.array([getattr(l, col) for l in self.lines], dtype='U64')

        result = self.quantum_schema.parse_bulk(labels)
        self._quantum_cache[state] = result
        return result

    def _get_column(self, column_name: str, cache_attr: str) -> np.ndarray:
        """Generic cached column accessor.

        All nine per-column getter methods (``get_wavelengths``,
        ``get_frequencies``, …) delegate here.  The logic is:

        1. Ensure lazy data is loaded.
        2. Return the cached array if present.
        3. Otherwise extract the column from the structured array
           (fast path) or fall back to iterating over ``MoleculeLine``
           objects, cache the result, and return it.

        Parameters
        ----------
        column_name : str
            Name of the field in the structured ``_LINE_DTYPE`` array
            (e.g. ``'lam'``, ``'freq'``, ``'a_stein'``).
        cache_attr : str
            Name of the instance attribute used to cache the result
            (e.g. ``'_wavelengths_cache'``).
        """
        self._ensure_data_loaded()

        cached = getattr(self, cache_attr)
        if cached is not None:
            return cached

        active_data = self._get_active_raw_data()
        if active_data is not None and len(active_data) > 0:
            result = active_data[column_name].copy()
        else:
            self._ensure_lines_created()
            if not self.lines:
                result = np.array([])
            else:
                result = np.array([getattr(line, column_name) for line in self.lines])

        setattr(self, cache_attr, result)
        return result

    def get_ndarray(self) -> np.ndarray:
        """
        Convert the line data to a numpy ndarray.

        Returns
        -------
        np.ndarray
            Numpy array containing the line data.
        """
        self._ensure_data_loaded()
        self._ensure_lines_created()
        if not self.lines:
            return np.array([])
        return np.array([line.get_ndarray() for line in self.lines])
    
    def get_pandas_table(self, include_extras: bool = False) -> "pandas.DataFrame":
        """
        Get all lines as a pandas DataFrame.
        
        Parameters
        ----------
        include_extras : bool, optional
            When *True* and extra_fields are available, they are appended
            as additional columns.  Default is *False* for backward
            compatibility.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all line data
        """
        pd = _get_pandas()
        if pd is None:
            raise ImportError("Pandas required to create table")
            
        self._ensure_data_loaded()
        
        # Create DataFrame directly from structured numpy array if available
        active_data = self._get_active_raw_data()
        if active_data is not None and len(active_data) > 0:
            df = pd.DataFrame(active_data)

            if include_extras and self._extra_fields:
                n_active = len(active_data)
                for col, values in self._extra_fields.items():
                    if len(values) == n_active:
                        df[col] = values
                    elif self._wavelength_range is None and len(values) == len(self._raw_lines_data):
                        df[col] = values

            return df
        
        self._ensure_lines_created()
        if not self.lines:
            return pd.DataFrame()
        
        # Combine all individual line DataFrames efficiently
        line_dfs = [line.get_pandas_table() for line in self.lines]
        return pd.concat(line_dfs, ignore_index=True)
    
    def get_partition_table(self):
        """
        Get partition function as a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing partition function data
        """
        pd = _get_pandas()
        if pd is None:
            raise ImportError("Pandas required to create table")
            
        self._ensure_data_loaded()
        if self.partition_function is None:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'temp': self.partition_function.t,
            'q': self.partition_function.q
        })
    
    @property
    def partition(self):
        """Access to partition function"""
        self._ensure_data_loaded()
        return self.partition_function
    
    def __len__(self) -> int:
        """Return the number of lines in the list."""
        return self.num_lines
    
    @property
    def num_lines(self):
        """Number of lines in the list (respects wavelength_range filter)."""
        self._ensure_data_loaded()
        # Use structured array if available for faster count
        active_data = self._get_active_raw_data()
        if active_data is not None:
            return len(active_data)
        elif self.lines is not None:
            return len(self.lines)
        return 0
    
    # Compatibility properties for legacy code expecting MolData format
    @property 
    def lines_as_namedtuple(self):
        """Get all line data as a single namedtuple structure for compatibility"""
        self._ensure_data_loaded()
        
        # Use cache if valid
        if self._lines_cache_valid and self._lines_cache is not None:
            return self._lines_cache
        
        # Use structured array for fast column access (O(1) slicing)
        active_data = self._get_active_raw_data()
        if active_data is not None and len(active_data) > 0:
            # Direct column access from structured array - no list comprehension needed
            self._lines_cache = self._lines_type(
                active_data['nr'],
                active_data['lev_up'],
                active_data['lev_low'],
                active_data['lam'],
                active_data['freq'],
                active_data['a_stein'],
                active_data['e_up'],
                active_data['e_low'],
                active_data['g_up'],
                active_data['g_low']
            )
        else:
            # Fallback to using MoleculeLine objects
            self._ensure_lines_created()
            if not self.lines:
                self._lines_cache = self._lines_type(
                    np.array([], dtype=np.int32),
                    np.array([], dtype='U64'),
                    np.array([], dtype='U64'),
                    np.array([], dtype=np.float64),
                    np.array([], dtype=np.float64),
                    np.array([], dtype=np.float64),
                    np.array([], dtype=np.float64),
                    np.array([], dtype=np.float64),
                    np.array([], dtype=np.int32),
                    np.array([], dtype=np.int32),
                )
            else:
                line_data = list(zip(*[(line.nr, line.lev_up, line.lev_low, line.lam, line.freq,
                                       line.a_stein, line.e_up, line.e_low, line.g_up, line.g_low) 
                                      for line in self.lines]))
                
                self._lines_cache = self._lines_type(*[np.array(data) for data in line_data])
        
        self._lines_cache_valid = True
        return self._lines_cache
    
    @property
    def name(self):
        """Molecule name for compatibility"""
        return self.molecule_id
    
    def get_wavelengths(self):
        """
        Get wavelengths of all lines as a numpy array.
        
        Returns
        -------
        np.ndarray
            Array of wavelengths in microns
        """
        return self._get_column('lam', '_wavelengths_cache')
    
    def get_frequencies(self):
        """
        Get frequencies of all lines as a numpy array.
        
        Returns
        -------
        np.ndarray
            Array of frequencies in Hz
        """
        return self._get_column('freq', '_frequencies_cache')
    
    def get_einstein_coefficients(self):
        """
        Get all Einstein A coefficients from the lines.
        
        Returns
        -------
        np.ndarray
            Array of Einstein A coefficients
        """
        return self._get_column('a_stein', '_a_stein_cache')
    
    def get_upper_energies(self):
        """
        Get all upper level energies from the lines.
        
        Returns
        -------
        np.ndarray
            Array of upper level energies in K
        """
        return self._get_column('e_up', '_e_up_cache')
    
    def get_lower_energies(self):
        """
        Get all lower level energies from the lines.
        
        Returns
        -------
        np.ndarray
            Array of lower level energies in K
        """
        return self._get_column('e_low', '_e_low_cache')
    
    def get_upper_weights(self):
        """
        Get all upper level statistical weights from the lines.
        
        Returns
        -------
        np.ndarray
            Array of upper level statistical weights
        """
        return self._get_column('g_up', '_g_up_cache')
    
    def get_lower_weights(self):
        """
        Get all lower level statistical weights from the lines.
        
        Returns
        -------
        np.ndarray
            Array of lower level statistical weights
        """
        return self._get_column('g_low', '_g_low_cache')
    
    def get_lines_in_range(self, lam_min: float, lam_max: float):
        """
        Get lines within a wavelength range as MoleculeLine objects.
        
        Parameters
        ----------
        lam_min : float
            Minimum wavelength in microns
        lam_max : float
            Maximum wavelength in microns
            
        Returns
        -------
        list
            List of MoleculeLine objects within the range
        """
        self._ensure_data_loaded()
        
        # Use structured array boolean mask for efficient filtering
        active_data = self._get_active_raw_data()
        if active_data is not None and len(active_data) > 0:
            # Boolean mask indexing is O(n) but very fast with numpy
            lam_values = active_data['lam']
            mask = (lam_values >= lam_min) & (lam_values <= lam_max)
            filtered_raw = active_data[mask]
            
            # Create MoleculeLine objects from filtered structured array
            from iSLAT.Modules.DataTypes.MoleculeLine import MoleculeLine
            return [
                MoleculeLine(self.molecule_id, {
                    'nr': int(row['nr']),
                    'lev_up': str(row['lev_up']),
                    'lev_low': str(row['lev_low']),
                    'lam': float(row['lam']),
                    'freq': float(row['freq']),
                    'a_stein': float(row['a_stein']),
                    'e_up': float(row['e_up']),
                    'e_low': float(row['e_low']),
                    'g_up': int(row['g_up']),
                    'g_low': int(row['g_low'])
                })
                for row in filtered_raw
            ]
        else:
            # Fallback to using existing lines
            self._ensure_lines_created()
            lines_in_range = []
            for line in self.lines:
                if lam_min <= line.lam <= lam_max:
                    lines_in_range.append(line)
            return lines_in_range
    
    def get_ndarray_of_attribute(self, attribute_name):
        """
        Get a numpy array of a specific attribute for all lines.
        
        Parameters
        ----------
        attribute_name : str
            Name of the attribute to extract from each line
            
        Returns
        -------
        np.ndarray
            Array of attribute values
        """
        self._ensure_data_loaded()
        
        # Use structured array column access if the attribute is a field
        active_data = self._get_active_raw_data()
        if active_data is not None and len(active_data) > 0:
            if attribute_name in active_data.dtype.names:
                return active_data[attribute_name].copy()
        
        # Fallback to line objects
        self._ensure_lines_created()
        return np.array([getattr(line, attribute_name) for line in self.lines])

    @property
    def fname(self):
        """File name for compatibility with old MolData interface"""
        return getattr(self, '_filename', None)
    
    @fname.setter
    def fname(self, value):
        """Set file name"""
        self._filename = value

    def enable_parallel_line_loading(self, use_parallel=True):
        """
        Enable or disable parallel line loading for very large molecules.
        This can significantly speed up loading of molecules with >100k lines.
        """
        self._use_parallel_loading = use_parallel
        
    def _load_from_file_parallel(self, filename: str, chunk_size: int = 50000):
        """
        Load molecular data from a .par file using parallel processing for large files.
        
        Parameters
        ----------
        filename : str
            Path to the .par file
        chunk_size : int
            Number of lines to process per chunk
        """
        from iSLAT.Modules.FileHandling.molecular_data_reader import read_molecular_data
        import multiprocessing as mp
        import numpy as np
        
        # First, read the file to determine size
        with open(filename, "r") as f:
            data_raw = f.readlines()
        
        # Count clean lines to estimate size
        data_clean = [line for line in data_raw if line.strip() and not line.strip().startswith("#")]
        
        if len(data_clean) < 3:
            return self._load_from_file(filename)  # Fallback to regular loading
            
        n_partition = int(data_clean[0])
        num_lines = len(data_clean) - n_partition - 2  # Subtract partition function and headers
        
        # Use parallel loading only for large files
        if num_lines < 100000 or not hasattr(self, '_use_parallel_loading') or not self._use_parallel_loading:
            return self._load_from_file(filename)  # Use regular loading
            
        print(f"Using parallel loading for {num_lines} lines...")
        
        # Use regular loading for now - parallel loading can be complex with file I/O
        # This is a placeholder for future implementation
        return self._load_from_file(filename)
    
    def _get_pandas_dataframe(self):
        """Get or create cached pandas DataFrame from structured array."""
        # Fast path: return cached DataFrame if available
        if self._pandas_df_cache is not None:
            return self._pandas_df_cache
        
        # Check if we have raw data to convert
        active_data = self._get_active_raw_data()
        if active_data is None or len(active_data) == 0:
            return None
        
        pd = _get_pandas()
        if pd is None:
            return None
        
        # Create DataFrame directly from structured numpy array - very efficient
        self._pandas_df_cache = pd.DataFrame(active_data)
        
        return self._pandas_df_cache
    
    @property
    def molar_mass(self):
        """Get the molar mass of the molecule."""
        self._ensure_data_loaded()
        return self._molar_mass
    
    def get_upper_levels(self) -> np.ndarray:
        """
        Get all upper level quantum state labels from the lines.
        
        Returns
        -------
        np.ndarray
            Array of upper level quantum state labels
        """
        return self._get_column('lev_up', '_lev_up_cache')
    
    def get_lower_levels(self) -> np.ndarray:
        """
        Get all lower level quantum state labels from the lines.
        
        Returns
        -------
        np.ndarray
            Array of lower level quantum state labels
        """
        return self._get_column('lev_low', '_lev_low_cache')

    # ================================
    # .par File Writing
    # ================================

    def _write_line_data(self, fh, lines_df: Optional["pandas.DataFrame"] = None):
        """Write line data to an open file handle in .par format.

        Writes directly from the internal structured array or from an
        externally supplied DataFrame (e.g. a filtered subset).

        Parameters
        ----------
        fh : file handle
            Writable file handle.
        lines_df : pandas.DataFrame, optional
            A DataFrame with the same columns as ``get_pandas_table()``.
            When provided, these rows are written instead of the full
            internal line list.

        Returns
        -------
        fh : file handle
            The same handle, for chaining.
        """
        if lines_df is not None:
            raw = lines_df.to_records(index=False)
        else:
            self._ensure_data_loaded()
            raw = self._raw_lines_data

        numlines = len(raw) if raw is not None else 0

        fh.write("Number of lines\n")
        fh.write(str(numlines) + '\n')

        fh.write("#    Nr                        Lev_up                       Lev_low   Lambda    Frequency")
        fh.write("       Einstein-A     E_up           E_low        g_up   g_low\n")
        fh.write("#                                                                    [micron]  [GHz]    ")
        fh.write("       [s**-1]        [K]            [K]        \n")

        if raw is not None:
            for row in raw:
                # freq is stored in Hz internally; .par format expects GHz
                freq_ghz = float(row['freq']) / 1e9
                fh.write(
                    "{:6d}{:>30s}{:>30s}{:11.5f}{:15.8f}{:13.4e}{:15.5f}{:15.5f}{:7.1f}{:7.1f}\n".format(
                        int(row['nr']),
                        str(row['lev_up']),
                        str(row['lev_low']),
                        float(row['lam']),
                        freq_ghz,
                        float(row['a_stein']),
                        float(row['e_up']),
                        float(row['e_low']),
                        float(row['g_up']),
                        float(row['g_low']),
                    )
                )

        return fh

    def write_par_file(
        self,
        file_path: str | Path,
        header: Optional["pandas.DataFrame"] = None,
        lines_df: Optional["pandas.DataFrame"] = None,
        partition_df: Optional["pandas.DataFrame"] = None
    ) -> None:
        """Write the line list to a .par file.

        Reuses the existing ``write_partition_function`` writer for the
        partition-function block and writes line data directly from the
        internal structured array (or from *lines_df* when supplied).

        Parameters
        ----------
        file_path : str or Path
            Output path for the .par file.
        header : pandas.DataFrame, optional
            Single-row DataFrame that can override default header fields.
            Recognised columns (all optional): ``'mol'``, ``'M'``,
            ``'isotopologue'``, ``'G'``, ``'molar_mass'``, ``'source'``,
            ``'date'``.  Missing or ``None`` values fall back to defaults
            derived from the line list's own metadata.
        lines_df : pandas.DataFrame, optional
            A filtered DataFrame (same columns as ``get_pandas_table()``).
            When provided, only these rows are written.
        partition_df : pandas.DataFrame, optional
            A DataFrame with columns ``'temp'`` and ``'q'`` for the partition function.
             When provided, this is written instead of the line list's own partition function data.
        """
        import datetime
        from pathlib import Path as _Path
        from iSLAT.Modules.FileHandling.partition_function_writer import write_partition_function

        self._ensure_data_loaded()
        file_path = _Path(file_path)
        os.makedirs(file_path.parent, exist_ok=True)

        # Build header overrides dict
        overrides: dict = {}
        if header is not None and not header.empty:
            overrides = {
                col: header[col].iloc[0]
                for col in header.columns
                if header[col].iloc[0] is not None
            }

        h_mol = overrides.get('mol', self.molecule_id or '')
        h_M = overrides.get('M', '')
        h_iso = overrides.get('isotopologue', '')
        h_G = overrides.get('G', '')
        h_mass = overrides.get('molar_mass', self._molar_mass or '')
        h_source = overrides.get('source', 'Written from MoleculeLineList')
        h_date = overrides.get('date', str(datetime.date.today()))

        with open(file_path, 'w') as fh:
            fh.write(f"# HITRAN {h_mol}; id:{h_M}; iso:{h_iso};gid:{h_G}\n")
            fh.write(f"# Molar Mass: {h_mass}\n")
            fh.write(f"# {h_source}\n")
            fh.write(f"# {h_date}\n")

            partition_function = partition_df if partition_df is not None else self.partition_function

            # Partition function — normalise column names to temp/q
            if partition_function is not None:
                pd = _get_pandas()
                if pd is not None:
                    if type(partition_function) is not pd.DataFrame:
                        qdata = pd.DataFrame({
                            'temp': partition_function.t,
                            'q': partition_function.q,
                        })
                    else:
                        qdata = partition_function
                    fh = write_partition_function(fh, qdata)

            fh = self._write_line_data(fh, lines_df=lines_df)