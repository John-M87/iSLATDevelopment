import numpy as np

#: Most decimals ever written for a temperature.
_MAX_TEMPERATURE_DECIMALS = 6


def _temperature_decimals(temps) -> int:
    """Fewest decimals (at least 1) that write every temperature exactly."""
    temps = np.asarray(temps, dtype=np.float64)
    for decimals in range(1, _MAX_TEMPERATURE_DECIMALS + 1):
        if np.allclose(np.round(temps, decimals), temps, rtol=0.0, atol=1e-9):
            return decimals
    return _MAX_TEMPERATURE_DECIMALS


def write_partition_function(fh, qdata):
    """
    Write partition function data to a file.

    Temperatures are written with one decimal, or with as many more (up to
    six) as needed to keep fractional temperature steps exact.

    Parameters
    ----------
    fh : file handle
        The file handle to write the data to.
    qdata : DataFrame
        A pandas DataFrame containing temperature and partition function data.

    Returns
    -------
    fh : file handle
        The file handle after writing the data.
    """
    fh.write("# Number of Partition function entries\n")
    fh.write("{:d}\n".format(len(qdata['temp'])))

    fh.write("# Temperature    Q(T)\n")
    fh.write("# [K]\n")

    decimals = _temperature_decimals(qdata['temp'])
    row_format = "{:>%d.%df}  {:>15.6f}\n" % (5 + decimals, decimals)
    for t, q in zip(qdata['temp'], qdata['q']):
        fh.write(row_format.format(t, q))

    return fh


def write_partition_function_csv(path, temps, q):
    """
    Write a partition function table to a CSV file with columns ``temp,q``.

    Parameters
    ----------
    path : str or Path
        Output file path.
    temps, q : array-like
        Temperatures [K] and partition function values.

    Returns
    -------
    path : str or Path
        The path written.
    """
    import pandas as pd
    pd.DataFrame({'temp': np.asarray(temps, dtype=np.float64),
                  'q': np.asarray(q, dtype=np.float64)}).to_csv(path, index=False)
    return path
