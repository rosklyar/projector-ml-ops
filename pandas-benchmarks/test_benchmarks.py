from pathlib import Path
import benchmarks
import numpy as np
import pandas as pd

n_rows = 100
n_columns = 10
data = np.random.rand(n_rows, n_columns)
columns = [f"col_{i}" for i in range(n_columns)]
df = pd.DataFrame(data, columns=columns)

def test_csv_benchmarking():
    _, _, size = benchmarks.benchmark_csv(df)
    assert size > 0

def test_feather_benchmarking():
    _, _, size = benchmarks.benchmark_feather(df)
    assert size > 0

def test_pickle_benchmarking():
    _, _, size = benchmarks.benchmark_pickle(df)
    assert size > 0

def test_npy_benchmarking():
    _, _, size = benchmarks.benchmark_npy(df)
    assert size > 0

def test_hdf5_benchmarking():
    _, _, size = benchmarks.benchmark_hdf5(df)
    assert size > 0

def test_json_benchmarking():
    _, _, size = benchmarks.benchmark_json(df)
    assert size > 0

def test_h5netcdf_benchmarking():
    _, _, size = benchmarks.benchmark_h5netcdf(df)
    assert size > 0

def test_excel_benchmarking():
    _, _, size = benchmarks.benchmark_excel(df)
    assert size > 0
    