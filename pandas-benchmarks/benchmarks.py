import pandas as pd
import numpy as np
import time
import xarray as xr

from path import Path

# Context manager for timing operations
class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = time.time() - self.t0

# Wrapper function for benchmarking
def benchmark_wrapper(save_func, load_func, file_path):
    with Timer() as save_timer:
        save_func(file_path)
    with Timer() as load_timer:
        load_func(file_path)
    file_size = Path(file_path).stat().st_size
    Path(file_path).unlink()
    return save_timer.elapsed_time, load_timer.elapsed_time, file_size

def benchmark_csv(df):
    return benchmark_wrapper(
        lambda path: df.to_csv(path, index=False),
        pd.read_csv,
        "pandas-benchmarks/data.csv"
    )

def benchmark_parquet(df):
    return benchmark_wrapper(
        lambda path: df.to_parquet(path, index=False),
        pd.read_parquet,
        "pandas-benchmarks/data.parquet"
    )

def benchmark_feather(df):
    return benchmark_wrapper(
        lambda path: df.to_feather(path),
        pd.read_feather,
        "pandas-benchmarks/data.feather"
    )

def benchmark_pickle(df):
    return benchmark_wrapper(
        lambda path: df.to_pickle(path),
        pd.read_pickle,
        "pandas-benchmarks/data.pickle"
    )

def benchmark_npy(df):
    return benchmark_wrapper(
        lambda path: np.save(path, df),
        np.load,
        "pandas-benchmarks/data.npy"
    )

def benchmark_hdf5(df):
    return benchmark_wrapper(
        lambda path: df.to_hdf(path, key="data", mode="w"),
        pd.read_hdf,
        "pandas-benchmarks/data.hdf5"
    )

def benchmark_json(df):
    return benchmark_wrapper(
        lambda path: df.to_json(path),
        pd.read_json,
        "pandas-benchmarks/data.json"
    )

def benchmark_h5netcdf(df):
    return benchmark_wrapper(
        lambda path: df.to_xarray().to_netcdf(path, engine="h5netcdf"),
        lambda path: xr.open_dataset(path, engine='h5netcdf').to_dataframe(),
        "pandas-benchmarks/data.nc"
    )

def benchmark_excel(df):
    return benchmark_wrapper(
        lambda path: df.to_excel(path, index=False),
        pd.read_excel,
        "pandas-benchmarks/data.xlsx"
    )

if __name__ == "__main__":
    # Create a random dataset
    n_rows = 100000
    n_columns = 10
    data = np.random.rand(n_rows, n_columns)
    columns = [f"col_{i}" for i in range(n_columns)]
    df = pd.DataFrame(data, columns=columns)
    functions_dict = {
        "csv": benchmark_csv,
        "parquet": benchmark_parquet,
        "feather": benchmark_feather,
        "pickle": benchmark_pickle,
        "npy": benchmark_npy,
        "hdf5": benchmark_hdf5,
        "json": benchmark_json,
        "h5netcdf": benchmark_h5netcdf,
        "excel": benchmark_excel}
    with open("pandas-benchmarks/result.txt", "w") as f:
        f.write("Format - Save time in sec / Load time in sec / Size in MB\n")
        for fmt in ["csv", "parquet", "feather", "pickle", "npy", "hdf5", "json", "h5netcdf", "excel"]:
            if fmt not in functions_dict:
                f.write(f"{fmt} - not supported\n")
                continue
            save, load, size = functions_dict[fmt](df)
            f.write(f"{fmt} - {load:.2f}s / {save:.2f}s / {size/(1024 * 1024):.2f}\n")
