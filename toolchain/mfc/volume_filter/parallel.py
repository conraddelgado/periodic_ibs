"""Independent snapshot workers with bounded, array-at-a-time moment transfer."""

import multiprocessing as mp
import traceback
from multiprocessing.connection import wait

from .maps import Moments

ARRAYS = ("count", "weight", "mean", "m2", "m3", "m4")


def _worker(config, root, indices, interval, worker_id, connection):
    from .__main__ import process_snapshots

    try:
        moments, records = process_snapshots(config, root, indices, interval, worker_id)
        keys = sorted(moments)
        connection.send(("result", records, [(key, moments[key].mean.shape) for key in keys]))
        for key in keys:
            value = moments.pop(key)
            for name in ARRAYS:
                connection.send_bytes(memoryview(getattr(value, name)).cast("B"))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def run_workers(config, root, workers, interval):
    """Return merged moments; no worker writes files or pickles large arrays."""
    context = mp.get_context("spawn")
    processes, pending = [], {}
    moments, records = {}, []
    try:
        for worker_id in range(workers):
            receive, send = context.Pipe(duplex=False)
            indices = range(worker_id, len(config["snapshots"]), workers)
            process = context.Process(target=_worker, args=(config, root, indices, interval, worker_id, send))
            process.start()
            send.close()
            processes.append(process)
            pending[receive] = worker_id
        while pending:
            connection = wait(list(pending))[0]
            worker_id = pending[connection]
            try:
                message = connection.recv()
            except EOFError as error:
                raise RuntimeError(f"Snapshot worker {worker_id} exited before returning statistics") from error
            if message[0] == "error":
                raise RuntimeError(f"Snapshot worker {worker_id} failed:\n{message[1]}")
            _, local_records, descriptors = message
            print(f"Merging snapshot worker {worker_id}: {len(local_records)} snapshots", flush=True)
            for key, shape in descriptors:
                value = Moments(shape)
                for name in ARRAYS:
                    array = getattr(value, name)
                    if connection.recv_bytes_into(memoryview(array).cast("B")) != array.nbytes:
                        raise RuntimeError(f"Invalid moment transfer from snapshot worker {worker_id}")
                if key in moments:
                    moments[key].merge(value, workers=config.get("fft_workers", 1))
                else:
                    moments[key] = value
            records.extend(local_records)
            connection.close()
            del pending[connection]
        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"Snapshot worker exited with status {process.exitcode}")
    finally:
        for connection in pending:
            connection.close()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()
    return moments, records
