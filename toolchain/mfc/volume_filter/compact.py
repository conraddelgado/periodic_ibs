"""Convert completed version-2 statistics to the four-statistic output format.

PYTHONPATH=toolchain python -m mfc.volume_filter.compact old.npz compact.npz
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from .__main__ import KEY_FORMAT, STATISTICS, VERSION, atomic_save


def convert(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"{output} exists; select a new output path")
    pattern = re.compile(r"^sigma(\d+)\.(.+)\.pooled\.(mean|variance|skewness|kurtosis)$")
    with np.load(source, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        statistics = metadata.get("statistics", {})
        if metadata.get("version") != 2 or statistics.get("pooled_weighting") != "equal_realizations":
            raise ValueError("Conversion requires version-2 equally weighted pooled statistics")
        if metadata["completed"] != len(metadata["config"]["snapshots"]):
            raise ValueError("Source statistics are incomplete; only completed runs can be converted")
        groups = {}
        entries = []
        for key in data.files:
            match = pattern.fullmatch(key)
            if match is None:
                continue
            index, name, statistic = match.groups()
            group = (name, int(index))
            groups.setdefault(group, set()).add(statistic)
            entries.append((key, KEY_FORMAT.format(field=name, statistic=statistic, filter_index=int(index))))
        if not groups or any(values != set(STATISTICS) for values in groups.values()):
            raise ValueError("Source must contain all four pooled statistics for every field/filter")
        metadata.update(version=VERSION, output_mode="final_only", key_format=KEY_FORMAT)
        statistics.pop("snapshot_weighting", None)
        statistics["outputs"] = list(STATISTICS)

        def arrays():
            yield "metadata", np.array(json.dumps(metadata, sort_keys=True))
            for old, new in entries:
                yield new, data[old]

        # Read/write one selected array at a time, without loading the archive.
        atomic_save(output, arrays())
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(f"Saved compact statistics: {convert(args.source, args.output)}")


if __name__ == "__main__":
    main()
