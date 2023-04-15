import numpy as np
from typing import Any, List
from pathlib import Path

from infer_subc.core.img import apply_mask

import pandas as pd
from infer_subc.utils.stats import _assert_uint16_labels

from .stats import get_aXb_stats_3D, get_summary_stats_3D, get_simple_stats_3D


def shell_cross_stats(
    organelle_names: List[str], organelles: List[np.ndarray], mask: np.ndarray, out_data_path: Path, source_file: str
) -> int:
    """
    get all cross stats between organelles `a` and `b`, and "shell of `a`" and `b`.   "shell" is the boundary of `a`
    calls `get_aXb_stats_3D`
    """
    count = 0
    for j, target in enumerate(organelle_names):
        print(f"getting stats for A = {target}")
        a = organelles[j]
        # loop over Bs
        for i, nmi in enumerate(organelle_names):
            if i != j:
                # get overall stats of intersection
                print(f"  X {nmi}")
                b = organelles[i]
                stats_tab = get_aXb_stats_3D(a, b, mask)
                csv_path = out_data_path / f"{source_file.stem}-{target}X{nmi}-stats.csv"
                stats_tab.to_csv(csv_path)

                e_stats_tab = get_aXb_stats_3D(a, b, mask, use_shell_a=True)
                csv_path = out_data_path / f"{source_file.stem}-{target}_shellX{nmi}-stats.csv"
                e_stats_tab.to_csv(csv_path)

                count += 1
    return count


def organelle_stats(
    organelle_names: List[str],
    organelles: List[np.ndarray],
    intinsities: List[np.ndarray],
    mask: np.ndarray,
    out_data_path: Path,
    source_file: str,
) -> int:
    """
    get summary and all cross stats between organelles `a` and `b`
    calls `get_summary_stats_3D`
    """
    count = 0
    org_stats_tabs = []
    for j, target in enumerate(organelle_names):
        print(f"getting stats for A = {target}")
        a = organelles[j]
        # A_stats_tab, rp = get_simple_stats_3D(A,mask)
        a_stats_tab, rp = get_summary_stats_3D(a, intinsities[j], mask)

        # loop over Bs
        for i, nmi in enumerate(organelle_names):
            if i != j:
                # get overall stats of intersection
                print(f"  b = {nmi}")
                count += 1
                # add the list of touches
                b = _assert_uint16_labels(organelles[i])

                ov = []
                b_labs = []
                labs = []
                for idx, lab in enumerate(a_stats_tab["label"]):  # loop over A_objects
                    xyz = tuple(rp[idx].coords.T)
                    cmp_org = b[xyz]

                    # total number of overlapping pixels
                    overlap = sum(cmp_org > 0)
                    # overlap?
                    labs_b = cmp_org[cmp_org > 0]
                    b_js = np.unique(labs_b).tolist()

                    # if overlap > 0:
                    labs.append(lab)
                    ov.append(overlap)
                    b_labs.append(b_js)

                # add organelle B columns to A_stats_tab
                a_stats_tab[f"{nmi}_overlap"] = ov
                a_stats_tab[f"{nmi}_labels"] = b_labs  # might want to make this easier for parsing later

        # org_stats_tabs.append(A_stats_tab)
        csv_path = out_data_path / f"{source_file.stem}-{target}-stats.csv"
        a_stats_tab.to_csv(csv_path)

    print(f"dumped {count} csvs")
    return count


def dump_stats(
    name: str,
    segmentation: np.ndarray,
    intensity_img: np.ndarray,
    mask: np.ndarray,
    out_data_path: Path,
    source_file: str,
) -> pd.DataFrame:
    """
    get summary stats of organelle only
    calls `get_summary_stats_3D`
    """

    stats_table, _ = get_summary_stats_3D(segmentation, intensity_img, mask)
    csv_path = out_data_path / f"{source_file.stem}-{name}-basicstats.csv"
    stats_table.to_csv(csv_path)
    print(f"dumped {name} table to {csv_path}")

    return stats_table


# refactor to just to a target vs. list of probes
# for nuclei mask == cellmask
# for all oother mask == cytoplasm
