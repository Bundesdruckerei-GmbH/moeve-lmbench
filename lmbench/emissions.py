"""Tracks emissions."""

import pandas as pd
from codecarbon import OfflineEmissionsTracker

from lmbench.config.config import OUTPUT_FOLDER
from lmbench.utils import is_gpu_environment


class EmissionTracker:
    """A wrapper around code carbons OfflineEmissionsTracker that also tracks cache hits."""

    def __init__(self, experiment_name: str):
        """Initializes the tracker and immediately starts tracking.

        Args:
            experiment_name (str): The name of the experiment to track.
        """
        if is_gpu_environment():
            self.tracker = OfflineEmissionsTracker(
                country_iso_code="DEU",
                measure_power_secs=5,
                save_to_file=False,
                experiment_id=experiment_name,
                log_level="ERROR",
                tracking_mode="machine",
            )
            self.output_file = OUTPUT_FOLDER / "emissions.csv"
            self.tracker.start()

    def stop(self, cache_hits: int, dataset_length: int) -> None:
        """Stops tracking and saves emissions.csv in OUTPUT_FOLDER.

        Args:
            cache_hits (int): The number of cache hits that should be added to the output.
            dataset_length (int): The length of the dataset that should be used for a relative cache rate.
        ```
        """
        if not is_gpu_environment():
            return
        self.tracker.stop()
        data = self.tracker.final_emissions_data.values
        data["cache_hits"] = cache_hits
        data["relative_cache_rate"] = cache_hits / dataset_length if dataset_length > 0 else 0

        if self.output_file.exists():
            df = pd.read_csv(self.output_file)
        else:
            df = pd.DataFrame(columns=list(data.keys()))
        df.loc[len(df)] = data
        df.to_csv(self.output_file, index=False)
