import pandas as pd
#!/usr/bin/env python3
"""
Simple Data API for Converted Neural Data

This API reads the converted simple format files without requiring pynwb.
Only requires numpy, pandas, and standard library.

Data Fields Available:
---------------------
Two task types in dataset:
  - CO: Center-Out reaching (8 targets in circle, one reach per trial)
  - RT: Random Target sequential reaching (multiple reaches to same target)

Common columns in all trials:
  - trial_index: Sequential trial number (0-based)
  - start_time: Trial start (seconds)
  - stop_time: Trial end (seconds)
  - target_dir: Target direction in radians
  - result: Trial outcome
    * 'R': Reward (successful)
    * 'A': Abort (trial ended early)
    * 'I': Incorrect (wrong target)
    * 'F': Failed (timeout)

CO (Center-Out) specific columns:
  - target_id: Integer 0-7 identifying which of 8 targets
  - target_dir: One of 8 directions in radians:
    * 0: 0.0 rad (0°, rightward)
    * 1: 0.785 rad (45°)
    * 2: 1.571 rad (90°, upward)
    * 3: 2.356 rad (135°)
    * 4: 3.142 rad (180°, leftward)
    * 5: -2.356 rad (-135°)
    * 6: -1.571 rad (-90°, downward)
    * 7: -0.785 rad (-45°)
  - target_corners: List [x1, y1, x2, y2] defining target box
  - target_on_time: When target appeared (seconds)
  - go_cue_time: Single go cue time (seconds)

RT (Random Target) specific columns:
  - target_id: Array of target IDs (typically [1, 1, 1, 1])
  - target_dir: Fixed at 0.785 rad (45°) for all targets
  - num_targets: Number of sequential targets (always 4)
  - num_attempted: Number of targets monkey attempted
  - go_cue_time_array: Array of go cue times for each target
  - target_size: Size of target

units DataFrame columns:
  - unit_id: Sequential unit identifier (0-based)
  Note: brain_area and electrode_idx not reliably extracted from all sessions

spike_times dict:
  - Keys: 'unit_0', 'unit_1', etc.
  - Values: numpy arrays of spike timestamps (seconds)

behavior dict:
  - position_data: (n_timepoints, 2) array of x,y hand position
  - position_timestamps: (n_timepoints,) array of timestamps
  - velocity_data: (n_timepoints, 2) array of x,y hand velocity
  - velocity_timestamps: (n_timepoints,) array of timestamps
  - acceleration_data: (n_timepoints, 2) array of x,y hand acceleration
  - acceleration_timestamps: (n_timepoints,) array of timestamps
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class Session:
    """Container for a converted session's data"""
    session_name: str
    metadata: Dict
    trials: pd.DataFrame
    units: pd.DataFrame
    spike_times: Dict[str, np.ndarray]
    behavior: Dict[str, np.ndarray]
    manifest: Dict

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def subject_id(self) -> str:
        return self.metadata.get('subject_id', 'unknown')

    @property
    def session_date(self) -> str:
        return self.metadata.get('session_start', 'unknown')


class DataReader:
    """
    Simple API for reading converted neural data.

    This reader provides easy access to:
    - Trial information (CSV)
    - Neural spike times (NPZ)
    - Behavioral data (NPZ)
    - Session metadata (JSON)

    No complex dependencies required!
    """

    def __init__(self, data_dir: Union[str, Path] = "/home/gyokang/monkey_data"):
        """
        Initialize reader with converted data directory.

        Args:
            data_dir: Path to directory containing converted sessions
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory {data_dir} does not exist")

        # Cache for loaded sessions
        self._cache = {}

    def list_sessions(self) -> List[str]:
        """List all available converted sessions."""
        sessions = []
        for session_dir in sorted(self.data_dir.iterdir()):
            if session_dir.is_dir() and (session_dir / 'manifest.json').exists():
                sessions.append(session_dir.name)
        return sessions

    def load_session(self, session_name: str, cache: bool = True) -> Session:
        """
        Load a converted session.

        Args:
            session_name: Name of session directory
            cache: Whether to cache the loaded session

        Returns:
            Session object with all data
        """
        # Check cache
        if cache and session_name in self._cache:
            return self._cache[session_name]

        session_dir = self.data_dir / session_name
        if not session_dir.exists():
            raise ValueError(f"Session {session_name} not found")

        # Load manifest
        with open(session_dir / 'manifest.json', 'r') as f:
            manifest = json.load(f)

        # Load metadata
        with open(session_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        # Load trials
        trials_file = session_dir / manifest['data_files'].get('trials', 'trials.csv')
        trials = pd.read_csv(trials_file) if trials_file.exists() else pd.DataFrame()

        # Load units
        units_file = session_dir / manifest['data_files'].get('units', 'units.csv')
        units = pd.read_csv(units_file) if units_file.exists() else pd.DataFrame()

        # Load spike times
        spike_times = {}
        spikes_file = session_dir / manifest['data_files'].get('spike_times', 'spike_times.npz')
        if spikes_file.exists():
            spike_data = np.load(spikes_file)
            spike_times = {key: spike_data[key] for key in spike_data.files}

        # Load behavior
        behavior = {}
        behavior_file = session_dir / manifest['data_files'].get('behavior', 'behavior.npz')
        if behavior_file.exists():
            behavior_data = np.load(behavior_file)
            behavior = {key: behavior_data[key] for key in behavior_data.files}

        # Create session object
        session = Session(
            session_name=session_name,
            metadata=metadata,
            trials=trials,
            units=units,
            spike_times=spike_times,
            behavior=behavior,
            manifest=manifest
        )

        # Cache if requested
        if cache:
            self._cache[session_name] = session

        return session

    def get_trial_spikes(self, session: Session, trial_idx: int,
                        bin_size: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get binned spike counts for a specific trial.

        Args:
            session: Loaded session
            trial_idx: Trial index
            bin_size: Bin size in seconds

        Returns:
            spike_counts: (n_bins, n_units) array
            time_bins: (n_bins,) array of bin centers
        """
        if trial_idx >= len(session.trials):
            raise ValueError(f"Trial index {trial_idx} out of range")

        trial = session.trials.iloc[trial_idx]
        start_time = trial['start_time']
        stop_time = trial['stop_time']

        # Create time bins
        time_bins = np.arange(start_time, stop_time + bin_size, bin_size)
        bin_centers = time_bins[:-1] + bin_size / 2

        n_bins = len(bin_centers)
        n_units = session.n_units
        spike_counts = np.zeros((n_bins, n_units))

        # Bin spikes for each unit
        for unit_idx in range(n_units):
            unit_key = f'unit_{unit_idx}'
            if unit_key in session.spike_times:
                spikes = session.spike_times[unit_key]
                # Find spikes in trial window
                trial_spikes = spikes[(spikes >= start_time) & (spikes <= stop_time)]
                # Bin them
                counts, _ = np.histogram(trial_spikes, bins=time_bins)
                spike_counts[:, unit_idx] = counts

        return spike_counts, bin_centers

    def get_trial_behavior(self, session: Session, trial_idx: int,
                          signal: str = 'position') -> Optional[np.ndarray]:
        """
        Get behavioral data for a specific trial.

        Args:
            session: Loaded session
            trial_idx: Trial index
            signal: Type of signal ('position', 'velocity', 'acceleration')

        Returns:
            Behavioral data array or None if not available
        """
        if trial_idx >= len(session.trials):
            raise ValueError(f"Trial index {trial_idx} out of range")

        trial = session.trials.iloc[trial_idx]
        start_time = trial['start_time']
        stop_time = trial['stop_time']

        # Check if signal exists
        data_key = f'{signal}_data'
        time_key = f'{signal}_timestamps'

        if data_key not in session.behavior or time_key not in session.behavior:
            return None

        data = session.behavior[data_key]
        timestamps = session.behavior[time_key]

        # Find data within trial window
        mask = (timestamps >= start_time) & (timestamps <= stop_time)
        return data[mask]

    def filter_units_by_area(self, session: Session, brain_area: str) -> List[int]:
        """Get unit indices for a specific brain area."""
        if 'brain_area' not in session.units.columns:
            return []

        area_units = session.units[session.units['brain_area'] == brain_area]
        return area_units['unit_id'].tolist()

    def get_aligned_data(self, session: Session,
                        align_event: str = 'go_cue_time',
                        time_window: Tuple[float, float] = (-0.5, 1.0),
                        bin_size: float = 0.01,
                        trials: Optional[List[int]] = None) -> Dict:
        """
        Get trial-aligned neural and behavioral data.

        Args:
            session: Loaded session
            align_event: Column name in trials DataFrame to align to
            time_window: Time window relative to alignment (seconds)
            bin_size: Bin size for neural data
            trials: List of trial indices (None = all)

        Returns:
            Dictionary with aligned data
        """
        if align_event not in session.trials.columns:
            raise ValueError(f"Align event '{align_event}' not found in trials")

        if trials is None:
            trials = list(range(len(session.trials)))

        aligned_spikes = []
        aligned_position = []
        trial_info = []

        for trial_idx in trials:
            trial = session.trials.iloc[trial_idx]

            # Skip if alignment time is missing
            if pd.isna(trial[align_event]):
                continue

            align_time = trial[align_event]

            # Get spike data for full trial
            spike_counts, time_bins = self.get_trial_spikes(session, trial_idx, bin_size)

            # Find bins within alignment window
            relative_times = time_bins - align_time
            mask = (relative_times >= time_window[0]) & (relative_times <= time_window[1])

            if np.any(mask):
                aligned_spikes.append(spike_counts[mask])

                # Get aligned behavior if available
                behavior = self.get_trial_behavior(session, trial_idx, 'position')
                if behavior is not None:
                    # Simple alignment - would need interpolation for exact alignment
                    aligned_position.append(behavior)

                # Store trial info
                trial_info.append({
                    'trial_idx': trial_idx,
                    'target_id': trial.get('target_id', -1),
                    'result': trial.get('result', 'unknown'),
                    'target_dir': trial.get('target_dir', -1)
                })

        return {
            'spike_counts': aligned_spikes,  # List of arrays
            'position': aligned_position,
            'trial_info': pd.DataFrame(trial_info),
            'time_window': time_window,
            'align_event': align_event
        }

    def compute_firing_rates(self, spike_counts: np.ndarray,
                           bin_size: float = 0.01,
                           smooth_sigma: Optional[float] = 0.05) -> np.ndarray:
        """
        Compute firing rates from spike counts.

        Args:
            spike_counts: (n_bins, n_units) array
            bin_size: Bin size in seconds
            smooth_sigma: Gaussian smoothing sigma (seconds), None for no smoothing

        Returns:
            Firing rates in Hz
        """
        # Convert to Hz
        rates = spike_counts / bin_size

        # Smooth if requested
        if smooth_sigma is not None:
            from scipy.ndimage import gaussian_filter1d
            sigma_bins = smooth_sigma / bin_size
            rates = gaussian_filter1d(rates, sigma_bins, axis=0)

        return rates


# Convenience functions

def load_session(session_name: str, data_dir: str = "/home/gyokang/monkey_data") -> Session:
    """Quick function to load a session."""
    reader = DataReader(data_dir)
    return reader.load_session(session_name)


def load_multiple_sessions(session_names: List[str],
                         data_dir: str = "/home/gyokang/monkey_data") -> Dict[str, Session]:
    """Load multiple sessions at once."""
    reader = DataReader(data_dir)
    sessions = {}
    for name in session_names:
        print(f"Loading {name}...")
        sessions[name] = reader.load_session(name)
    return sessions


# Example usage
if __name__ == "__main__":
    # Example of using the API
    reader = DataReader()

    # List available sessions
    
    sessions = np.array(reader.list_sessions())
    mask_co = np.char.find(sessions, "CO") != -1
    sessions = sessions[mask_co]

    print(f"Available sessions: {sessions}")

    if sessions is not None:
        rand = np.random.RandomState(42)
        idxs = rand.choice(len(sessions), 5, replace=False)

        # Load first session
        # session = reader.load_session(sessions[1])
        picked_sessions = sessions[idxs]
        for k, session in zip(idxs, picked_sessions):
            
            session = reader.load_session(session)
            """
            Fetch multiple, RANDOM sessions -- only CO tasks.
            """

            print(f"\nLoaded session: {session.session_name}")
            print(f"Subject: {session.subject_id}")
            print(f"Trials: {session.n_trials}")
            print(f"Units: {session.n_units}")

            # Get brain areas
            if 'brain_area' in session.units.columns:
                areas = session.units['brain_area'].value_counts()
                print("\nUnits by brain area:")
                print(areas)

            # Get spike data for first trial
            if session.n_trials > 0:

                import pandas as pd
                import numpy as np

                aligned_data = reader.get_aligned_data(session, 'go_cue_time', (-0.5, 1), 0.01, None)

                bin_size = 0.01
                time_window = aligned_data["time_window"]
                time_bins = np.arange(time_window[0], time_window[1], bin_size)  # relative time

                all_trials = []
                all_rows = []

                for trial_idx, spikes in enumerate(aligned_data["spike_counts"]):
                    trial_meta = aligned_data["trial_info"].iloc[trial_idx].to_dict()
                    # firing_rate = reader.compute_firing_rates(spikes, 0.01, 0.05)
                    n_bins, n_units = spikes.shape
                    for b in range(n_bins):
                        row = {
                            "trial_idx": trial_meta["trial_idx"],
                            # "target_id": trial_meta["target_id"],
                            "result": trial_meta["result"],
                            "target_dir": trial_meta["target_dir"],
                            "time": time_bins[b]
                        }
                        # add unit firing counts
                        for u in range(n_units):
                            row[f"unit_{u}"] = spikes[b, u]
                            # row[f"unit_{u}_fr"] = firing_rate[b, u]
                        all_rows.append(row)
                df = pd.DataFrame(all_rows)
                df.to_csv(f"/home/gyokang/Sessions/sub-{session.subject_id}_idx-{k}_aligned.csv", index=False)
                print(df.shape)

