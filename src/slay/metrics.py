import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, find_peaks_cwt, sosfiltfilt
from scipy.stats import poisson, wasserstein_distance
from spikeinterface.core import SortingAnalyzer
from tqdm import tqdm
from joblib import Parallel, delayed


def compute_final_metric(
    similarity: NDArray[np.floating],
    ccg_metric: NDArray[np.floating],
    refractory_penalty: NDArray[np.floating],
    merge_parameters: dict[str, float],
) -> NDArray[np.floating]:
    """
    Compute the final merge metric combining similarity, CCG metric, and refractory penalty.

    Parameters
    ----------
    similarity : NDArray[np.floating]
        Pairwise similarity matrix between units.
    ccg_metric : NDArray[np.floating]
        Cross-correlogram significance metric matrix.
    refractory_penalty : NDArray[np.floating]
        Refractory period violation penalty matrix.
    merge_parameters : dict[str, float]
        Dictionary containing weights "k1" and "k2" for ccg_metric and refractory_penalty.

    Returns
    -------
    NDArray[np.floating]
        The final merge metric matrix.
    """
    final_metric = (
        similarity
        + merge_parameters["k1"] * ccg_metric
        - merge_parameters["k2"] * refractory_penalty
    )
    np.fill_diagonal(final_metric, 0)

    return final_metric


def compute_ccg_metric(
    sorting_analyzer: SortingAnalyzer,
    ccgs: NDArray[np.floating],
    bin_size_ms: float,
    pair_mask: NDArray[np.bool_],
) -> NDArray[np.floating]:
    """
    Compute the cross-correlogram significance metric for all unit pairs.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer containing unit information.
    ccgs : NDArray[np.floating]
        3D array of cross-correlograms with shape (n_units, n_units, n_bins).
    bin_size_ms : float
        The bin size of the correlograms in milliseconds.
    pair_mask : NDArray[np.bool_]
        Boolean mask indicating which unit pairs to compute metrics for.

    Returns
    -------
    NDArray[np.floating]
        Matrix of cross-correlogram significance metrics for each unit pair.
    """
    unit_ids = sorting_analyzer.unit_ids
    ccg_metric = np.zeros((len(unit_ids), len(unit_ids)))

    pairs = np.argwhere(pair_mask)
    bin_size_s = bin_size_ms / 1000

    def compute_pair(i, j):
        return (
            i,
            j,
            _compute_ccg_metric_pair(ccgs[i, j, :], bin_size_s, min_ccg_rate=1000),
        )

    results = Parallel(n_jobs=-1)(
        delayed(compute_pair)(i, j)
        for i, j in tqdm(pairs, desc="Calculating CCG significance")
    )

    for i, j, metric in results:
        ccg_metric[i, j] = metric
        ccg_metric[j, i] = metric

    return ccg_metric


def compute_refractory_penalty(
    sorting_analyzer: SortingAnalyzer,
    ccgs: NDArray[np.floating],
    bin_size_ms: float,
    maximum_contamination: float,
    pair_mask: NDArray[np.bool_],
    noise_margin_z: float = 2.0,
) -> NDArray[np.floating]:
    """
    Compute the refractory period violation penalty for all unit pairs.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer containing unit information.
    ccgs : NDArray[np.floating]
        3D array of auto/cross-correlograms with shape (n_units, n_units, n_bins).
    bin_size_ms : float
        The bin size of the correlograms in milliseconds.
    maximum_contamination : float
        The maximum acceptable contamination threshold.
    pair_mask : NDArray[np.bool_]
        Boolean mask indicating which unit pairs to compute penalties for.
    noise_margin_z : float, default: 2.0
        Number of Poisson standard deviations added to the expected count
        threshold.  Prevents high-firing-rate neurons from being penalized
        for statistically insignificant excursions above the contamination
        threshold.

    Returns
    -------
    NDArray[np.floating]
        Matrix of refractory period violation penalties for each unit pair.
    """
    unit_ids = sorting_analyzer.unit_ids
    refractory_penalty = np.zeros((len(unit_ids), len(unit_ids)))

    pairs = np.argwhere(pair_mask)

    def compute_pair(i, j):
        return (
            i,
            j,
            _sliding_RP_viol_pair(
                ccgs[i, j, :], bin_size_ms, maximum_contamination, noise_margin_z,
            ),
        )

    results = Parallel(n_jobs=-1)(
        delayed(compute_pair)(i, j)
        for i, j in tqdm(pairs, desc="Calculating refractory penalty")
    )

    for i, j, penalty in results:
        refractory_penalty[i, j] = penalty
        refractory_penalty[j, i] = penalty

    return refractory_penalty


def _compute_ccg_metric_pair(
    ccg: NDArray[np.floating],
    bin_size_s: float,
    min_ccg_rate: float,
) -> float:
    """
    Calculates a cross-correlation significance metric for a cluster pair.

    Uses the wasserstein distance between an observed cross-correlogram and a null
    distribution as an estimate of how significant the dependence between
    two neurons is. Low spike count cross-correlograms have large wasserstein
    distances from null by chance, so we first try to expand the window size. If
    that fails to yield enough spikes, we apply a penalty to the metric.

    Parameters
    ----------
    ccg : NDArray[np.floating]
        The raw cross-correlogram for the cluster pair.
    bin_size_s : float
        The width in seconds of the bin size of the input ccgs.
    min_ccg_rate : float
        The minimum ccg firing rate in Hz.

    Returns
    -------
    metric : float
        The calculated cross-correlation significance metric.
    """
    # calculate low-pass filtered second derivative of ccg
    fs = 1 / bin_size_s
    cutoff_freq = 100
    nyquist = fs / 2
    cutoff = cutoff_freq / nyquist
    peak_width = 0.002 / bin_size_s

    ccg_double_derivative = np.diff(ccg, 2)
    sos = butter(4, cutoff, output="sos")
    ccg_double_derivative = sosfiltfilt(sos, ccg_double_derivative)

    if ccg.sum() == 0:
        return 0

    # find negative peaks of second derivative of ccg, these are the edges of dips in ccg
    peaks = find_peaks_cwt(-ccg_double_derivative, peak_width, noise_perc=90) + 1
    # if no peaks are found, return a very low significance
    if peaks.shape[0] == 0:
        return -4
    peaks = np.abs(peaks - ccg.shape[0] / 2)
    peaks = peaks[peaks > 0.5 * peak_width]
    min_peaks = np.sort(peaks)

    # start with peaks closest to 0 and move to the next set of peaks if the event count is too low
    window_width = min_peaks * 1.5
    starts = np.maximum(ccg.shape[0] / 2 - window_width, 0)
    ends = np.minimum(ccg.shape[0] / 2 + window_width, ccg.shape[0] - 1)
    ind = 0
    ccg_windowed = ccg[int(starts[0]) : int(ends[0] + 1)]
    ccg_windowed_sum = ccg_windowed.sum()
    window_size = ccg_windowed.shape[0] * bin_size_s

    while (ccg_windowed_sum < (min_ccg_rate * window_size * 10)) and (
        ind < starts.shape[0]
    ):
        ccg_windowed = ccg[int(starts[ind]) : int(ends[ind] + 1)]
        ccg_windowed_sum = ccg_windowed.sum()
        window_size = ccg_windowed.shape[0] * bin_size_s
        ind += 1
    # use the whole ccg if peak finding fails
    if ind == starts.shape[0]:
        ccg_windowed = ccg

    if np.abs(ccg_windowed).sum() == 0:
        return 0

    metric = (
        wasserstein_distance(
            np.arange(ccg_windowed.shape[0]) / ccg_windowed.shape[0],
            np.arange(ccg_windowed.shape[0]) / ccg_windowed.shape[0],
            ccg_windowed,
            np.ones_like(ccg_windowed),
        )
        * 4
    )

    if ccg_windowed.sum() < (min_ccg_rate * window_size):
        metric *= (ccg_windowed.sum() / (min_ccg_rate * window_size)) ** 2

    if ccg_windowed.sum() < (min_ccg_rate / 4 * window_size):
        metric = -4  # don't merge if the event count is way too low

    return metric


def _sliding_RP_viol_pair(
    ccg: NDArray[np.floating],
    bin_size_ms: float,
    accept_threshold: float = 0.15,
    noise_margin_z: float = 2.0,
) -> float:
    """
    Calculate the sliding refractory period violation confidence for a cluster pair.

    Adapted from the SpikeInterface/IBL sliding RP metric, but the baseline rate is
    based on the maximum rate of the smoothed correlogram, instead of the rate at 2s
    from the center.

    Uses absolute refractory period test points (0.25–20 ms) instead of
    bin-relative multiples, so that sub-millisecond refractory periods of
    fast-spiking neurons (e.g. thalamic relay cells) can be resolved.  A
    Poisson noise margin prevents high-firing-rate neurons from being
    penalized for statistically insignificant excursions above the
    contamination threshold.

    Parameters
    ----------
    ccg : NDArray[np.floating]
        The merged auto-correlogram of the cluster pair.
    bin_size_ms : float
        The width in ms of the bin size of the input ccgs.
    accept_threshold : float, default: 0.15
        The maximum acceptable contamination fraction relative to the
        baseline firing rate.
    noise_margin_z : float, default: 2.0
        Number of Poisson standard deviations added to the expected count
        threshold.

    Returns
    -------
    refractory_penalty : float
        The refractory period violation confidence for the cluster pair.
        0 = clean refractory period, 1 = severe violations.
    """
    # Absolute refractory period test points (ms).  Includes sub-ms points
    # to resolve short refractory periods of fast-spiking neurons.
    rp_test_ms = np.array(
        [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 12.0, 16.0, 20.0]
    )
    # Convert to bin indices (1-based: bin 1 = 0–bin_size ms)
    test_indices = np.round(rp_test_ms / bin_size_ms).astype(int)

    # Keep only test points that map to distinct, valid bins
    n_bins_half = len(ccg) // 2
    valid = (test_indices >= 1) & (test_indices <= n_bins_half)
    test_indices = test_indices[valid]
    rp_test_ms = rp_test_ms[valid]

    if len(test_indices) == 0:
        return 0.0

    # Deduplicate indices that map to the same bin (keep the first = shortest RP)
    _, unique_idx = np.unique(test_indices, return_index=True)
    test_indices = test_indices[unique_idx]
    rp_test_ms = rp_test_ms[unique_idx]

    # Average both halves of ccg to ensure symmetry, keep the positive-lag half.
    # Handle both even-length and odd-length (center bin excluded) CCGs.
    if len(ccg) % 2 == 1:
        half = len(ccg) // 2
        ccg = (ccg[half + 1:] + ccg[:half][::-1]) / 2
    else:
        half = len(ccg) // 2
        ccg = (ccg[half:] + ccg[:half][::-1]) / 2

    ccg_cumsum = np.cumsum(ccg)
    sum_res = ccg_cumsum[test_indices - 1]  # -1 bc 0th bin = 0–bin_size ms

    # Low-pass filter ccg and use max as baseline event rate
    order = 4
    cutoff_freq = 250  # Hz
    fs = 1 / bin_size_ms * 1000
    nyquist = fs / 2
    cutoff = cutoff_freq / nyquist
    if cutoff >= 1.0:
        # Bin size too coarse for this filter — use unfiltered max
        smoothed_ccg = ccg
    else:
        sos = butter(order, cutoff, btype="low", output="sos")
        smoothed_ccg = sosfiltfilt(sos, ccg)

    max_bin_rate = np.max(smoothed_ccg)

    # Expected counts at each RP under accept_threshold contamination,
    # plus a Poisson noise margin so high-firing-rate pairs are not
    # penalized for small excursions within sampling noise.
    rp_test_s = rp_test_ms / 1000
    bin_size_s = bin_size_ms / 1000
    max_conts_base = (rp_test_s / bin_size_s) * (max_bin_rate * accept_threshold)
    max_conts_max = max_conts_base + noise_margin_z * np.sqrt(
        np.maximum(max_conts_base, 1.0)
    )

    # Confidence of less than accept_threshold contamination at each RP
    confs = 1 - poisson.cdf(sum_res, max_conts_max)
    refractory_penalty = 1 - confs.max()

    return refractory_penalty
