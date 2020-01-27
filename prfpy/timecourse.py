import numpy as np
import scipy.signal as signal
from statsmodels.tsa.arima_process import arma_generate_sample


def convolve_stimulus_dm(stimulus, hrf):
    """convolve_stimulus_dm

    convolve_stimulus_dm convolves an N-D (N>=2) stimulus array with an hrf

    Parameters
    ----------
    stimulus : numpy.ndarray, N-D (N>=2) 
        stimulus experimental design, with the final dimension being time
    hrf : numpy.ndarray, 1D
        contains kernel for convolution

    """
    hrf_shape = np.ones(len(stimulus.shape), dtype=np.int)
    hrf_shape[-1] = hrf.shape[0]

    return signal.fftconvolve(stimulus, hrf.reshape(hrf_shape), mode='full', axes=(-1))[..., :stimulus.shape[-1]]


def stimulus_through_prf(prfs, stimulus, mask=None):
    """stimulus_through_prf

    dot the stimulus and the prfs

    Parameters
    ----------
    prfs : numpy.ndarray
        the array of prfs. 
    stimulus : numpy.ndarray
        the stimulus design matrix, either convolved with hrf or not.
    mask : numpy.ndarray
        a mask in feature space, of dimensions equal to 
        the spatial dimensions of both stimulus and receptive field

    """
    assert prfs.shape[1:] == stimulus.shape[:-1], \
        """prf array dimensions {prfdim} and input stimulus array dimensions {stimdim} 
        must have same dimensions""".format(
            prfdim=prfs.shape[1:],
            stimdim=stimulus.shape[:-1])
    if mask == None:
        prf_r = prfs.reshape((prfs.shape[0], -1))
        stim_r = stimulus.reshape((-1, stimulus.shape[-1]))
    else:
        assert prfs.shape[1:] == mask.shape and mask.shape == stimulus.shape[:-1], \
            """mask dimensions {maskdim}, prf array dimensions {prfdim}, 
            and input stimulus array dimensions {stimdim} 
            must have same dimensions""".format(
                maskdim=mask.shape,
                prfdim=prfs.shape[1:],
                stimdim=stimulus.shape[:-1])
        prf_r = prfs[:, mask]
        stim_r = stimulus[mask, :]
    return prf_r @ stim_r


def sgfilter_predictions(predictions, window_length=201, polyorder=3,
                         highpass=True, add_mean=True, task_lengths=None,
                         task_names=None, late_iso_dict=None, **kwargs):
    """sgfilter_predictions

    savitzky golay filter predictions, to conform to data filtering

    Parameters
    ----------
    predictions : numpy.ndarray
        array containing predictions, last dimension is time
    window_length : int, optional
        window length for SG filter (the default is 201, which is ok for prf experiments, and 
        a bit long for event-related experiments)
    polyorder : int, optional
        polynomial order for SG filter (the default is 3, which performs well for fMRI signals
        when the window length is longer than 2 minutes)
    highpass : bool, optional
        whether to use the sgfilter as highpass (True, default) or lowpass (False)
    add_mean : bool, optional
        whether to add the mean of the time-courses back to the signal after filtering
        (True, default) or not (False)
    task_lengths : list or numpy.ndarray, optional
        specify length of each condition in TRs
        If not None, the predictions are split in the time dimension in len(task_lengths) chunks,
        and the savgol filter is applied to each chunk separately.
        The i^th chunk has size task_lengths[i]

    **kwargs are passed on to scipy.signal.savgol_filter

    Raises
    ------
    ValueError
        when window_length is even

    Returns
    -------
    numpy.ndarray
        filtered version of the array
    """
    if window_length % 2 != 1:
        raise ValueError  # window_length should be odd

    if task_lengths is None:
        task_lengths = [predictions.shape[-1]]

    # first assess that the number and sizes of chunks are compatible with the predictions
    assert np.sum(task_lengths) == predictions.shape[-1], "Task lengths \
    are incompatible with the number of prediction timepoints."

    lp_filtered_predictions = np.zeros_like(predictions)
    if highpass:
        hp_filtered_predictions = np.zeros_like(predictions)

    baselines = dict()

    start = 0
    for i, task_length in enumerate(task_lengths):

        stop = start+task_length

        try:
            lp_filtered_predictions[..., start:stop] = signal.savgol_filter(
            predictions[..., start:stop], window_length=window_length,
            polyorder=polyorder, **kwargs)
        except:
            print("Error occurred during predictions savgol filtering.\
                  Using unfiltered prediction instead")
            #lp_filtered_predictions[..., start:stop] = predictions[..., start:stop]

        if add_mean:
            if highpass:
                lp_filtered_predictions[..., start:stop] -= np.mean(
                    predictions[..., start:stop], axis=-1)[..., np.newaxis]
            else:
                lp_filtered_predictions[..., start:stop] += np.mean(
                    predictions[..., start:stop], axis=-1)[..., np.newaxis]

        if highpass:
            hp_filtered_predictions[..., start:stop] = predictions[..., start:stop]\
                - lp_filtered_predictions[..., start:stop]

            if late_iso_dict is not None:
                baselines[task_names[i]] = np.median(hp_filtered_predictions[..., start:stop][...,late_iso_dict[task_names[i]]],
                                                   axis=-1)

        start += task_length

    if late_iso_dict is not None and highpass:
        baseline_full = np.median([baselines[task_name] for task_name in task_names], axis=0)

        start = 0
        for i, task_length in enumerate(task_lengths):
            stop = start+task_length
            baseline_diff = baseline_full - baselines[task_names[i]]
            hp_filtered_predictions[..., start:stop] += baseline_diff[...,np.newaxis]
            start += task_length

    if highpass:
        return hp_filtered_predictions
    else:
        return lp_filtered_predictions



def generate_random_legendre_drifts(dimensions=(1000, 120),
                                    amplitude_ranges=[[500, 600], [-50, 50], [-20, 20], [-10, 10], [-5, 5]]):
    """generate_random_legendre_drifts

    generate_random_legendre_drifts generates random slow drifts

    Parameters
    ----------
    dimensions : tuple, optional
        shape of the desired data, latter dimension = timepoints 
        the default is (1000,120), which creates 1000 timecourses for a brief fMRI run
    amplitude_ranges : list, optional
        Amplitudes of each of the components. Ideally, this should follow something like 1/f. 
        the default is [[500,600],[-50,50],[-20,20],[-10,10],[-5,5]]

    Returns
    -------
    numpy.ndarray
        legendre poly drifts with dimensions [dimensions]
    numpy.ndarray
        random multiplication factors that created the drifts
    """
    nr_polys = len(amplitude_ranges)
    drifts = np.polynomial.legendre.legval(
        x=np.arange(dimensions[-1]), c=np.eye(nr_polys)).T
    drifts = (drifts-drifts.mean(0))/drifts.mean(0)
    drifts[:, 0] = np.ones(drifts[:, 0].shape)
    random_factors = np.array([ar[0] + (ar[1]-ar[0])/2.0 + (np.random.rand(dimensions[0])-0.5) * (ar[1]-ar[0])
                               for ar in amplitude_ranges])
    return np.dot(drifts, random_factors), random_factors


def generate_random_cosine_drifts(dimensions=(1000, 120),
                                  amplitude_ranges=[[500, 600], [-50, 50], [-20, 20], [-10, 10], [-5, 5]]):
    """generate_random_cosine_drifts

    generate_random_cosine_drifts generates random slow drifts

    Parameters
    ----------
    dimensions : tuple, optional
        shape of the desired data, latter dimension = timepoints 
        the default is (1000,120), which creates 1000 timecourses for a brief fMRI run
    amplitude_ranges : list, optional
        Amplitudes of each of the components. Ideally, this should follow something like 1/f. 
        the default is [[500,600],[-50,50],[-20,20],[-10,10],[-5,5]]

    Returns
    -------
    numpy.ndarray
        discrete cosine drifts with dimensions [dimensions]
    numpy.ndarray
        random multiplication factors that created the drifts
    """
    nr_freqs = len(amplitude_ranges)
    x = np.linspace(0, np.pi, dimensions[-1])
    drifts = np.array([np.cos(x*f) for f in range(nr_freqs)]).T
    random_factors = np.array([ar[0] + (ar[1]-ar[0])/2.0 + (np.random.rand(dimensions[0])-0.5) * (ar[1]-ar[0])
                               for ar in amplitude_ranges])
    return np.dot(drifts, random_factors), random_factors


def generate_arima_noise(ar=(1, 0.4),
                         ma=(1, 0.0),
                         dimensions=(1000, 120),
                         **kwargs):
    """generate_arima_noise

    generate_arima_noise creates temporally correlated noise

    Parameters
    ----------
    ar : tuple, optional
        arima autoregression parameters for statsmodels generation of noise 
        (the default is (1,0.4), which should be a reasonable setting for fMRI noise)
    ma : tuple, optional
        arima moving average parameters for statsmodels generation of noise 
        (the default is (1,0.0), which should be a reasonable setting for fMRI noise)        
    dimensions : tuple, optional
        the first dimension is the nr of separate timecourses, the second dimension
        is the timeseries length.
        (the default is (1000,120), a reasonable if brief length for an fMRI run)

    **kwargs are passed on to statsmodels.tsa.arima_process.arma_generate_sample

    Returns 
    -------
    numpy.ndarray
        noise of requested dimensions and properties

    """
    return np.array([arma_generate_sample(ar, ma, dimensions[1], **kwargs) for _ in range(dimensions[0])])
