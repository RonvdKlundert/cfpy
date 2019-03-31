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
    hrf_shape = np.ones(len(stimulus.shape))
    hrf_shape[-1] = hrf.shape[0]
    return signal.fftconvolve(stimulus, hrf.reshape(hrf_shape), mode='full', axes=(-1))[...,:stimulus.shape[-1]]

def stimulus_through_prf(prfs, stimulus):
    """stimulus_through_prf
    
    dot the stimulus and the prfs
    
    [description]
    
    Parameters
    ----------
    prfs : numpy.ndarray
        the array of prfs. 
    stimulus : numpy.ndarray
        the stimulus design matrix, either convolved with hrf or not.
    
    """
    assert prfs.shape[1:] == stimulus.shape[:-1], \
        'prf array dimensions {prfdim} and input stimulus array dimensions {stimdim} must have same dimensions'.format(
            prfdim=prfs.shape[1:], 
            stimdim=stimulus.shape[:-1])
    return np.dot(prfs, stimulus)

def generate_arima_noise(ar=(1,0.4), 
                            ma=(1,0.4), 
                            dimensions=(1000,120)):
    """generate_arima_noise
    
    generate_arima_noise creates temporally correlated noise
    
    Parameters
    ----------
    ar : tuple, optional
        arima autoregression parameters for statsmodels generation of noise 
        (the default is (1,0.4), which should be a reasonable setting for fMRI noise)
    ma : tuple, optional
        arima moving average parameters for statsmodels generation of noise 
        (the default is (1,0.4), which should be a reasonable setting for fMRI noise)        
    dimensions : tuple, optional
        the first dimension is the nr of separate timecourses, the second dimension
        is the timeseries length.
        (the default is (1000,120), a reasonable if brief length for an fMRI run)
    
    """
    return np.array([arma_generate_sample(ar, ma, dimensions[1]) for i in range(dimensions[0])])

def generate_random_legendre_drifts(dimensions=(1000,120), 
                                    amplitude_ranges=[[500,600],[-50,50],[-20,20],[-10,10],[-5,5]]):
    """generate_random_legendre_drifts

    generate_random_legendre_drifts generates random slow drifts

    Parameters
    ----------
    dimensions : tuple, optional
        [description] (the default is (1000,120), which [default_description])
    amplitude_ranges : list, optional
        [description] (the default is [[500,600],[-50,50],[-20,20],[-10,10],[-5,5]], which [default_description])

    Returns
    -------
    numpy.ndarray
        legendre poly drifts with dimensions [dimensions]
    numpy.ndarray
        random multiplication factors that created the drifts
    """
    nr_polys = len(amplitude_ranges)
    drifts = np.polynomial.legendre.legval(x=np.arange(dimensions[-1]), c=np.eye(nr_polys)).T
    drifts = (drifts-drifts.mean(0))/drifts.mean(0)
    drifts[:,0] = np.ones(drifts[:,0].shape)
    random_factors = np.array([ar[0] + (ar[1]-ar[0])/2.0 + (np.random.rand(dimensions[0])-0.5) * (ar[1]-ar[0])
                            for ar in amplitude_ranges])
    return np.dot(drifts, random_factors), random_factors

def generate_random_cosine_drifts(dimensions=(1000,120), 
                                    amplitude_ranges=[[500,600],[-50,50],[-20,20],[-10,10],[-5,5]]):
    """generate_random_cosine_drifts

    generate_random_cosine_drifts generates random slow drifts

    Parameters
    ----------
    dimensions : tuple, optional
        [description] (the default is (1000,120), which [default_description])
    amplitude_ranges : list, optional
        [description] (the default is [[500,600],[-50,50],[-20,20],[-10,10],[-5,5]], which [default_description])

    Returns
    -------
    numpy.ndarray
        discrete cosine drifts with dimensions [dimensions]
    numpy.ndarray
        random multiplication factors that created the drifts
    """
    nr_freqs = len(amplitude_ranges)
    x = np.linspace(0,np.pi,dimensions[-1])
    drifts = np.array([np.cos(x*f) for f in range(nr_freqs)]).T
    random_factors = np.array([ar[0] + (ar[1]-ar[0])/2.0 + (np.random.rand(dimensions[0])-0.5) * (ar[1]-ar[0])
                            for ar in amplitude_ranges])
    return np.dot(drifts, random_factors), random_factors    