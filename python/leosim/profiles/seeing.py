"""This module holds the 1D Gaus-Kolmogorov provile commonly used to describe
effects of atmospheric seeing on a point source. The function is described in:
  Bektesevic & Vinkovic et. al. 2017 (arxiv: 1707.07223)
"""
import numpy as np
from scipy.interpolate import interp1d

from leosim.profiles.convolutionobj import ConvolutionObject
from leosim.profiles.consts import FWHM2SIGMA
import os

__all__ = ["GausKolmogorov", "VonKarman"]

class GausKolmogorov(ConvolutionObject):
    """Simple Gaus-Kolmogorov seeing. Convolving with this function has the
    effect of blurring the original object.
    Parameters
    ----------
    fwhm : float
      FWHM of the Gaus-Kolmogorov profile
    scale : list, tuple or np.array
      if scale is given then GK will be evaluated over it
    res : float
      if scale is not given one will be created from the estimated width of the
      GK profile and resolution in arcseconds
    """
    def __init__(self, fwhm, scale=None, res=0.001, **kwargs):
        self.seeingfwhm = fwhm
        self.sigma = 1.035/FWHM2SIGMA*fwhm
        self.sigma2 = 2.*self.sigma

        if scale is None:
            scale = np.arange(-4*self.sigma2, 4*self.sigma2, res)
            obj = self.f(scale)
        else:
            scale = scale
            obj = self.f(scale)
        ConvolutionObject.__init__(self, obj, scale)

    def f(self, r, sigma=None):
        """Evaluates the GK at a point r. Providing sigma estimates GK at r for
        a different GK distribution with a FWHM= sigma*2.436/1.035 - for
        convenience only.
        """
        if sigma is None:
            sigma = self.sigma
        sigma2 = 2*sigma*sigma

        def gk(x):
            return 0.909 * (1 / (np.pi * sigma2) * np.exp((-x * x) / sigma2)
                               + 0.1 * (1 / (np.pi * sigma2) * np.exp((-x * x) / sigma2)))
        try:
            return gk(r)
        except TypeError:
            return gk(np.asarray(r))

class VonKarman(ConvolutionObject):
    """Simple Von Karman seeing, interpolated from existing points. Convolving 
    with this function has the effect of blurring the original object.
    Parameters
    ----------
    fwhm : float
      FWHM of the Von Karman profile
    scale : list, tuple or np.array
      if scale is given then VK will be evaluated over it
    res : float
      if scale is not given one will be created from the estimated width of the
      VK profile and resolution in arcseconds
    """
   
    def __init__(self, fwhm, scale=None, res=0.001):
                
        self.seeingfwhm = fwhm
        self.sigma = 1.035/FWHM2SIGMA*fwhm
        self.sigma2 = 2.*self.sigma
        
        ## Load VK datapoints
        fname = 'r_vonK_Kolm.txt'
        this_file = os.path.abspath(__file__)
        this_dir = os.path.dirname(this_file)
        vk_atm = np.loadtxt(os.path.join(this_dir, 'data', fname))
        
        ## Extend to negative
        vk_x = np.zeros(2*vk_atm.shape[0]-1)
        vk_x[500:] = vk_atm[:, 0]
        vk_x[:500] = -vk_atm[:0:-1, 0]
        vk_y = np.zeros(2*vk_atm.shape[0]-1)
        vk_y[500:] = vk_atm[:, 1]
        vk_y[:500] = vk_atm[:0:-1, 1]
        
        self.f = interp1d(vk_x*self.seeingfwhm, vk_y, kind='cubic')

        if scale is None:
            scale = np.arange(-4*self.sigma2, 4*self.sigma2, res)
            obj = self.f(scale)
        else:
            scale = scale
            obj = self.f(scale)
        ConvolutionObject.__init__(self, obj, scale)
