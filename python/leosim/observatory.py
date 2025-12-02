# This file is part of leosim.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ("Observatory",)

import os
import astropy.units as u
import numpy as np

import rubin_sim.phot_utils as photUtils
from rubin_sim.data import get_data_dir

class Observatory:
    """A class representing an observatory consisting of the telescope geometry 
    and camera properties.

    Parameters
    ----------
    outer_radius : `astropy.units.Quantity`
        Outer radius of the telescope primary mirror.
    inner_radius : `astropy.units.Quantity`
        Inner radius of the telescope primary mirror.
    pixel_scale : `astropy.units.Quantity`
        Pixel scale of the observatory camera.
    bands : `list` [`str`], optional
        List of names of the filter bands. (['u', 'g', 'r', 'i', 'z', 'y'], 
        by default).
    gain : `astropy.units.Quantity``, optional
        Gain of the observatory camera (1.0, by default).

    Raises
    ------
    ValueError
        Raised if parameter ``outer_radius`` is not greater than parameter
        ``inner_radius``.
    """

    def __init__(self, outer_radius, inner_radius, pixel_scale, bands=["u", "g", "r", "i", "z", "y"],
                 gain=1.0*u.electron/u.adu):

        if outer_radius.to(u.m) <= inner_radius.to(u.m):
            raise ValueError("Outer radius must be greater than inner radius.")
        self._outer_radius = outer_radius.to(u.m)
        self._inner_radius = inner_radius.to(u.m)        
        self._pixel_scale = pixel_scale.to(u.arcsec/u.pix)
        self._bandpasses = dict()
        self.add_bandpasses(bands)
        self._gain = gain.to(u.electron/u.adu)

    @property
    def outer_radius(self):
        """Outer radius of the telescope primary mirror 
        (`astropy.units.Quantity`, read-only).
        """
        return self._outer_radius

    @property
    def inner_radius(self):
        """Inner radius of the telescope primary mirror 
        (`astropy.units.Quantity`, read-only).
        """
        return self._inner_radius

    @property
    def pixel_scale(self):
        """Pixel scale of the observatory camera (`astropy.units.Quantity`, 
        read-only).
        """
        return self._pixel_scale

    @property
    def gain(self):
        """Gain of the observatory camera (`float`, read-only).
        """
        return self._gain

    @property
    def bandpasses(self):
        """Dictionary of telescope throughput curves. Keys are filter band 
        names (`dict` [`str`, `rubin_sim.phot_utils.Bandpass`], read-only).
        """
        return self._bandpasses

    @property
    def effective_area(self):
        """Effective collecting area of the telescope 
        (`astropy.units.Quantity`, read-only).
        """
        effective_area = np.pi*(self.outer_radius**2 - self.inner_radius**2)
        return effective_area.to(u.m*u.m)

    def get_photo_params(self, exptime):
        """Create photometric parameters for an exposure.

        Parameters
        ----------
        exptime : `astropy.units.Quantity`
            Exposure time.

        Returns
        -------
        photo_params : `rubin_sim.phot_utils.PhotometricParameters`
            Photometric parameters for the exposure.
        """
        photo_params = photUtils.PhotometricParameters(exptime=exptime.to_value(u.s), nexp=1, 
                                                       gain=self.gain.to_value(u.electron/u.adu),
                                                       effarea=self.effective_area.to_value(u.cm*u.cm),
                                                       platescale=self.pixel_scale.to_value(u.arcsec/u.pix))

        return photo_params

    def add_bandpasses(self, bands):
        """Add bandpasses to the dictionary.

        Parameters
        ----------
        bands : `list` [`str`]
            List of names of the filter bands.

        Raises
        ------
        ValueError
            Raised if a filter band name is a key for a bandpass already 
            present in the dictionary.
        """
        if not isinstance(bands, list):
            bands = [bands]

        for band in bands:
            if band in self.bandpasses:
                raise ValueError(f"Band {band} is already present in the dictionary.")
            filename = os.path.join(get_data_dir(), "throughputs", "baseline", f"total_{band}.dat")
            bandpass = photUtils.Bandpass()
            bandpass.read_throughput(filename)
            self._bandpasses[band] = bandpass

    def get_bandpass(self, band):
        """Get a bandpass from the dictionary.
        
        Parameters
        ----------
        band : `str`
            Name of the filter band.

        Returns
        -------
        bandpass : `rubin_sim.phot_utils.Bandpass`
            Telescope throughput curve.

        Raises
        ------
        ValueError
            Raised if a filter band name is not a key for a bandpass in the
            dictionary. 
        """
        if band not in self.bandpasses:
            raise ValueError(f"Band {band} is not present in the dictionary")
        bandpass = self.bandpasses[band]

        return bandpass
