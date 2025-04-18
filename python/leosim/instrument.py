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

__all__ = ("Instrument",)

import os
import astropy.units as u
import numpy as np

import rubin_sim.phot_utils as photUtils
from rubin_sim.data import get_data_dir

class Instrument:
    """A class representing a telescope and camera.

    Parameters
    ----------
    outer_radius : `(astropy.units.Quantity)`
        Outer radius of the primary mirror.
    inner_radius : `(astropy.units.Quantity)`
        Inner radius of the primary mirror.
    pixel_scale : `(astropy.units.Quantity)`
        Pixel scale of the instrument camera.
    gain : `float`, optional
        Gain of the instrument camera (electrons per ADU). Default is 1.

    Raises
    ------
    ValueError
        Raised if parameter ``outer_radius`` is not greater than parameter
        ``inner_radius``.
    """

    gain = None
    """Gain of the instrument camera (`float`)."""

    def __init__(self, outer_radius, inner_radius, plate_scale, gain=1.0):

        if outer_radius <= inner_radius:
            raise ValueError("Outer radius must be greater than inner radius.") 
        self._outer_radius = outer_radius.to(u.m)
        self._inner_radius = inner_radius.to(u.m)
        self._plate_scale = plate_scale.to(u.arcsec/u.pix)
        self.gain = gain

    @property
    def outer_radius(self):
        """Outer radius of the primary mirror (`astropy.units.Quantity`, 
        read-only).
        """
        return self._outer_radius

    @property
    def inner_radius(self):
        """Inner radius of the primary mirror (`astropy.units.Quantity`, 
        read-only).
        """
        return self._inner_radius

    @property
    def plate_scale(self):
        """Plate scale of camera (`astropy.units.Quantity`, read-only)."""
        return self._pixel_scale

    @property
    def effarea(self):
        """Effective collecting area (`astropy.units.Quantity`, read-only)."""
        return np.pi*(self.outer_radius**2 - self.inner_radius**2)

    def get_photo_params(self, exptime):
        """Generate photometric parameters for a given exposure.

        Parameters
        ----------
        exptime : `float`
            Exposure time (seconds).

        Returns
        -------
        photo_params : `rubin_sim.phot_utils.PhotometricParameters`
            Photometric parameters for the exposure.
        """
        plate_scale = self.plate_scale.to_value(u.arcsec/u.pix)
        effarea = self.effarea.to_value(u.cm*u.cm)
        photo_params = photUtils.PhotometricParameters(exptime=exptime, nexp=1, effarea=effarea,
                                                       gain=self.gain, platescale=plate_scale)
        return photo_params
