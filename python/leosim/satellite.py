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

__all__ = ("DiskSatellite", "RectangularSatellite", "ComponentSatellite")

import numpy as np
import os
from astropy.constants import G, M_earth, R_earth
import astropy.units as u
import galsim

import rubin_sim.phot_utils as photUtils

from .component import * # is this needed?

class BaseSatellite:
    """A class representing a base satellite object.

    Parameters
    ----------
    height : `astropy.units.Quantity`
        Orbital height.
    zangle : `astropy.units.Quantity`
        Observed angle from zenith.
    phi : `astropy.units.Quantity`, optional
        Rotational angle (90 degrees, by default).

    Raises
    ------
    ValueError
        Raised if parameter ``zangle`` is less than 0 deg.
    """

    def __init__(self, height, zangle, phi=90*u.deg): 
        self.height = height
        self.zangle = zangle
        self.phi = phi
        self._sed = photUtils.Sed()
        self._sed.set_flat_sed()

    @property
    def height(self):
        """Orbital height (`astropy.units.Quantity`)."""
        return self._height

    @height.setter
    def height(self, value):
        self._height = value.to(u.km)

    @property
    def zangle(self):
        """Observed angle from zenith (`astropy.units.Quantity`)."""
        return self._zangle

    @zangle.setter
    def zangle(self, value):
        if value.to(u.deg) < 0.:
            raise ValueError('zangle cannot be less than 0 deg')
        self._zangle = value.to(u.deg)

    @property
    def phi(self):
        """Rotational angle (`astropy.units.Quantity`)."""
        return self._phi

    @phi.setter
    def phi(self, value):
        self._phi = value.to(u.deg)

    @property
    def theta(self):
        """Nadir angle (`astropy.units.Quantity`)."""
        theta = np.arcsin(R_earth*np.sin(self.zangle)/(R_earth + self.height))
        return theta.to(u.deg)

    @property
    def distance(self):
        """Distance to satellite (`astropy.units.Quantity`, read-only)."""
        if np.isclose(self.theta.value, 0):
            distance = self.height
        else:
            distance = np.sin(self.zangle - self.theta)*R_earth/np.sin(self.theta)
        return distance.to(u.km)

    @property
    def sed(self):
        """Spectral energy distribution (`rubin_sim.phot_utils.Sed`)."""
        return self._sed

    @property
    def profile(self):
        """Surface brightness profile (`galsim.GSObject`, read-only)
        """
        return None

    @property
    def orbital_velocity(self):
        """Orbital velocity (`astropy.units.Quantity`, read-only)."""
        v = np.sqrt(G*M_earth/(R_earth + self.height))
        return v.to(u.m/u.s, equivalencies=u.dimensionless_angles())

   @property
    def orbital_omega(self):
        """Orbital angular velocity (`astropy.units.Quantity`, read-only)."""
        omega = self.orbital_velocity/(R_earth + self.height)
        return omega.to(u.rad/u.s, equivalencies=u.dimensionless_angles())

    @property
    def tangential_velocity(self):
        """Velocity tangential to the line-of-sight (`astropy.units.Quantity`, 
        read-only).
        """
        v = self.orbital_velocity*np.cos(self.theta)
        return v.to(u.m/u.s, equivalencies=u.dimensionless_angles())

    @property
    def tangential_omega(self):
        """Angular velocity tangential to the line-of-sight 
        (`astropy.units.Quantity`, read-only).
        """
        omega = self.tangential_velocity/(self.distance)
        return omega.to(u.rad/u.s, equivalencies=u.dimensionless_angles())

    def get_defocus_profile(self, instrument):
        """Calculate a defocusing profile for a given instrument.

        Parameters
        ----------
        instrument : `leosim.Instrument`
            Instrument used for observation.
        
        Returns
        -------
        defocus_profile : `galsim.GSObject`
            Defocus kernel profile.
        """
        r_o = (instrument.outer_radius/self.distance).to_value(u.arcsec, 
                                                               equivalencies=u.dimensionless_angles())
        r_i = (instrument.inner_radius/self.distance).to_value(u.arcsec, 
                                                               equivalencies=u.dimensionless_angles())
        defocus_profile = galsim.TopHat(r_o) - galsim.TopHat(r_i, flux=(r_i/r_o)**2.)

        return defocus_profile

    def get_exptime(self, plate_scale):
        """Calculate effective pixel exposure time for a given plate scale.

        Parameters
        ----------
        plate_scale : `astropy.units.Quantity`
            Instrument plate scale.
        
        Returns
        -------
        exptime : `astropy.units.Quantity`
            Effective pixel exposure time.
        """
        plate_scale = plate_scale.to(u.arcsec/u.pix)
        exptime = (plate_scale/self.tangential_omega)

        return exptime.to(u.s, equivalencies=[(u.pix, None)])
    
    def get_flux(self, magnitude, bandpass, instrument):
        """Calculate the number of ADU for a given observation.

        Parameters
        ----------
        magnitude : `float`
            Stationary AB magnitude.
        bandpass : `rubin_sim.phot_utils.Bandpass`
            Telescope throughput curve.
        instrument : `leosim.Instrument`
            Instrument used for observation.

        Returns
        -------
        adu : `float`
            Number of ADU.
        """
        exptime = self.get_exptime(intrument.plate_scale)
        photo_params = instrument.get_photo_params(exptime=exptime.to_value(u.s)

        m0_adu = self.sed.calc_adu(bandpass, phot_params=photo_params)
        adu = m0_adu*(10**(-magnitude/2.5))

        return adu

    def get_stationary_profile(self, seeing_profile, instrument, magnitude=None, bandpass=None, **flux_kwargs):
        """Create the satellite stationary surface brightness profile.

        The satellite stationary surface brightness profile is created by 
        convolving the satellite surface brightness profile with the defocus
        kernel profile (determined by the instrument geometry), and an atmospheric
        PSF. By providing a magnitude and a bandpass, the resulting profile
        will be scaled by the appropriate flux value.

        Parameters
        ----------
        seeing_profile : `galsim.GSObject`
            A surface brightness profile reprsenting an atmospheric PSF.
        instrument : `leosim.Instrument`
            Instrument used for observation.
        magnitude : `float`, optional
            Stationary AB magnitude (None by default).
        bandpass : `rubin_sim.phot_utils.Bandpass`, optional
            Telescope throughput curve (None by default).
        **flux_kwargs
            Additional keyword arguments passed to `get_flux`.

            ``wavelen``:
                Wavelength array for spectral energy distribution (nm).
            ``fnu``:
                Flux density array for spectral energy distribution (Jy).

        Returns
        -------
        stationary_profile : `galsim.GSObject`
            The satellite stationary surface brightness profile.
        """
    
        defocus_profile = self.get_defocus_profile(instrument)
        stationary_profile = galsim.Convolve([self.profile, defocus_profile, seeing_profile])

        if (magnitude is not None) and (bandpass is not None):
            adu = self.get_flux(magnitude, bandpass, instrument, **flux_kwargs)
        else:
            adu = 1.0
        stationary_profile = stationary_profile.withFlux(adu)

        return stationary_profile

    def get_normalized_profile(self, seeing_profile, instrument, step_size, steps):

        defocus_profile = self.get_defocus_profile(instrument)
        final_profile = galsim.Convolve([self.profile, defocus_profile, seeing_profile])
        image = final_profile.drawImage(scale=step_size, nx=steps, ny=steps)

        profile = np.sum(image.array, axis=0)
        normalized_profile = profile/np.max(profile)
        scale = np.linspace(-int(steps*step_size/2), int(steps*step_size/2), steps)

        return scale, normalized_profile

    def get_surface_brightness_profile(self, magnitude, bandpass, seeing_profile, instrument, step_size, steps):
        """Calculate the cross-sectional surface brightness profile.

        Parameters
        ----------
        magnitude : `float`
            Stationary AB magnitude.
        bandpass : `rubin_sim.phot_utils.Bandpass`, optional
            Telescope throughput curve (None by default).
        seeing_profile : `galsim.GSObject`
            A surface brightness profile representing an atmospheric PSF.
        instrument : `leosim.Instrument`
            Instrument used for observation.
        step_size : `float`
            Pixel scale for the image in arcseconds.
        steps : `int`
            Size of image in x and y direction.

        Returns
        -------
        scale : `numpy.ndarray`
            Angle array for cross-sectional surface brightness profile (arcsec).
        profile : `numpy.ndarray`
            Flux linear density array for cross-sectional surface brightness profile (adu/pixel).
        """
        flux = self.get_flux(magnitude, bandpass, instrument)
        defocus_profile = self.get_defocus_profile(instrument)
        final_profile = galsim.Convolve([self.profile, defocus_profile, seeing_profile])
        final_profile = final_profile.withFlux(flux)
        image = final_profile.drawImage(scale=step_size, nx=steps, ny=steps)
       
        profile = np.sum(image.array, axis=0)*instrument.pixel_scale.to_value(u.arcsec/u.pix)/step_size
        scale = np.linspace(-int(steps*step_size/2), int(steps*step_size/2), steps)

        return scale, profile
   
class DiskSatellite(BaseSatellite):
    """A class representing a circular disk satellite.

    Parameters
    ----------
    height : `astropy.units.Quantity`
        Orbital height.
    zangle : `astropy.units.Quantity`
        Observed angle from zenith.
    radius : `astropy.units.Quantity`
        Radius of the satellite.
    """

    def __init__(self, height, zangle, radius): 
        super().__init__(height, zangle)
        self._radius = radius.to(u.m)

    @property
    def radius(self):
        """Radius of the satellite (`astropy.units.Quantity`)."""
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value.to(u.m)

    @property
    def profile(self):
        """Surface brightness profile (`galsim.TopHat`, read-only)."""
        r = (self.radius/self.distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        return galsim.TopHat(r)

class RectangularSatellite(BaseSatellite):
    """A class representing a rectangular satellite.

    Parameters
    ----------
    height : `astropy.units.Quantity`
        Orbital height.
    zangle : `astropy.units.Quantity`
        Observed angle from zenith.
    width : `astropy.units.Quantity`
        Width of the satellite.
    length : `astropy.units.Quantity`
        Length of the satellite.
    """

    def __init__(self, height, zangle, width, length):
        super().__init__(height, zangle)
        self._width = width.to(u.m)
        self._length = length.to(u.m)

    @property
    def width(self):
        """Width of the satellite (`astropy.units.Quantity`)."""
        return self._width

    @width.setter
    def width(self, value):
        self._width = value.to(u.m)

    @property
    def length(self):
        """Length of the satellite (`astropy.units.Quantity`)."""
        return self._length

    @length.setter
    def length(self, value):
        self._length = value.to(u.m)

    @property
    def profile(self):
        """Surface brightness profile (`galsim.Box`, read-only)."""
        w = (self.width/self.distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        l = (self.length/self.distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        return galsim.Box(w, l)

class ComponentSatellite(BaseSatellite):
    """A class representing a satellite assembled from components.

    Parameters
    ----------
    height : `astropy.units.Quantity`
        Orbital height.
    zangle : `astropy.units.Quantity`
        Observed angle from zenith.
    components : `list` [`leosim.Component`]
        A list of satellite components.

    Raises
    ------
    ValueError
        Raised if ``components`` is of length 0.
    """

    def __init__(self, height, zangle, components):
        super().__init__(height, zangle)

        if len(components) == 0:
            raise ValueError("components list must include at least one component.")
        self._components = components

    @property
    def components(self):
        """A list of satellite components. (`list` [`leosim.Component`], 
        read-only).
        """
        return self._components
        
    @property
    def profile(self):
        """Surface brightness profile (`galsim.GSObject`, read-only)."""
        profile = self.components[0].create_profile(self.distance)
        
        for component in self.components[1:]:
            profile += component.create_profile(self.distance)
            
        return profile

