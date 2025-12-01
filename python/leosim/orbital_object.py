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

__all__ = ("DiskOrbitalObject", "RectangularOrbitalObject", "ComponentOrbital")

import numpy as np
import os
from astropy.constants import G, M_earth, R_earth
import astropy.units as u
import galsim

import rubin_sim.phot_utils as photUtils

from .component import * # is this needed?

class BaseOrbitalObject:
    """A base class that defines attributes and methods common to all orbital
    objects.

    Parameters
    ----------
    height : `astropy.units.Quantity`
        Orbital height.
    zenith_angle : `astropy.units.Quantity`
        Observed angle from telescope zenith.
    phi : `astropy.units.Quantity`, optional
        Rotational angle (90 degrees, by default).

    Raises
    ------
    ValueError
        Raised if parameter ``zenith_angle`` is less than 0 deg.
    """

    def __init__(self, height, zenith_angle, phi=90*u.deg):
        self.height = height
        self.zenith_angle = zenith_angle
        self.phi = phi # May change in future updates
        self._sed = photUtils.Sed()
        self._sed.set_flat_sed()

    @property
    def height(self):
        """Orbital height (`astropy.units.Quantity`).
        """
        return self._height

    @height.setter
    def height(self, value):
        self._height = value.to(u.km)

    @property
    def zenith_angle(self):
        """Angle from telescope zenith to orbital object 
        (`astropy.units.Quantity`).
        """
        return self._zenith_angle

    @zenith_angle.setter
    def zenith_angle(self, value):
        if value.to(u.deg) < 0.:
            raise ValueError('zenith_angle cannot be less than 0 deg')
        self._zenith_angle = value.to(u.deg)

    @property
    def phi(self):
        """Rotational angle (`astropy.units.Quantity`)."""
        return self._phi

    @phi.setter
    def phi(self, value):
        self._phi = value.to(u.deg)

    @property
    def nadir_angle(self):
        """Angle from orbital object nadir to telescope 
        (`astropy.units.Quantity`, read-only).
        """
        nadir_angle = np.arcsin(R_earth*np.sin(self.zenith_angle)/(R_earth + self.height))
        return nadir_angle.to(u.deg)

    @property
    def distance(self):
        """Distance to orbital object from telescope (`astropy.units.Quantity`, 
        read-only).
        """
        if np.isclose(self.theta.value, 0):
            distance = self.height
        else:
            distance = np.sin(self.zenith_angle - self.nadir_angle)*R_earth/np.sin(self.zenith_angle)
        return distance.to(u.km)

    @property
    def sed(self):
        """Spectral energy distribution (`rubin_sim.phot_utils.Sed`, 
        read-only).
        """
        return self._sed

    @property
    def profile(self):
        """Orbital object geometric surface brightness profile 
        (`galsim.GSObject`, read-only).
        """
        return None

    # Should these be described as amplitudes or speeds?

    @property
    def orbital_velocity(self):
        """Orbital velocity (`astropy.units.Quantity`, read-only).
        """
        v = np.sqrt(G*M_earth/(R_earth + self.height))
        return v.to(u.m/u.s, equivalencies=u.dimensionless_angles())

    @property
    def orbital_omega(self):
        """Orbital angular velocity (`astropy.units.Quantity`, read-only).
        """
        omega = self.orbital_velocity/(R_earth + self.height)
        return omega.to(u.rad/u.s, equivalencies=u.dimensionless_angles())

    @property
    def perpendicular_velocity(self): # Expand to include all directions of orbital object motion
        """Velocity perpendicular to the line-of-sight vector 
        (`astropy.units.Quantity`, read-only).
        """
        v = self.orbital_velocity*np.cos(self.theta)
        return v.to(u.m/u.s, equivalencies=u.dimensionless_angles())

    @property
    def perpendicular_omega(self): # Need new name
        """Angular velocity perpendicular to the line-of-sight vector 
        (`astropy.units.Quantity`, read-only).
        """
        omega = self.perpendicular_velocity/self.distance
        return omega.to(u.rad/u.s, equivalencies=u.dimensionless_angles())

    def get_defocus(self, instrument): # Maybe simplify to single top hat for instances where r_i is close to zero.
        """Calculate the defocus kernel profile for a given instrument.

        Parameters
        ----------
        instrument : `leosim.Instrument`
            Instrument used for observation.
        
        Returns
        -------
        defocus : `galsim.GSObject`
            Defocus kernel profile.
        """
        r_o = (instrument.outer_radius/self.distance).to_value(u.arcsec, 
                                                               equivalencies=u.dimensionless_angles())
        r_i = (instrument.inner_radius/self.distance).to_value(u.arcsec, 
                                                               equivalencies=u.dimensionless_angles())
        defocus = galsim.TopHat(r_o) - galsim.TopHat(r_i, flux=(r_i/r_o)**2.)

        return defocus

# Construction starts here (11/30/2024, 17:45).

    def get_pixel_exptime(self, pixel_scale): # ensure pixel vs plate scale
        """Calculate the pixel traversal exposure time.

        The pixel traversal exposure time is the time for the orbital object to
        traverse a single pixel that is dependent on the angular velocity
        perpendicular to the line-of-sight vector.

        Parameters
        ----------
        pixel_scale : `astropy.units.Quantity`
            Instrument pixel scale.

        Returns
        -------
        pixel_exptime : `astropy.units.Quantity`
            Pixel traversal exposure time.
        """
        pixel_scale = pixel_scale.to(u.arcsec/u.pix)
        pixel_exptime = pixel_scale/self.perpendicular_omega

        return pixel_exptime.to(u.s, equivalencies=[(u.pix, None)])

# Below is flagged for replacement and removal (11/30/2024, 17:45).
   
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
        exptime = self.get_exptime(instrument.plate_scale)
        photo_params = instrument.get_photo_params(exptime=exptime.to_value(u.s))

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
       
        profile = np.sum(image.array, axis=0)*instrument.plate_scale.to_value(u.arcsec/u.pix)/step_size
        scale = np.linspace(-int(steps*step_size/2), int(steps*step_size/2), steps)

        return scale, profile

# Below child classes have been largely updated except for CompositeOrbitalObject (11/30/2025, 17:30). 

class DiskOrbitalObject(BaseOrbitalObject):
    """A circular disk orbital object.

    Parameters
    ----------
    height : `astropy.units.Quantity`
        Orbital height.
    zangle : `astropy.units.Quantity`
        Observed angle from telescope zenith.
    radius : `astropy.units.Quantity`
        Radius of the orbital object.
    """

    def __init__(self, height, zangle, radius): 
        super().__init__(height, zangle)
        self._radius = radius.to(u.m)

    @property
    def radius(self):
        """Radius of the orbital object (`astropy.units.Quantity`, read-only).
        """
        return self._radius

    @property
    def profile(self):
        """Orbital object geometric surface brightness profile 
        (`galsim.TopHat`, read-only).
        """
        r = (self.radius/self.distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        return galsim.TopHat(r)

class RectangularOrbitalObject(BaseOrbitalObject):
    """A rectangular orbital object.

    Parameters
    ----------
    height : `astropy.units.Quantity`
        Orbital height.
    zangle : `astropy.units.Quantity`
        Observed angle from telescope zenith.
    width : `astropy.units.Quantity`
        Width of the orbital object.
    length : `astropy.units.Quantity`
        Length of the orbital object.
    """

    def __init__(self, height, zangle, width, length):
        super().__init__(height, zangle)
        self._width = width.to(u.m)
        self._length = length.to(u.m)

    @property
    def width(self):
        """Width of the orbital object (`astropy.units.Quantity`, read-only).
        """
        return self._width

    @property
    def length(self):
        """Length of the orbital object (`astropy.units.Quantity`, read-only).
        """
        return self._length

    @property
    def profile(self):
        """Orbital object geometric surface brightness profile (`galsim.Box`, 
        read-only).
        """
        w = (self.width/self.distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        l = (self.length/self.distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        return galsim.Box(w, l)

class CompositeOrbitalObject(BaseOrbitalObject): # This needs work to perform proper component list checks.
    """A composite orbital object made up of smaller components.

    Parameters
    ----------
    height : `astropy.units.Quantity`
        Orbital height.
    zangle : `astropy.units.Quantity`
        Observed angle from telescope zenith.
    components : `list` [`leosim.Component`]
        A list of components.

    Raises
    ------
    ValueError
        Raised if ``components`` is of length 0.
    """

    def __init__(self, height, zangle, components):
        super().__init__(height, zangle)

        if len(components) == 0: # Need a way to check this is a non-empty list or tuple (or similar).
            raise ValueError("components list must include at least one component.")
        self._components = components

    @property
    def components(self):
        """A list of components. (`list` [`leosim.Component`], 
        read-only).
        """
        return self._components        

    @property
    def profile(self):
        """Orbital object geometric surface brightness profile 
        (`galsim.GSObject`, read-only).
        """
        # Check create_profile method for proper astropy unit conversion.
        component_profiles = [component.create_profile(self.distance) for component in self.components]
        return galsim.Sum(*component_profiles)
