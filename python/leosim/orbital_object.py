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

__all__ = ("DiskOrbitalObject", "RectangularOrbitalObject", "CompositeOrbitalObject")

import numpy as np
import os
from astropy.constants import G, M_earth, R_earth
import astropy.units as u
import galsim
import scipy

import rubin_sim.phot_utils as photUtils

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
        if np.isclose(self.nadir_angle.value, 0):
            distance = self.height
        else:
            distance = np.sin(self.zenith_angle - self.nadir_angle)*R_earth/np.sin(self.nadir_angle)
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
        v = self.orbital_velocity*np.cos(self.nadir_angle)
        return v.to(u.m/u.s, equivalencies=u.dimensionless_angles())

    @property
    def perpendicular_omega(self):
        """Angular velocity perpendicular to the line-of-sight vector 
        (`astropy.units.Quantity`, read-only).
        """
        omega = self.perpendicular_velocity/self.distance
        return omega.to(u.rad/u.s, equivalencies=u.dimensionless_angles())

    def get_defocus_profile(self, observatory):
        """Create the defocus kernel profile.

        Parameterss
        ----------
        observatory : `leosim.Observatory`
            Observatory viewing the orbital object.
        
        Returns
        -------
        defocus : `galsim.GSObject`
            Defocus kernel profile.
        """
        r_o = (observatory.outer_radius/self.distance).to_value(u.arcsec, 
                                                                equivalencies=u.dimensionless_angles())
        r_i = (observatory.inner_radius/self.distance).to_value(u.arcsec, 
                                                                equivalencies=u.dimensionless_angles())
        defocus = galsim.TopHat(r_o) - galsim.TopHat(r_i, flux=(r_i/r_o)**2.)

        return defocus

    def calculate_pixel_exptime(self, pixel_scale): # take an observatory or any pixel scale?
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

    def calculate_adu(self, observatory, band, magnitude, exptime=None):
        """Calculate the number of ADU from the camera.

        Parameters
        __________
        observatory : `leosim.Observatory`
            Observatory viewing the orbital object.
        band : `str`
            Name of filter band.
        magnitude : `float`
            Stationary AB magnitude.
        exptime : `astropy.units.Quantity`, optional
            Exposure time. If None, the pixel traversal exposure time will be 
            used.

        Returns
        -------
        adu : `float`
            Number of ADU.
        """
        if exptime is None:
            exptime = self.calculate_pixel_exptime(observatory.pixel_scale)

        photo_params = observatory.get_photo_params(exptime.to(u.s))
        bandpass = observatory.get_bandpass(band)
        m0_adu = self.sed.calc_adu(bandpass, phot_params=photo_params)
        adu = m0_adu*(10**(-magnitude/2.5))

        return adu

    def get_final_profile(self, psf, observatory, band=None, magnitude=None, exptime=None):
        """Create the convolution of the atmospheric PSF, defocus kernel, and 
        satellite profiles.
        
        Parameters
        ----------
        psf : `galsim.GSObject`
            A surface brightness profile representing an atmospheric PSF.
        observatory : `leosim.Observatory`
            Observatory viewing the orbital object.
        band : `str`, optional
            Name of filter band (None, by default)
        magnitude : `float`, optional
            Stationary AB magnitude (None, by default)
        exptime : `astropy.units.Quantity`, optional
            Exposure time. If None, the pixel traversal exposure time will be
            used.

        Returns
        -------
        final : `galsim.GSObject`
            Final convolution of the component profiles.
        """
        defocus = self.get_defocus_profile(observatory)
        final = galsim.Convolve([self.profile, defocus, psf])

        if (band is not None) and (magnitude is not None):
            adu = self.calculate_adu(observatory, band, magnitude, exptime)
            final = final.withFlux(adu)

        return final

    def get_streak_cross_section(self, psf, observatory, band, magnitude, nx, ny, scale, apply_gain=True):
        """Calculate the cross section of a streak created by the orbital 
        object in an image.

        Parameters
        ----------
        psf : `galsim.GSObject`
            A surface brightness profile representing an atmospheric PSF.
        observatory : `leosim.Observatory`
            Observatory viewing the orbital object.
        band : `str`
            Name of filter band.
        magnitude : `float`
            Stationary AB magnitude        
        nx : `int`
            The x-direction size of the image.
        ny : `int`
            The y-direction size of the image.
        scale : `float`
            Pixel scale of the image.
        apply_gain: `bool`, optional
            If `True`, apply gain (`True`, by default).

        Returns
        -------
        angular_distance : `astropy.units.Quantity`, (nx,)
            Array of angular distances from the streak center.
        cross_section : `astropy.units.Quantity`, (nx,)
            Array of streak cross section signal values per pixel.
        """
        final = self.get_final_profile(psf, observatory, band=band, magnitude=magnitude)
        image = final.drawImage(nx=nx, ny=ny, scale=scale)
        cross_section = np.sum(image.array, axis=0)*observatory.pixel_scale.to_value(u.arcsec/u.pix)/scale
        angular_distance = np.linspace(-int(nx*scale/2), int(nx*scale/2), nx)

        # Set array units
        cross_section *= u.adu/u.pix
        if apply_gain:
            cross_section *= observatory.gain

        angular_distance *= u.arcsec

        return angular_distance, cross_section

    def get_glint_image(self, psf, observatory, band, magnitude, glint_time, nx, ny, scale):
        """Create an image of a glint.

        Parameters
        ----------
        psf : `galsim.GSObject`
            A surface brightness profile representing an atmospheric PSF.
        observatory : `leosim.Observatory`
            Observatory viewing the orbital object.
        band : `str`
            Name of filter band.
        magnitude : `float`
            Stationary AB magnitude
        glint_time : `astropy.units.Quantity`
            Duration of glint.
        nx : `int`
            The x-direction size of the image.
        ny : `int`
            The y-direction size of the image.
        scale : `float`
            Pixel scale of the image.

        Returns
        -------
        glint_image : `galsim.Image`
            Image of a glint.
        """

        final = self.get_final_profile(psf, observatory, band=band, magnitude=magnitude, exptime=glint_time)        
        image = final.drawImage(nx=nx, ny=ny, scale=scale)
        
        pixel_travel = int((glint_time*self.perpendicular_omega).to_value(u.arcsec)/scale)
        glint_array = np.zeros(image.array.shape)
        
        win = scipy.signal.windows.boxcar(pixel_travel)/pixel_travel
        for i in range(image.array.shape[1]):
            glint_array[:, i] = scipy.signal.convolve(image.array[:, i], win, mode='same')
        glint_array = glint_array*np.sum(image.array)/np.sum(glint_array)

        glint_image = galsim.Image(glint_array, scale=image.scale)

        return glint_image

# For now omit phi angle in the child classes until rework
class DiskOrbitalObject(BaseOrbitalObject):
    """A circular disk orbital object.

    Parameters
    ----------
    height : `astropy.units.Quantity`
        Orbital height.
    zenith_angle : `astropy.units.Quantity`
        Observed angle from telescope zenith.
    radius : `astropy.units.Quantity`
        Radius of the orbital object.
    """

    def __init__(self, height, zenith_angle, radius): 
        super().__init__(height, zenith_angle)
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
    zenith_angle : `astropy.units.Quantity`
        Observed angle from telescope zenith.
    width : `astropy.units.Quantity`
        Width of the orbital object.
    length : `astropy.units.Quantity`
        Length of the orbital object.
    """

    def __init__(self, height, zenith_angle, width, length):
        super().__init__(height, zenith_angle)
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
    zenith_angle : `astropy.units.Quantity`
        Observed angle from telescope zenith.
    components : `list` [`leosim.Component`]
        A list of components.

    Raises
    ------
    ValueError
        Raised if ``components`` is of length 0.
    """

    def __init__(self, height, zenith_angle, components):
        super().__init__(height, zenith_angle)

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
