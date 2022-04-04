# derived quantities not included in Aus400 catalogue
# basically all functions here assume the inputs are xarray Dataarrays or Datasets

import numpy as np
import xarray as xr

##----------------------------------------------
##
## Constants
##
##----------------------------------------------

g = 9.81                       # gravitational acceleration
R_e = 6371e3                   # radius of earth
R_d = 287                      # dry gas constant for earth's atmopshere
R_v = 461                      # gas cosntant for water vapour
eps = R_d/R_v                  # ratio of dry/moist gas constants
c_p = 1004                     # specific heat of earth's air (constant pressure)
p_0 = 1e5                      # reference pressure
omega = 2*np.pi/(24*60*60)     # angular velocity of earth
L = 2260e3                     # latent heat of vapourisation of water.

##----------------------------------------------
##
## Dynamics
##
##----------------------------------------------

def rel_vort(u, v):
    """
    vertical component of relative vorticity
    """

    lats_rad, lons_rad = u.latitude * np.pi/180, u.longitude * np.pi/180

    # 180/pi comes out when converting degrees to radians
    dv_dl = v.differentiate('longitude') * 180/np.pi

    rv = u * np.cos(lats_rad)
    rv = -rv.differentiate('latitude') * 180/np.pi
    rv += dv_dl
    rv *= 1/(R_e * np.cos(lats_rad))
    
    return rv


def divergence(u, v):
    
    lats_rad, lons_rad = u.latitude * np.pi/180, u.longitude * np.pi/180
    
    # 180/pi comes out when converting degrees to radians
    du_dl = u.differentiate('longitude') * 180/np.pi

    div = v * np.cos(lats_rad)
    div = div.differentiate('latitude') * 180/np.pi
    div += du_dl
    div *= 1/(R_e * np.cos(lats_rad))
    
    return div


def stretch_def(u, v):
    
    lats_rad, lons_rad = u.latitude * np.pi/180, u.longitude * np.pi/180
    
    # 180/pi comes out when converting degrees to radians
    du_dl = u.differentiate('longitude') * 180/np.pi

    E = v / np.cos(lats_rad)
    E = E.differentiate('latitude') * 180/np.pi
    E *= -np.cos(lats_rad) / R_e
    E += 1/(R_e * np.cos(lats_rad)) * du_dl
    
    return E


def shear_def(u, v):

    lats_rad, lons_rad = u.latitude * np.pi/180, u.longitude * np.pi/180
    
    # 180/pi comes out when converting degrees to radians
    dv_dl = v.differentiate('longitude') * 180/np.pi

    F = u / np.cos(lats_rad)
    F = F.differentiate('latitude') * 180/np.pi
    F *= np.cos(lats_rad) / R_e
    F += 1/(R_e * np.cos(lats_rad)) * dv_dl
    
    return F


def total_def(u, v):
    return stretch_def(u, v)**2 + shear_def(u, v)**2


def okubo_weiss(u, v):
    return rel_vort(u, v)**2 - total_def(u, v) # note total_def is S**2 so no need to square again


def geostrophic_wind(T, p, q):
    # should this be done in spherical coords?

    rho = density(T, p, q)

    lats_rad = T.latitude*np.pi/180
    f = 2 * omega * np.sin(lats_rad)

    # now build up the equation in stages
    u_g = -1/(f*rho) * (p).differentiate('latitude') * 180/np.pi / (R_e)
    v_g = 1/(f*rho) * (p).differentiate('longitude') * 180/np.pi / (R_e * np.cos(lats_rad))
    
    return [u_g, v_g]


def w_to_omega(w, T, q):
    """
    Vertical velocity in pressure coordinates
    Assumes hydrostatic balance
    Inputs must be in pressure coords
    """
    rho = density(T, T.pressure, q)
    return -rho * g * w


def w_to_omega_full(u, v, w, p):
    """
    Vertical velocity in height coordinates (for dp/dz calc, no hydrostatic balance)
    Full equation with no assumptions
    """
    lats_rad = p.latitude*np.pi/180
    
    dp_dt = p.differentiate('time', datetime_unit = 's')
    dp_dx = p.differentiate('longitude') * 180/np.pi / (R_e * np.cos(lats_rad))
    dp_dy = p.differentiate('latitude') * 180/np.pi / R_e
    dp_dz = p.differentiate('level_height') # height_rho
    
    return dp_dt + u * dp_dx + v * dp_dy + w * dp_dz


##----------------------------------------------
##
## Humidity
##
##----------------------------------------------

def dewpoint(p, q):
    """
    Calculates dewpoint temperature from pressure and specific humidity
    """
    r = q / (1-q) # mixing ratio
    e = (p * r) / (r + eps) # vapour pressure
    Td = (4826.56 - 29.65 * np.log(e/611.2)) / (17.67 - np.log(e/611.2)) # Bolton formula
    return Td


def rel_hum(T, p, q):
    """
    Calculates relative humidity from temperature, pressure and specific humidity
    """
    e = vapour_pressure(p, q)
    e_s = saturation_vapour_pressure(T)
    return e / e_s


def mixing_ratio(q):
    """
    Converts specific humdity to mixing ratio
    """
    return q / (1 - q)


def vapour_pressure(p, q):
    """
    Calculates vapour pressure from pressure and specific humidity
    """
    r = mixing_ratio(q)
    e = (p * r) / (r + eps)
    return e


def spec_hum(p, e):
    """
    converts vapour pressure to specific humidity
    """
    r = (eps * e) / (p - e)
    return r / (1 + r)


def saturation_vapour_pressure(T):
    """
    Bolton formula
    """
    return 611.2 * np.exp((17.67 * (T - 273.15)) / (T - 29.65))


def virtual_temp(T, p, q):
    """
    Using the exact formula
    """
    e = vapour_pressure(p, q)
    Tv = T / (1 - (e/p)*(1 - eps))
    return Tv

##----------------------------------------------
##
## Thermodynamics
##
##----------------------------------------------

def dry_density(T, p):
    """
    Using ideal gas law. No moisture considered.
    """
    return p / (R_d * T)


def density(T, p, q):
    """
    Using ideal gas law. Also considers mass of water vapour.
    """
    Tv = virtual_temp(T, p, q)
    return p / (R_d * Tv)


def potential_temp(T, p):
    """
    Potential temperature from temperature/pressure
    Should be the same as 'theta' variable in some aus400 streams
    """
    return T * (p_0/p) ** (R_d/c_p)


def equiv_potential_temp(T, p, q):
    """
    Equivalent potential temperature using Bolton (1980) formula 
    """
    
    Td = dewpoint(p, q)
    r = mixing_ratio(q)
    e = vapour_pressure(p, q)
    
    # temperature at LCL
    T_L = 1 / (1/(Td-56) + (np.log(T/Td))/(800)) + 56
    
    # potential temperature at LCL
    theta_L = T * (p_0/(p-e))**0.2854 * (T/T_L)**(0.28*r)
    
    theta_e = theta_L * np.exp((3036/T_L - 1.78) * r * (1 + 0.448 * r))

    return theta_e


def sat_equiv_potential_temp(T, p):
    """
    Saturated equivalent potential temperature using Bolton (1980) formula
    """
    
    e_s = saturation_vapour_pressure(T)
    q_s = spec_hum(p, e_s)
    r_s = mixing_ratio(q_s)
    
    T_L = T # at saturation, we're already at the LCL
    
    # potential temperature at LCL
    theta_L = T * (p_0/(p-e_s))**0.2854 * (T/T_L)**(0.28*r_s)
    
    theta_es = theta_L * np.exp((3036/T_L - 1.78) * r_s * (1 + 0.448 * r_s))

    return theta_es


def buoy_freq(theta):
    """
    Calculates the Brunt-Vaisala (buoyancy) frequency
    Input theta must be in height coordinates
    """
    return g / theta * theta.differentiate('height_rho')

##----------------------------------------------
##
## Potential vorticity
##
##----------------------------------------------

def pot_vort(u, v, w, T, p, q):
    """
    Potential vorticity calculation
    Can be used with model, height, pressure or theta (isentropic) levels
    """
    
    # calculate potential temperature and density    
    theta = potential_temp(T, p)
    rho = density(T, p, q)
    
    # check in dimensions in the input and call the appropriate subfunction
    if 'level_height' in u.dims:
        return _pot_vort_height(u, v, w, theta, rho, 'level_height')
    elif 'height_rho' in u.dims:
        return _pot_vort_height(u, v, w, theta, rho, 'height_rho')
    elif 'pressure' in u.dims:
        omg = w_to_omega(w, T, q)
        return _pot_vort_pres(u, v, omg, theta)
    elif 'theta' in u.dims:
        return _isen_pot_vort(u, v, p)


def equiv_pot_vort(u, v, w, T, p, q):
    """
    Equivalent potential vorticity calculation
    Can be used with model, height, pressure or theta levels
    """    
    
    # calculate potential temperature and density    
    theta_e = equiv_potential_temp(T, p, q)
    rho = density(T, p, q)
    
    # check in dimensions in the input and call the appropriate subfunction
    if 'level_height' in u.dims:
        return _pot_vort_height(u, v, w, theta_e, rho, 'level_height')
    elif 'height_rho' in u.dims:
        return _pot_vort_height(u, v, w, theta_e, rho, 'height_rho')
    elif 'pressure' in u.dims:
        omg = w_to_omega(w, T, q)
        return _pot_vort_pres(u, v, omg, theta_e)
    elif 'theta' in u.dims:
        return _isen_pot_vort(u, v, p) # I'm guessing this is probably different somehow? There needs to be a theta_e/q there somewhere...


def _pot_vort_height(u, v, w, theta, rho, vert_dim):
    
    lats_rad = u.latitude * np.pi/180
    f = 2 * omega * np.sin(lats_rad)
    
    # calculate necessary derivatives for PV
    
    dtheta_dx = theta.differentiate('longitude') * 180/np.pi / (R_e * np.cos(lats_rad))
    dtheta_dy = theta.differentiate('latitude') * 180/np.pi / R_e
    dtheta_dz = theta.differentiate(vert_dim)

    du_dy = u.differentiate('latitude') * 180/np.pi / R_e
    du_dz = u.differentiate(vert_dim)

    dv_dx = v.differentiate('longitude') * 180/np.pi / (R_e * np.cos(lats_rad))
    dv_dz = v.differentiate(vert_dim)

    dw_dx = w.differentiate('longitude') * 180/np.pi / (R_e * np.cos(lats_rad))
    dw_dy = w.differentiate('latitude') * 180/np.pi / R_e

    
    PV = (dtheta_dx * (dw_dy - dv_dz) + dtheta_dy * (du_dz - dw_dx) + dtheta_dz * (f + dv_dx - du_dy)) / rho
    return PV


def _pot_vort_pres(u, v, omg, theta):
    
    lats_rad = u.latitude * np.pi/180
    f = 2 * omega * np.sin(lats_rad)
    
    # calculate necessary derivatives for PV
    
    dtheta_dx = theta.differentiate('longitude') * 180/np.pi / (R_e * np.cos(lats_rad))
    dtheta_dy = theta.differentiate('latitude') * 180/np.pi / R_e
    dtheta_dp = theta.differentiate('pressure')

    du_dy = u.differentiate('latitude') * 180/np.pi / R_e
    du_dp = u.differentiate('pressure')

    dv_dx = v.differentiate('longitude') * 180/np.pi / (R_e * np.cos(lats_rad))
    dv_dp = v.differentiate('pressure')

    dw_dx = omg.differentiate('longitude') * 180/np.pi / (R_e * np.cos(lats_rad))
    dw_dy = omg.differentiate('latitude') * 180/np.pi / R_e

    PV = -g * (dtheta_dx * (dw_dy - dv_dp) + dtheta_dy * (du_dp - dw_dx) + dtheta_dp * (f + dv_dx - du_dy))
    return PV


def _isen_pot_vort(u, v, p):
    
    lats_rad = u.latitude * np.pi/180
    f = 2 * omega * np.sin(lats_rad)
    
    # calculate necessary derivatives for PV
    du_dy = u.differentiate('latitude') * 180/np.pi / R_e
    dv_dx = v.differentiate('longitude') * 180/np.pi / (R_e * np.cos(lats_rad))
    dtheta_dp = (p.differentiate('theta')) ** -1

    PV = -g * (f + dv_dx - du_dy) * dtheta_dp
    return PV


##----------------------------------------------
##
## Dimensionless parameters
##
##----------------------------------------------

def scorer_param(U, theta_0):
    """
    Calculates the Scorer parameter, used to determine the presence/nature of gravity waves over mountains
    U is the basic-state wind speed approaching the mountains, theta is basic-state potential temperature
    Inputs must be in height coordinates
    """
    
    N02 = buoy_freq(theta_0)
    U_zz = U.differentiate('height_rho').differentiate('height_rho')
    
    return N02 / U**2 - U_zz / U


##----------------------------------------------
##
## Coordinate transforms
##
##----------------------------------------------


def winds_to_polar(u, v, centre_pos):
    """
    converts u/v to polar coordinates, i.e. radial/tangential velocity
    The new origin is defined at centre_pos
    """
    
    # define theta grid (no need for r grid here)
    theta = np.arctan2((u.latitude-centre_pos.latitude), (u.longitude-centre_pos.longitude))
    
    v_r = u * np.cos(theta) + v * np.sin(theta)
    v_theta = v * np.cos(theta) - u * np.sin(theta)
    
    return v_r, v_theta


def cart_to_pol(var, centre_pos, r_round=2):
    """
    Add new coordinates (r, theta) for input data
    Can still keep the data in (lon, lat) but can also do stuff with polar coords
    r is measure in degrees here
    """
    
    r = ((var.latitude-centre_pos.latitude)**2 + (var.longitude-centre_pos.longitude)**2) ** 0.5
    r = r.round(r_round)
    
    theta = np.arctan2((var.latitude-centre_pos.latitude), (var.longitude-centre_pos.longitude))
    
    var = var.assign_coords(r=r)
    var = var.assign_coords(theta=theta)
    
    return var


def cart_to_pol_dist(var, centre_pos, r_round=2):
    """
    Add new coordinates (r, theta) for input data
    Can still keep the data in (lon, lat) but can also do stuff with polar coords
    Now r is measured in km rather than degrees
    """
    Re = 6371e3
    
    dlon = (var.longitude - centre_pos.longitude) * np.pi/180
    dlat = (var.latitude - centre_pos.latitude) * np.pi/180
    
    # in km
    dx = (Re * np.cos(var.latitude * np.pi/180) * dlon) / 1000
    dy = (Re * dlat) / 1000
    
    
    r = (dx**2 + dy**2) ** 0.5
    r = r.round()#r_round)
    
    theta = np.arctan2((var.latitude-centre_pos.latitude), (var.longitude-centre_pos.longitude))
    
    var = var.assign_coords(r=r)
    var = var.assign_coords(theta=theta)
    
    return var


##----------------------------------------------
##
## Tracking
##
##----------------------------------------------


def find_min_pos(data):
    """
    Find the minimum index and value in the input data
    Returns the results as an xarray DataArray
    """
    
    argmin = data.argmin(...)
    min_pos = data.isel(latitude=argmin['latitude'], longitude=argmin['longitude'])

    # tie-breaker: just pick one
    if min_pos.latitude.size > 1:
        min_pos = min_pos.isel(latitude=0)
    if min_pos.longitude.size > 1:
        min_pos = min_pos.isel(longitude=0)        
    
    return min_pos


def cyclone_centre(mslp, thresh=500):
    """
    Finds the "centre" of the TC
    First find the point of min p, then take a 'disc' of values near the minimum.
    The centre is at the middle of this disc.
    
    - thresh: disc is min(mslp) + thresh in Pa
    """
    min_mslp = find_min_pos(mslp)
    
    mslp_disc = mslp.where(mslp < min_mslp + thresh, drop=True)
    
    mslp_disc_centre = mslp_disc.sel({'latitude': mslp_disc.latitude.mean(), 'longitude': mslp_disc.longitude.mean()}, method='pad')
    
    return mslp_disc_centre


