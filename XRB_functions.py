
import os
import sys

import numpy as np
import matplotlib.pyplot as plt                              # type: ignore
from scipy.optimize import newton                            # type: ignore 
import pandas as pd                                          # type: ignore

import posydon.utils.constants as const                      # type: ignore
from posydon.utils.common_functions import roche_lobe_radius # type: ignore
from posydon.utils.posydonwarning import Pwarn               # type: ignore

# !  For the moment, all the functions (except XRB_selection_function()) use scalar inputs only. Changes have to be done in order to support array inputs. 
# !  It will allow to speed up the calculations when dealing with large populations.

#TODO : Rewrite all of these functions in a more efficient way within a new file XRB_functions_NEW.py. 
#       Allows side functions to read array type data in order to compute all the required elements more rapidly

#* Checked and DocString updated
def BH_efficiency(mass, spin): 
    """
    Compute the radiative efficiency of a Kerr black hole (or Schwarzschild BH if spin = 0)
    as a function of the spin parameter *a* and the mass according to Novikov-Thorne radiative efficiency [[1]_] .
    
    Parameters
    ---------- 
    mass : float
        Mass of the black hole in [M⊙].
    spin : float
        Spin of the black hole. Can be given as dimensionless spin "*a**" [c*J/(GM^2)] or dimensional spin "*a*" [J/M].
        Its value can be given either positively (prograde accretion) or negatively (retrograde accretion) 
        (spin  ∈ [-0.998, 0.998] with ± 0.998 being the Thorne limit).

    Returns
    -------
    tuple
        A tuple with three floats: (E_ISCO, r_tilde, eta).
        - E_ISCO (float): Specific energy at the ISCO, in units of mc^2.
        - r_tilde (float): Radius of the ISCO in units of GM / c^2.
        - eta (float): Radiative efficiency (1 - E_ISCO), dimensionless.

    References
    ----------
    .. [1] Bambi, C., Malafarina, D., & Tsukamoto, N. (2014).
            "Note on the radiation emitted by a thin accretion disk around a rotating compact object."
            *Phys. Rev. D*, 89(12), 127302. https://arxiv.org/abs/1406.2181
    """
    
    # To control that the spin is in the range that gives the maximal efficiency
    a_star = float(np.clip(spin, -0.998, 0.998))

    #=== Compute efficiency [1]=== 
    
    # Use sign = +1 for spin ≥ 0 (prograde) and -1 otherwise (retrograde)
    sign = 1 if a_star >= 0 else -1

    Z1 = 1 + (1 - a_star**2)**(1/3) * ((1 + a_star)**(1/3) + (1 - a_star)**(1/3))
    Z2 = np.sqrt(3 * a_star**2 + Z1**2)

    # Radius at the innermost stable circular orbit (ISCO)
    r_tilde = 3 + Z2 - sign * np.sqrt((3 - Z1)*(3 + Z1 + 2 * Z2)) # Dimensionless

    # Energy at ISCO in units of mc^2 (i.e E_ISCO < 1, E = 1 means particle energy at infinity)
    E_ISCO = (r_tilde**(3/2) - 2 * r_tilde**(1/2) + a_star)/(r_tilde**(3/4) * np.sqrt(r_tilde**(3/2) - 3 * r_tilde**(1/2) + 2 * a_star))
    
    # Radiative efficiency
    eta = 1 - E_ISCO

    return E_ISCO, r_tilde, eta

#* Checked and DocString updated
def NS_efficiency(mass,log_radius, **kwargs):
    """
    Compute the radiative efficiency of a neutron star as a function of its mass and radius. 
    
    If there is a magnetic field, it will compare the magnetospheric radius with the radius of the neutron star to compute the accretion radius accordingly to [[1]_]. 
    The strength of the magntic field affects the magnetospheric radius of the NS and therefore, if the magnetospheric radius is larger than the NS radius, 
    the efficiency is lowered because the accretion disk is truncated and accretion happens along the magnetic lines up to the poles on smaller surface area.
    This results as a beaming effect of the X-ray emission due to smaller accretion area.

    Parameters
    ----------
    mass : float
        Mass of the neutron star in [M⊙]
    log_radius : float
        Log10(Radius of the neutron star) in [R⊙]
    **kwargs : dict, optional
        - B_surf (float) : Surface magnetic field strength of the neutron star in [G]. If provided along with accretor_logMdot, the magnetosphere radius will be computed.
        - accretor_logMdot (float) : Log10(Mass accretion rate onto the neutron star) in [M⊙/yr]. Required if B_surf is provided.

    Returns
    -------
    eta : float
        Radiative efficiency of the neutron star, dimensionless.
    R : float
        Accretion radius used for the efficiency computation in [cm]. It is either the NS radius or the magnetosphere radius if B_surf is provided and R_mag > R_NS.

    References
    ----------
    .. [1] Misra et al. (2024), Astronomy & Astrophysics, 682, A69,
           **Exploring the nature of Ultra-luminous X-ray sources across stellar population ages using detailed binary evolution calculations**
           https://doi.org/10.1051/0004-6361/202347880

    """
    #===== Optional parameters =====
    allowed_keys = {'B_surf', 'accretor_logMdot'}

    B_surf = kwargs.get('B_surf', None)  # Magnetic field [G]
    accretor_logMdot = kwargs.get('accretor_logMdot', None)  # [M⊙/yr]

    if not allowed_keys.issuperset(kwargs.keys()):
        raise TypeError(f"NS_efficiency: unknown parameter(s): {sorted(set(kwargs.keys()) - allowed_keys)}")
    #===== Mass =====
    if mass is None or not np.isfinite(mass) or mass <= 0.0:
            mass = 1.45 # Default mass of a NS in [M⊙]
    
    M = float(mass) * const.Msun # [gr]

    #===== Radius =====
    if log_radius is None or not np.isfinite(log_radius):
            # fallback radius ~12.5 km -> 1.25e6 cm
            R_Ns_cm = 1.25e6
            
    else:
            R_Ns_sun = 10.0**float(log_radius)   # [R⊙]
            R_Ns_cm = R_Ns_sun * const.Rsun # [cm]
            

    # If there is a magnetic field, compute the magnetosphere radius R_mag (POSYDON do not provide B field for now)
    R_mag = 0.0
    if B_surf is not None and accretor_logMdot is not None:
        mu = B_surf * R_Ns_cm**3  # Magnetic dipole moment [G cm^3]
        Mdot_NS = 10**accretor_logMdot * const.Msun / const.secyer  # [g/s]
        R_mag = (mu**4 / (2 * const.standard_cgrav * M * Mdot_NS**2))**(1/7)  # [cm]

    # Choose the larger radius between R_NS and R_mag for the accretion radius
    if R_mag > R_Ns_cm:
        R = R_mag # Magnetic field truncates the accretion disk
    else:
        R = R_Ns_cm # Magnetosphere radius inside the NS radius
  
    #===== Efficiency computation =====
    eta = const.standard_cgrav * M /(R * const.clight**2) # dimensionless ~0.1-0.3
    
    return eta, R 

#* Checked and DocString updated
def Orbital_sep_and_velo_from_period(Mass1, Mass2, period_days):
    """
    Calculate the orbital separation "a" given the period of the system from the 3rd Kepler's law. 
    And calculate also the orbital velocity of the companion/donor star.

    Parameters 
    ---------
    Mass1 : float 
        Mass of the central body/accretor where Mass1 > Mass2 in [M⊙]
    Mass2 : float
        Mass of the orbiting object/donor where Mass2 < Mass1 in [M⊙]
    period_days : float 
        Period of the orbiting companion in [days]
    
    Returns
    -------
    a_cm : float
        Orbital separation of the binary system in [cm]
    V_orb : float
        Orbital velocity of the companion/donor star in [cm/s]
    """
    #===== Control parameters =====
    if np.any(Mass1 < 0.0) or np.any(Mass2 < 0.0):
        raise ValueError("Mass1 and Mass2 must be non-negative values.")
    if np.any(period_days <= 0.0):
        raise ValueError("Orbital period must be a positive value.")
    if period_days is None:
        return np.nan, np.nan

    #===== Calculation =====
    period_seconds = period_days * const.day2sec
    M1_gr = Mass1 * const.Msun
    M2_gr = Mass2 * const.Msun
    a_cm = (const.standard_cgrav * (M1_gr + M2_gr) * period_seconds**2 /(4 * np.pi**2))**(1/3) # [cm]

    V_orb = 2 * np.pi * a_cm / (float(period_seconds)) # [cm/s]

    return a_cm, V_orb

#* Checked and DocString updated

def BHL_mdot_acc(Macc_Msun, donor_mass_Msun, accretor_type, 
                 Mdot_w_Msun_per_yr, a_cm, ecc, V_wind, **kwargs):
    """
    Calculate the Bondi-Hoyle Littleton accretion rate through winds in gram per second.\n 
    User can choose if the winds, in presence of a black hole, must form a disk before being accreted with the keyword 'Wind_Disk_BH_formation'. 

    Parameters
    ----------
    Macc_Msun : float 
        Mass of the accretor in [M⊙]
    donor_mass_Msun : float
        Mass of the donor in [M⊙]
    accretor_type : str
        Type of the accretor. Can be either 'NS' or 'BH'.
    Mdot_w_Msun_per_yr : float
        Accretion rate from winds in [M⊙/yr] (Usually represents the difference between total mass loss 
        rate from the donor and the mass transfer rate through RLO)
    a_cm : float
        Orbital separation of the binary system in [cm]
    ecc : float
        Orbital eccentricity of the donor star, dimensionless (ecc ∈ [0, 0.999])
    V_wind : float
        Wind velocity from the donor to the accretor [cm/s]
    **kwargs : dict, optional
        Additional keyword arguments:
        - alpha_BHL (float, default = 1.5) : Alpha parameter of Bondi-Hoyle Littleton (alpha_BHL ∈ [1, 2]).
        - Wind_Disk_BH_formation (bool, default = False) : Whether to check for disk formation criteria around BH accretor.
        - r_ISCO (float) : ISCO radius [cm] of a BH accretor (required if Wind_Disk_BH_formation=True).
        - eta (float) : Radiative efficiency (dimensionless) (required if Wind_Disk_BH_formation=True).
        
        Any unknown keys will raise a TypeError.
            
    Returns
    -------
    Mdot_BHL: float 
        Accretion rate of Bondi-Hoyle Littleton through winds in [gr/s]  

    References
    ----------
    .. [1] Bondi, H., & Hoyle, F. 1944, MNRAS, 104, 273
    .. [2] Sen, K. ,Xu, X. -T., Langer, N., El Mellah, I. , Schurmann, C., Quast, M., 2021, A&A,
           Equation 10
    .. [3] Frank, J., King, A., & Raine, D. (2002). "Accretion Power in Astrophysics: Third Edition" Chapter 4.9 pages 73-74.
           *Cambridge University Press*. https://doi.org/10.1017/CBO9781139164417
    .. [4] Bambi, C., Malafarina, D., & Tsukamoto, N. (2014).
            "Note on the radiation emitted by a thin accretion disk around a rotating compact object."
            *Phys. Rev. D*, 89(12), 127302. https://arxiv.org/abs/1406.2181
    """
    #===== Optional parameters =====
    allowed_keys = {'alpha_BHL', 'Wind_Disk_BH_formation', 'r_ISCO', 'eta'}

    alpha_BHL = kwargs.get('alpha_BHL', 1.5) # Alpha parameter of Bondi-Hoyle Littleton
    if not (1.0 <= alpha_BHL <= 2.0):
        raise ValueError("alpha_BHL must be in the range [1.0, 2.0]")
    
    Wind_Disk_BH_formation = kwargs.get('Wind_Disk_BH_formation', False)
    r_ISCO = kwargs.get('r_ISCO', None)
    eta = kwargs.get('eta', None)

    if Wind_Disk_BH_formation and accretor_type == 'BH':
        if r_ISCO is None or eta is None:
            raise ValueError("r_ISCO and eta must be provided when Wind_Disk_BH_formation is True.")
        
    if not allowed_keys.issuperset(kwargs.keys()):
        raise TypeError(f"BHL_mdot_acc: unknown parameter {sorted(set(kwargs.keys()) - allowed_keys)}")
    
    #===== Control parameters =====
    if a_cm is None or not np.isfinite(a_cm) or a_cm <= 0.0:
        return 0.0
    if V_wind is None or not np.isfinite(V_wind) or V_wind < 0.0:
        return 0.0
    
    #===== Masses =====
    Macc = float(Macc_Msun) * const.Msun # [gr]
    Mdon = float(donor_mass_Msun) * const.Msun # [gr]
    
    #===== Orbital mechanics (Keplerian) =====
    ecc = 0.0 if (ecc is None or not np.isfinite(ecc)) else float(np.clip(ecc, 0.0, 0.999))

    # Orbital frequency [rad/s]
    Omega = np.sqrt(const.standard_cgrav * (Mdon + Macc) / a_cm**3)
    # Random orbital phase [s]
    t0 = np.random.rand() * (2.0 * np.pi / Omega)
    # Eccentric anomaly at time t0 from Kepler's equation
    E = newton(lambda x: x - ecc * np.sin(x) - Omega * t0, np.pi/2, maxiter=100)
    b = a_cm * np.sqrt(1.0 - ecc**2)
    r_vec = np.array([a_cm * (np.cos(E) - ecc), b * np.sin(E)])  # Radius in vectorial form [cm]
    v_dir = np.array([-a_cm * np.sin(E), b * np.cos(E)]) # Velocity direction vector [cm] (derivative of r_vec)
    R = np.linalg.norm(r_vec, axis=0)  # Distance between the two stars [cm]
    v_dir_norm = np.linalg.norm(v_dir, axis=0)
    if R > 0.0 and v_dir_norm > 0.0:
        cos_angle = np.dot(v_dir, r_vec) / (v_dir_norm * R)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
    else:
        cos_angle = 0.0

    V_orb = np.sqrt(const.standard_cgrav * (Mdon + Macc) * (2.0 / R - 1.0 / a_cm))  # Relative velocity [cm/s]

    # Total relative velocity of the winds [3]
    V_rel = np.sqrt(V_orb**2 + V_wind**2 + 2.0 * V_orb * V_wind * cos_angle)
    
    # Accreted material from winds
    Mdot_w = float(max(Mdot_w_Msun_per_yr, 0.0)) * const.Msun / const.secyer  # [g/s]
   
    # From [1], we calculate the BHL accretion rate
    if V_wind == 0.0:
        frac = 1.0
        Pwarn(f"Warning: Wind velocity is zero, setting accretion fraction to 1.0 (PID={os.getpid()})", 
              category="InappropriateValueWarning")
    elif V_rel == 0.0:
        frac = 0.0
        Pwarn(f"Warning: Orbital velocity is zero, setting BHL accretion rate to 0.0 g/s (PID={os.getpid()})", 
              category="InappropriateValueWarning")
    elif R == 0.0:
        frac = 0.0
        Pwarn(f"Warning: Orbital separation is zero, setting BHL accretion rate to 0.0 g/s (PID={os.getpid()})", 
              category="InappropriateValueWarning")
    else:
        frac = alpha_BHL * (const.standard_cgrav * Macc)**2 / (2 * V_rel**3 * V_wind * R**2)
    
    Mdot_BHL = frac * Mdot_w  # [g/s]

    # From [2], check if an accretion disk can form around the BH accretor
    if Wind_Disk_BH_formation and accretor_type == 'BH':
        gamma = r_ISCO/(6.0 * const.standard_cgrav * Macc / const.clight**2) # From [4]
        q = Mdon / Macc
        Rdisk_div_RISCO = ((2/3) * (eta / (1 + q))**2 * (V_orb /(const.clight * 0.01))**(-2) 
                           * (1 + (V_wind/V_orb)**2)**(-4) / gamma) 
        if Rdisk_div_RISCO < 1.0: # No disk formed
            Mdot_BHL = 0.0 # [g/s]

    return float(Mdot_BHL)

#* Checked and DocString updated
def Eddington_limit(accretor_mass, Donor_surface_H1, eta):
    """
    Compute the Eddington accretion rate from the accretor mass, its accretion efficieny and the content of hydrogen at the surface of the donor star.
    Also compute the Eddington luminosity.

    Parameters
    ----------
    accretor_mass : float
        Mass of the accretor in [M⊙].
    Donor_surface_H1 : float
        Hydrogen mass fraction of the donor star's surface (surface_h1_f).
    eta : float
        Radiative efficiency of the accretor (NS or BH), dimensionless.

    Returns
    -------
    Mdot_Edd_gps : float
        Eddington accretion rate in [g/s].
    L_edd : float 
        Eddington luminosity in [erg/s].
    
    """
    L_edd = 4.0 * np.pi * const.standard_cgrav * accretor_mass * const.Msun * const.clight /(0.2 * (1.0 + Donor_surface_H1)) # [erg/s]

    Mdot_Edd_gps = L_edd / (eta * const.clight**2) # [g/s]
    

    return Mdot_Edd_gps, L_edd

#* Checked and DocString updated

def Wind_velocity_BHL(accretor_mass, accretor_type, donor_mass, donor_logR, donor_logL, Log_Mdot_wind, 
                      donor_surface_H1, donor_He_core_mass, a_cm, 
                      Wind_fed_scheme = None, Wind_velo_scheme ='Hurley+2002'):
    '''
    Compute the wind velocity from the donor/companion star in cm/s.
    For wind-fed systems, a wind scheme calculation can be chosen and for donor star a filling threshold for RLO can be enabled [[1]_][[4]_]

    Parameters
    ----------
    accretor_mass : float
        Mass of the accretor in [M⊙]. Used to compute the Roche lobe radius.
    accretor_type : str
        Type of the accretor. Can be either 'NS' or 'BH'.
    donor_mass : float
        Mass of the donor star in [M⊙]
    donor_logR : float
        Log10(Radius of the donor star) in [R⊙]
    donor_logL : float
        Log10(Luminosity of the donor star) in [L⊙]
    donor_surface_H1 : float
        Hydrogen mass fraction of the donor star's surface (surface_h1_f).
    donor_He_core_mass : float
        Helium core mass of the donor star in [M⊙]
    a_cm : float
        Orbital separation of the binary system in [cm]
    Wind_fed_scheme : str, (default=None)
        If 'HiraiMandel21', the donor star must fill at least 80% of its Roche lobe to get winds [[4]_]. Default is None.
    Wind_velo_scheme : str (default='Hurley+2002')
        Scheme to compute the wind velocity. Can be either 'Hurley+2002' or 'Kudritzki+2000'.
    
    Returns
    -------
    V_wind : float
        Wind velocity from the donor to the accretor [cm/s].

    References
    ----------
    .. [1] Misra et al. A&A 672, A99 (2023) https://arxiv.org/abs/2209.05505v2
    .. [2] Hurley, J. R., Tout, C. A., & Pols, O. R. 2002, MNRAS, 329, 897
    .. [3] Kudritzki, R.-P., & Puls, J. 2000, ARA&A, 38, 613
    .. [4] Hirai, R. & Mandel, I. (2021).  
           *Conditions for accretion disc formation and observability of wind-accreting X-ray binaries.*  
           Publications of the Astronomical Society of Australia, 38, e056.  
           doi:10.1017/pasa.2021.53  
           https://doi.org/10.1017/pasa.2021.53
    .. [5] Sander A. A. C., Vink J. S., 2020, MNRAS, 499, 873
    '''

    RL_donor_Rsun = roche_lobe_radius(donor_mass, accretor_mass, a_cm) / const.Rsun # [R⊙] 

    # In [1] Misra et al. A&A 672. A99 (2023), based on [4] Hirai & Mandel 2021
    if (accretor_type=='BH' and 10**donor_logR < 0.8 * RL_donor_Rsun and Wind_fed_scheme == 'HiraiMandel21'): # Must fill at least 80% of its Roche lobe to get winds
        V_wind = 0.0 # No wind
    else:

        Teff = stefan_boltzmann_law(10.0**donor_logL, 10.0**donor_logR) # [K]

        #===== Wind velocity calculation schemes =====
        # [2] Hurley et al. 2002
        if Wind_velo_scheme == 'Hurley+2002':
            # For H-rich stars
            if donor_surface_H1 > 0.01:
                if donor_He_core_mass > 0.0 and 10.0**donor_logR > 900.0:
                    beta = 0.125
                elif donor_mass > 120.0:
                    beta = 7.0
                elif donor_mass < 1.4:
                    beta = 0.5
                else:
                    beta = 0.5 + (donor_mass - 1.4) / (120.0 - 1.4) * (6.5)
            else:  # For He-rich stars
                if donor_mass > 120.0:
                    beta = 7.0
                elif donor_mass < 10.0:
                    beta = 0.125
                else:
                    beta = 0.125 + (donor_mass - 10.0) / (120.0 - 10.0) * (6.875)

            f_m = float(np.sqrt(beta))

        # [3] Kudritzki & Puls 2000
        elif Wind_velo_scheme == 'Kudritzki+2000':
            if Teff >= 21000:
                f_m = 2.65
            elif Teff <= 10000:
                f_m = 1.0
            else:
                f_m = 1.4
        else:
            raise ValueError(f"Invalid scheme '{Wind_velo_scheme}'. Choose 'Hurley+2002' or 'Kudritzki+2000'.")

        V_esc = np.sqrt(2.0 * const.standard_cgrav * donor_mass * const.Msun / (10**donor_logR * const.Rsun)) # [cm/s] 
        V_wind = f_m * V_esc  # [cm/s]

        # Sander A. A. C., Vink J. S., 2020, MNRAS, 499, 873 [5]
        if donor_surface_H1 < 0.4 and Teff >= 1e4:
            if Log_Mdot_wind >= -5.25:
                slope = (3.7 - 3.25) / (-2.5 + 5.25)
            else:
                slope = (3.25 - 3.75) / (-5.25 + 7.25)
            V_wind = 10 ** (slope * Log_Mdot_wind + 3.25 + 5.25 * slope) * 1e5  # Convert from km/s to cm/s
        else:
            pass

    return V_wind

#* Checked and DocString updated
def Be_XRB_Lx(orbital_period_days, donor_mass, donor_radius, accretor_mass):
    '''
    Compute the X-ray luminosity of a system containing a Be star as companion. The donor star should have [[1]_]:
    - Surface velocity higher than 70% of its critical velocity
    - Orbital period between 10 and 300 days
    - Its state should be detached and not undergoing RLO
    - Donor mass of at least 6 M⊙
    - A decretion disc 100 times bigger than the star itself and it must go beyond the Roche-Lobe radius of the donor star

    Be stars have fast rotation surface which, when the surface velocity is close to the critical velocity,
    can form a decretion disk around them. The compact object (NS or BH) can then accrete material from this disk
    if it passes through the Roche lobe of the Be star and thus leading to X-ray emission when accreted.

    Parameters
    ----------
    orbital_period_days : float
        Orbital period of the binary system in [days].
    donor_mass : float
        Mass of the donor star in [M⊙].
    donor_radius : float
        Radius of the donor star in [R⊙]
    accretor_mass : float
        Mass of the accretor in [M⊙]

    Returns
    -------
    Lx : float
        X-ray luminosity of the system [erg/s].
    
    References
    ----------
    .. [1] Misra, D., Kovlakas, K., Fragos, T., Lazzarini, M., Bavera, S. S.,  
           Lehmer, B. D., Zezas, A., Zapartas, E., Xing, Z., Andrews, J. J., Dotter, A.,  
           Rocha, K. A., Srivastava, P. M., Sun, M. (2023).  
           *X-ray luminosity function of high-mass X-ray binaries: Studying the signatures of different physical processes using detailed binary evolution calculations*. 
           https://arxiv.org/abs/2209.05505v2 
    .. [2] Dai, Hai-Lang ;  Liu, Xi-Wei ;  Li, Xiang-Dong (2006). 
           *Exploration of the Ps-Porb Relation for Wind-fed X-Ray Pulsars*.
           https://doi.org/10.1086/508735

    '''
    
    #===== Calculation =====
    a_cm, _ = Orbital_sep_and_velo_from_period(accretor_mass, donor_mass, orbital_period_days) 
    RL_donor = roche_lobe_radius(donor_mass, accretor_mass, a_cm) # [cm]

    # Check if decretion disk radius is larger than 100 * donor's radius, from Misra et al. 2023 (A&A 672, A99), page 4
    Decretion_radii = (100 * donor_radius) > RL_donor
    
    # Based on the work of [1] Misra+2023 and [2] Dai+2006

    if 10.0 <= orbital_period_days <= 300.0 and Decretion_radii:
    
        Lx_Be = 1e35 * (10**(4.53))/((orbital_period_days)**(1.5)) # [erg/s]
    else:
        Lx_Be = 0.0

    return Lx_Be

#* Checked and DocString updated
def stefan_boltzmann_law(Luminosity, Radius):
    """
    Compute the effective temperature from the luminosity and radius of 
    a star using the Stefan-Boltzmann law.
    
    Parameters
    ----------
    Luminosity : float
        Luminosity of the star in [L⊙]
    Radius : float
        Radius of the star in [R⊙]

    Returns
    -------
    Teff : float
        Effective temperature of the star in [K].

    """

    #===== Check inputs =====
    if Luminosity <= 0.0 or Radius <= 0.0:
        raise ValueError("Luminosity and Radius must be positive values in units of L⊙ and R⊙ respectively.")
    
    #===== Calculation =====
    Teff = ((Luminosity * const.Lsun) / (4.0 * np.pi * (Radius * const.Rsun)**2.0 * const.boltz_sigma))**(1.0/4.0)
    
    return Teff

#* Checked and DocString updated
def Compute_Lx(donor_mass, donor_log_R, donor_logL, accretor_mass, accretor_log_R, accretor_type, 
               donor_log_mdot, log_Mtransfer_rate, accretor_log_mdot,
               Donor_H1_surf, Donor_He_core_mass, orbital_period_days, ecc, accretor_spin  = 0.0, 
               **kwargs): 
    """
    Compute the X-ray luminosity of a binary system as long as there is a compact accretor (NS or BH)
    and a donor star losing mass through Roche lobe overflow (RLO) and/or winds. The state of the system
    is saved as well (sub-Eddington, super-Eddington) according to the beaming of the total system.
    
    Parameters
    ----------
    donor_mass : float
        Mass of the donor star in [M⊙].
    donor_log_R : float
        Log10(Radius of the donor star) in [R⊙].
    donor_logL : float
        Log10(Luminosity of the donor star) in [L⊙].
    accretor_mass : float
        Mass of the accretor in [M⊙].
    accretor_log_R : float
        Log10(Radius of the accretor) in [R⊙].
    accretor_type : str
        Type of the accretor. Can be either 'NS' or 'BH'.
    donor_log_mdot : float
        Log10(Mass loss rate from the donor star) in [M⊙/yr]. Is the total mass loss rate from the donor star including both RLO and winds.
    log_Mtransfer_rate : float
        Log10(Mass transfer rate) in [M⊙/yr]. A fraction of donor_log_mdot.
    accretor_log_mdot : float
        Log10(Mass accretion rate onto the accretor) in [M⊙/yr]. Is the total mass accretion rate onto the accretor including both RLO and winds.
    Donor_H1_surf : float
        Hydrogen mass fraction of the donor star's surface (surface_h1_f).
    Donor_He_core_mass : float
        Helium core mass of the donor star in [M⊙] (he_core_mass_f).
    orbital_period_days : float
        Orbital period of the binary system in [days] initially set a 15 days. If not given, wind accretion cannot be computed.
    ecc : float
        Orbital eccentricity of the system.
    accretor_spin : float
        Spin of the accretor in the case of BH accretor. Initially set at 0.0 (non spinning BH (Schwarzchild)).
    **kwargs : dict, optional
        Additional keyword arguments:
        - alpha_BHL (float, default=1.5) : efficiency parameter for Bondi-Hoyle-Lyttleton accretion.
        - Wind_fed_scheme (str, default=None) : Check whether the donor star fills at least 80% of its Roche lobe to get winds. Option : 'HiraiMandel21' chooses ≥ 80% filled systems.
        - Wind_velo_scheme (str, default='Hurley+2002') : Wind velocity scheme to use in Wind_velocity_BHL function. 
        Can be either 'Hurley+2002' or 'Kudritzki+2000'.
        - Wind_Disk_BH_formation (bool, default=False) : Check whether for disk formation criteria around BH accretor.

        Any unknown keys will raise a TypeError.

    Returns
    -------
    Data : dict
    
        - Lx_Total : float
            Total X-ray luminosity in [erg/s].
        - Beaming : float
            Beaming factor (dimensionless).
        - Edd_state : str
            State of the system according to its Eddington limit. Can be either 'Sub-Eddington' or 'Super-Eddington'.
        - Mdot_BHL_gps : float
            Wind accretion rate [g/s].
        - Mdot_RLO_gps : float
            RLO mass-transfer rate delivered to the accretor vicinity [g/s]. 

    References
    ----------
    .. [1] Misra, D., Kovlakas, K., Fragos, T., Lazzarini, M., Bavera, S. S.,  
           Lehmer, B. D., Zezas, A., Zapartas, E., Xing, Z., Andrews, J. J., Dotter, A.,  
           Rocha, K. A., Srivastava, P. M., Sun, M. (2023).  
           *X-ray luminosity function of high-mass X-ray binaries: Studying the signatures of different physical processes using detailed binary evolution calculations*.  
           https://arxiv.org/abs/2209.05505v2
    .. [2] Shakura, N. I. & Sunyaev, R. A. (1973).  
           *Black holes in binary systems. Observational appearance.*  
           Astronomy and Astrophysics, 24, 337.  
           https://ui.adsabs.harvard.edu/abs/1973A%26A....24..337S
    .. [3] King, A. R. (2008).  
           *Accretion in compact binaries: the Eddington limit and beaming.*  
           Monthly Notices of the Royal Astronomical Society, 385, L113.  
           https://ui.adsabs.harvard.edu/abs/2008MNRAS.385L.113K
    """

    #===== Optional parameters =====
    allowed_keys = {'alpha_BHL', 'Wind_fed_scheme', 'Wind_velo_scheme', 'Wind_Disk_BH_formation', 'B_surf'}

    alpha_BHL = kwargs.get('alpha_BHL', 1.5)
    if alpha_BHL < 1.0 or alpha_BHL > 2.0:
        raise ValueError("alpha_BHL should be between 1.0 and 2.0.")

    Wind_fed_scheme = kwargs.get('Wind_fed_scheme', None) 
    if Wind_fed_scheme not in {None, 'HiraiMandel21'}:
        raise ValueError(f"Invalid Wind_fed_scheme '{Wind_fed_scheme}'. Choose None or 'HiraiMandel21'.")
    
    Wind_velo_scheme = kwargs.get('Wind_velo_scheme', 'Hurley+2002')
    if Wind_velo_scheme not in {'Hurley+2002', 'Kudritzki+2000'}:
        raise ValueError(f"Invalid scheme '{Wind_velo_scheme}'. Choose 'Hurley+2002' or 'Kudritzki+2000'.")
    
    Wind_Disk_BH_formation = kwargs.get('Wind_Disk_BH_formation', False)

    B_surf = kwargs.get('B_surf', None)
    if B_surf is not None:
        if B_surf < 0.0:
            raise ValueError("B_surf must be a non-negative value in Gauss.")
        
    if not allowed_keys.issuperset(kwargs.keys()):
        # catch typos / unexpected options early
         raise TypeError(f"Compute_Lx: unknown parameter(s): {list(kwargs.keys())}")

    #===== Efficiency according to CO ===== 
    r_ISCO = None
    if accretor_type == 'NS':
        eta, _ = NS_efficiency(accretor_mass, accretor_log_R, B_surf = B_surf)
    elif accretor_type == 'BH':
        _, r_ISCO, eta = BH_efficiency(accretor_mass, accretor_spin)
    else: # Non compact accretor case
        Data = {'Lx_total': 1e-54, 'Beaming': 1.0, 'Edd_state': np.nan, 'Mdot_BHL_gps' : 0.0, 'Mdot_RLO_gps' : 0.0, 
                'Edd_ratio': 0.0, 'Mdot_Edd_gps': 0.0}
        return Data
    
    if not np.isfinite(eta) or eta <= 0.0:
        eta = 1e-6 # Protect against zero or non-finite efficiency

    #===== Mass rates and other parameters ======

    # Mass-transfer rate (donor -> vicinity of accretor) through RLO
    Mdot_RLO_Msunyr = (0.0 if log_Mtransfer_rate is None or not np.isfinite(log_Mtransfer_rate) else 10.0**float(log_Mtransfer_rate))
    Mdot_RLO_gps = Mdot_RLO_Msunyr * const.Msun / const.secyer  # [g/s]

    # Donor mass loss rate (total, including RLO + winds) [Msun/yr]
    Mdot_donor_total_Msunyr = (0.0 if donor_log_mdot is None or not np.isfinite(donor_log_mdot) else 10.0**float(donor_log_mdot))
    mdot_wind_only_Msunyr = max(Mdot_donor_total_Msunyr - Mdot_RLO_Msunyr, 0.0) # [Msun/yr]
    log_Mdot_wind = np.log10(mdot_wind_only_Msunyr) if mdot_wind_only_Msunyr > 0.0 else -99.0 # No wind when -99

    # Check for hydrogen content at the surface of the donor star : 
    if Donor_H1_surf is None or not np.isfinite(Donor_H1_surf) or Donor_H1_surf < 0.0 or Donor_H1_surf > 1.0:
        Donor_H1_surf = 0.7155 # Default H1 content of a MS star

    # Validate orbital period before using Kepler (guard against NaN/zero/negative)
    if orbital_period_days is None or not np.isfinite(orbital_period_days) or orbital_period_days <= 0.0:
        a_cm = np.nan
        wind_ok = False
    else:
        try:
            a_cm, _ = Orbital_sep_and_velo_from_period(donor_mass, accretor_mass, orbital_period_days)
            wind_ok = True
        except Exception:
            a_cm = np.nan
            wind_ok = False
            Data = {'Lx_total': 1e-54, 'Beaming': 1.0, 'Edd_state': np.nan, 'Mdot_BHL_gps' : 0.0, 'Mdot_RLO_gps' : 0.0, 
                    'Edd_ratio': 0.0, 'Mdot_Edd_gps': 0.0}
            return Data

    if wind_ok:
            try: 
                V_wind = Wind_velocity_BHL(accretor_mass, accretor_type, donor_mass, donor_log_R, donor_logL, log_Mdot_wind,
                                           Donor_H1_surf, Donor_He_core_mass, a_cm, 
                                           Wind_fed_scheme=Wind_fed_scheme, Wind_velo_scheme=Wind_velo_scheme)
            except Exception:
                V_wind = 0.0
                wind_ok = False
    else:
        V_wind = 0.0

    if wind_ok and (V_wind >= 0.0):
        BHL_kwargs = dict(alpha_BHL=alpha_BHL, Wind_Disk_BH_formation=Wind_Disk_BH_formation)

        if Wind_Disk_BH_formation and accretor_type == 'BH' and r_ISCO is not None and eta is not None:
            BHL_kwargs.update(r_ISCO=r_ISCO, eta=eta)
        try:
            Mdot_BHL_gps = BHL_mdot_acc(accretor_mass, donor_mass, accretor_type, mdot_wind_only_Msunyr, 
                                        a_cm, ecc, V_wind, **BHL_kwargs)
            
        except Exception:
            Mdot_BHL_gps = 0.0
    else:
        Mdot_BHL_gps = 0.0
    
    if (accretor_log_mdot is not None and np.isfinite(accretor_log_mdot) and float(accretor_log_mdot) > -50.0):
        Mdot_acc_gps = 10.0**float(accretor_log_mdot) * const.Msun / const.secyer
    else:
        Mdot_acc_gps = Mdot_RLO_gps + Mdot_BHL_gps  # Total accreted mass rate [g/s]
     
    #===== Eddington ratio =====
    Mdot_Edd_gps, L_edd = Eddington_limit(accretor_mass, Donor_H1_surf, eta)
    if not np.isfinite(Mdot_Edd_gps) or Mdot_Edd_gps <= 0.0:
        Mdot_Edd_gps = 1e-99
    
    Edd_ratio = Mdot_acc_gps / Mdot_Edd_gps # Eddington ratio

    #===== X-ray luminosity calculation =====
    if Edd_ratio <= 1.0:
        Lx = eta * Mdot_acc_gps * const.clight**2 if Mdot_acc_gps > 0.0 else 0.0
        beaming = 1.0
        Edd_state = 'Sub-Eddington'
    
    else:
        if Edd_ratio > 150.0:
            beaming = 3.2e-3 # Max beaming factor from [3] King 2008
        elif Edd_ratio > 8.545:
            beaming = 73.0 / Edd_ratio**2
        else: # i.e. 1.0 < Edd_ratio <= 8.545
            beaming = 1.0

        Lx = L_edd / beaming * (1.0 + np.log(Edd_ratio))
        Edd_state = 'Super-Eddington' 

    if not np.isfinite(Lx) or Lx < 0.0:
        Lx = 0.0
    if not np.isfinite(beaming) or beaming <= 0.0:
       beaming = 1.0
    
    Data = {'Lx_total': float(Lx), 
            'beaming': float(beaming), 
            'Edd_state': str(Edd_state), 
            'Mdot_BHL_gps': float(Mdot_BHL_gps), 
            'Mdot_RLO_gps': float(Mdot_RLO_gps),
            'Edd_ratio' : float(Edd_ratio),
            'Mdot_Edd_gps' : float(Mdot_Edd_gps)}
    
    return Data

#* Checked and DocString updated
def XRB_selection_function(history_chunk, oneline_chunk, formation_channels_chunk = None, **kwargs):
    '''
    A XRB selection function to create a population of XRBs, where we store some important information (Only valid for **Population class** object).

    This function identifies X-ray binaries as an accretor and a donor star and returns their properties.
    It will only select systems where one of the stars is a compact object (NS or BH) as long as the companion is not a compact object, WD or massless remnant.
    The function will also compute the X-ray luminosity of each selected binary system using the Compute_Lx function.

    In addition, this function can be passed to **Population_name.create_transient_population()** method to create a population of XRBs from a binary population synthesis simulation.

    Parameters
    ----------
    history_chunk : pd.DataFrame
        A chunk of the history DataFrame from population class binaries. 
        Specific lines need to be provided. 
    oneline_chunk : pd.DataFrame
        A chunk of the oneline DataFrame from population class binaries.
    formation_channels_chunk : pd.DataFrame, optional
        A chunk of the formation channels DataFrame of the binaries. Default is None.
    **kwargs : dict, optional
        Additional arguments passed :
        - alpha_BHL (float, default = 1.5) : efficiency parameter for Bondi-Hoyle-Lyttleton accretion.
        - Wind_velo_scheme (str, default = 'Hurley+2002') : Wind velocity scheme to use in Wind_velocity_BHL function. 
        Can be either 'Hurley+2002' _[[4]] or 'Kudritzki+2000' _[[5]].
        - Wind_fed_scheme (str, default = None) : Define a minimum threshold for filling the Roche lobe in order to obtain accretion winds. Option : 'HiraiMandel21' check ≥ 80% filled systems _[[3].
        - Wind_Disk_BH_formation (bool, default = False) : Whether to check for disk formation criteria around BH accretor _[[2]].

    Returns
    -------
    DF_XRBs : pd.DataFrame
        A DataFrame containing the selected XRBs with accretor/donor data and X-ray luminosities associated to each binary.
    
    References
    ----------
    .. [1] Misra, D., Kovlakas, K., Fragos, T., Lazzarini, M., Bavera, S. S.,  
           Lehmer, B. D., Zezas, A., Zapartas, E., Xing, Z., Andrews, J. J., Dotter, A.,  
           Rocha, K. A., Srivastava, P. M., Sun, M. (2023).  
           *X-ray luminosity function of high-mass X-ray binaries: Studying the signatures of different physical processes using detailed binary evolution calculations*.  
           https://arxiv.org/abs/2209.05505v2
    .. [2] Sen, K. ,Xu, X. -T., Langer, N., El Mellah, I. , Schurmann, C., Quast, M., A&A 652, A138 (2021)
           https://doi.org/10.1051/0004-6361/202141214
           Section 2.3.2, Equation 10
    .. [3] Hirai, R. & Mandel, I. (2021).  
           *Conditions for accretion disc formation and observability of wind-accreting X-ray binaries.*  
           Publications of the Astronomical Society of Australia, 38, e056.  
           doi:10.1017/pasa.2021.53  
           https://doi.org/10.1017/pasa.2021.53
    .. [4] Hurley, J. R., Tout, C. A., & Pols, O. R. 2002, MNRAS, 329, 897
    .. [5] Kudritzki, R.-P., & Puls, J. 2000, ARA&A, 38, 613

    Notes
    -----
    The oneline_chunk DataFrame must contain at least the following columns:
    - 'S1_state_f', 'S2_state_f', 'state_f', 'S1_mass_f', 'S2_mass_f',
    - 'S1_log_R_f', 'S2_log_R_f', 'S1_lg_mdot_f', 'S2_lg_mdot_f', 'S1_log_L_f', 'S2_log_L_f',
    - 'S1_surf_avg_omega_div_omega_crit_f', 'S2_surf_avg_omega_div_omega_crit_f',
    - 'S1_spin_f', 'S2_spin_f', 'S1_surface_h1_f', 'S2_surface_h1_f', 'S1_he_core_mass_f', 'S2_he_core_mass_f',
    - 'time_i', 'time_f', 'metallicity', 'orbital_period_f', 'eccentricity_f', 'lg_mtransfer_rate_f', 
    '''
    
    #===== Optional parameters **kwargs =====
    allowed_keys = {'alpha_BHL', 'Wind_velo_scheme', 'Wind_fed_scheme', 'Wind_Disk_BH_formation'}

    alpha_BHL = kwargs.get('alpha_BHL', 1.5)
    if alpha_BHL < 1.0 or alpha_BHL > 2.0:
        raise ValueError("alpha_BHL should be between 1.0 and 2.0")

    Wind_velo_scheme = kwargs.get('Wind_velo_scheme', 'Hurley+2002') # Wind accretion scheme for BHL
    if Wind_velo_scheme not in {'Hurley+2002', 'Kudritzki+2000'}:
        raise ValueError(f"Invalid scheme '{Wind_velo_scheme}'. Choose 'Hurley+2002' or 'Kudritzki+2000'.")

    Wind_fed_scheme = kwargs.get('Wind_fed_scheme', None)
    if Wind_fed_scheme not in {None, 'HiraiMandel21'}:
        raise ValueError(f"Invalid Wind_fed_scheme '{Wind_fed_scheme}'. Choose None or 'HiraiMandel21'.")

    Wind_Disk_BH_formation = kwargs.get('Wind_Disk_BH_formation', False)

    if not allowed_keys.issuperset(kwargs.keys()):
        # catch typos / unexpected options early
         raise TypeError(f"XRB_selection_function: unknown parameter(s): {sorted(set(kwargs.keys()) - allowed_keys)}")
    
    #===== Check inputs =====
    required = [
        'S1_state_f','S2_state_f','S1_mass_f','S2_mass_f',
        'S1_log_R_f','S2_log_R_f','S1_lg_mdot_f','S2_lg_mdot_f','S1_log_L_f','S2_log_L_f',
        'S1_surf_avg_omega_div_omega_crit_f','S2_surf_avg_omega_div_omega_crit_f',
        'S1_spin_f','S2_spin_f','S1_surface_h1_f','S2_surface_h1_f','S1_he_core_mass_f','S2_he_core_mass_f',
        'time_f','time_i','metallicity','orbital_period_f','eccentricity_f','lg_mtransfer_rate_f','state_f'
    ]

    missing = [col for col in required if col not in oneline_chunk.columns]
    if missing:
        raise KeyError(f"oneline_chunk missing required columns: {missing}")
    
    #===== Initialization =====
    # Compact objects types for filtering
    Compact_types = {'NS','BH'}
    Objects_to_avoid = {'NS','BH','massless_remnant', 'WD'} # To avoid double CO, WD and massless remnants companions
    bad_systems_states = {'disrupted', 'ERR'} # Only detached, initial_RLOF systems

    # Mask to identify XRBs in detached state
    Star1, Star2, state = oneline_chunk['S1_state_f'], oneline_chunk['S2_state_f'], oneline_chunk['state_f']
    mask = ((Star1.isin(Compact_types)) & (~Star2.isin(Objects_to_avoid)) & (~state.isin(bad_systems_states))) | ((Star2.isin(Compact_types)) & (~Star1.isin(Objects_to_avoid)) & (~state.isin(bad_systems_states))) # Mask to filter out non XRB systems

    oneline_sel = oneline_chunk.loc[mask].copy()
    is_s1_compact = oneline_sel['S1_state_f'].isin(Compact_types)

    # Create an empty Dataframe with indices of the selected binaries
    indices = oneline_sel.index.to_numpy()
    DF_XRBs = pd.DataFrame(index = indices)

    DF_XRBs['time'] = oneline_sel['time_f'] * 1e-6 # Myr
    DF_XRBs['time_of_birth'] = oneline_sel['time_i'] * 1e-6 # Myr
    DF_XRBs['metallicity'] = oneline_sel['metallicity']
    DF_XRBs['final_state'] = oneline_sel['state_f'] # final state of the system
    
    #===== Identify accretor/donor and their properties =====
    # Accretor parameters
    DF_XRBs['Accretor_state'] = np.where(is_s1_compact, oneline_sel['S1_state_f'], oneline_sel['S2_state_f'])

    DF_XRBs['Accretor_mass'] = np.where(is_s1_compact, oneline_sel['S1_mass_f'], oneline_sel['S2_mass_f'])

    DF_XRBs['Accretor_logR'] = np.where(is_s1_compact, oneline_sel['S1_log_R_f'], oneline_sel['S2_log_R_f'])

    DF_XRBs['Accretor_logMdot'] = np.where(is_s1_compact, oneline_sel['S1_lg_mdot_f'], oneline_sel['S2_lg_mdot_f'])

    DF_XRBs['Accretor_spin'] = np.where(is_s1_compact, oneline_sel['S1_spin_f'], oneline_sel['S2_spin_f'])

    # Donor parameters
    DF_XRBs['Donor_state'] = np.where(is_s1_compact, oneline_sel['S2_state_f'], oneline_sel['S1_state_f'])

    DF_XRBs['Donor_mass'] = np.where(is_s1_compact, oneline_sel['S2_mass_f'], oneline_sel['S1_mass_f'])

    DF_XRBs['Donor_logR'] = np.where(is_s1_compact, oneline_sel['S2_log_R_f'], oneline_sel['S1_log_R_f'])

    DF_XRBs['Donor_logL'] = np.where(is_s1_compact, oneline_sel['S2_log_L_f'], oneline_sel['S1_log_L_f'])

    DF_XRBs['Donor_logMdot'] = np.where(is_s1_compact, oneline_sel['S2_lg_mdot_f'], oneline_sel['S1_lg_mdot_f'])

    DF_XRBs['Donor_surf_avg_omega_div_omega_crit'] = np.where(is_s1_compact, 
                                                              oneline_sel['S2_surf_avg_omega_div_omega_crit_f'], 
                                                              oneline_sel['S1_surf_avg_omega_div_omega_crit_f'])

    DF_XRBs['Donor_H1_surface_content'] = np.where(is_s1_compact, oneline_sel['S2_surface_h1_f'], oneline_sel['S1_surface_h1_f'])

    DF_XRBs['Donor_He_core_mass'] = np.where(is_s1_compact, oneline_sel['S2_he_core_mass_f'], oneline_sel['S1_he_core_mass_f'])
    
    #===== Additional parameters related to the system =====
    DF_XRBs['Orbital_period'] = oneline_sel['orbital_period_f'].to_numpy()  # days
    DF_XRBs['Eccentricity'] = oneline_sel['eccentricity_f'].to_numpy()
    DF_XRBs['Log_Mtransfer_rate'] = oneline_sel['lg_mtransfer_rate_f'].to_numpy()

    #===== Calculate XLF for the population =====
    # Luminosity calculation are splitted in two parts: if the companion is a Be star or not.

    Surf_velo = DF_XRBs['Donor_surf_avg_omega_div_omega_crit'].to_numpy(copy=False)
    orbital_period_days = DF_XRBs['Orbital_period'].to_numpy(copy=False)
    binsys_state = DF_XRBs['final_state'].to_numpy(copy=False)
    donor_mass = DF_XRBs['Donor_mass'].to_numpy(copy=False)
    accretor_mass = DF_XRBs['Accretor_mass'].to_numpy(copy=False)
    donor_logR = DF_XRBs['Donor_logR'].to_numpy(copy=False)


    cols = DF_XRBs[['Donor_mass', 'Donor_logR', 'Donor_logL', 'Accretor_mass','Accretor_logR', 'Accretor_state', 'Log_Mtransfer_rate', 
                    'Donor_logMdot', 'Accretor_logMdot', 'Orbital_period', 'Eccentricity', 'Accretor_spin', 
                    'Donor_H1_surface_content', 'Donor_He_core_mass', 'final_state']].itertuples(index=False, name=None)
    
    n = DF_XRBs.shape[0] # Number of rows in DF_XRBs
    Lx_Total = np.full(n, np.nan)
    Beaming = np.full(n, np.nan)
    Edd_State = np.full(n, None, dtype=object)

    
    Mdot_BHL_array = np.full(n, np.nan)
    Mdot_RLO_array = np.full(n, np.nan)
    Eddington_ratio = np.full(n, np.nan)
    mdot_edd_gps = np.full(n, np.nan)

    
    #----- Luminosity calculation Be XRBs -----
    mask_be = ((Surf_velo > 0.7) 
               & (orbital_period_days >= 10.0) & (orbital_period_days <= 300.0) 
               & (binsys_state == 'detached') 
               & (donor_mass >= 6.0)
               & (DF_XRBs['Donor_state'].to_numpy(copy=False) == 'H-rich_Core_H_burning')
               )
    
    if np.any(mask_be) :
        period_be = orbital_period_days[mask_be]
        donor_mass_be = donor_mass[mask_be]
        accretor_mass_be = accretor_mass[mask_be]
        donor_radius_be = (10**donor_logR * const.Rsun)[mask_be]
        Lx_Be = np.array([Be_XRB_Lx(P, Md, Dr, Ma) for P, Md, Dr, Ma in zip(period_be, donor_mass_be, donor_radius_be, accretor_mass_be) ])
        Lx_Total[mask_be] = Lx_Be
        Beaming[mask_be] = 0.1 # Assumed as Duty cycle for Be-XRBs
        Edd_State[mask_be] = 'Be-XRB system'
        Mdot_BHL_array[mask_be] = 0.0
        Mdot_RLO_array[mask_be] = 0.0
        # Add a label to Donor_state to indicate Be star
        DF_XRBs.loc[mask_be, 'Donor_state'] = (DF_XRBs.loc[mask_be, 'Donor_state'].fillna('').astype(str) + ' (Be star)')

    #----- Luminosity calculation classical XRBs -----

    # keep track of errors
    error_count = 0
    error_indices = []

    # Indices of the DataFrame (for error tracking)
    idxs = DF_XRBs.index.to_numpy()

    # Luminosity calculation loop
    for i, row in enumerate(cols):
        if mask_be[i]:
            continue

        Dm, DlR, DlL, Am, AlR, Atype, log_mt, Dlog, Alog, Op, ecc, spin, h1, He, State_sys = row
        
        res = None

        try:
            if (Atype is None) or (not np.isfinite(Dm)) or (not np.isfinite(Am)):
                continue
            res = Compute_Lx(
                donor_mass = Dm,
                donor_log_R = DlR,
                donor_logL = DlL,
                accretor_mass = Am,
                accretor_log_R = AlR,
                accretor_type = Atype,
                donor_log_mdot = Dlog,
                log_Mtransfer_rate = log_mt,
                accretor_log_mdot = Alog,
                Donor_H1_surf = h1,
                Donor_He_core_mass = He,
                orbital_period_days = Op,
                ecc = ecc,
                accretor_spin = spin,
                alpha_BHL = alpha_BHL,
                Wind_velo_scheme = Wind_velo_scheme,
                Wind_fed_scheme = Wind_fed_scheme,
                Wind_Disk_BH_formation = Wind_Disk_BH_formation,
            )
        except Exception as e:
            error_count += 1
            try: # store the original DataFrame index for later inspection
                error_indices.append(idxs[i])
            except Exception:
                pass
            continue

        # Normalize result to tuple of three floats
        if res is None:
            continue
        
        try:
            if isinstance(res, dict):
                lx = res.get('Lx_total', 0.0)
                be = res.get('beaming', 1.0)
                edd = res.get('Edd_state', 'Sub-Eddington')
                mdot_bhl = res.get('Mdot_BHL_gps', np.nan)
                m_rlo = res.get('Mdot_RLO_gps', np.nan)
                edd_ratio = res.get('Edd_ratio', 0.0)
                mdot_edd_val = res.get('Mdot_Edd_gps', np.nan)
            elif hasattr(res, '__len__') and len(res) >= 3:
                lx = res[0]
                be = res[1]
                edd = res[2]
                mdot_bhl = res[3] if len(res) > 3 else np.nan
                m_rlo = res[4] if len(res) > 4 else np.nan
                edd_ratio = res[5] if len(res) > 5 else 0.0
                mdot_edd_val = res[6] if len(res) > 6 else np.nan
            else:
                lx = res
                be = 1.0
                edd = 'N/A'
                mdot_bhl = np.nan
                m_rlo = np.nan
                edd_ratio = 0.0
                mdot_edd_val = np.nan

            Lx_Total[i] = float(lx) if np.isfinite(lx) else np.nan
            Beaming[i] = float(be) if np.isfinite(be) else 1.0
            Edd_State[i] = str(edd)
            Mdot_BHL_array[i] = float(mdot_bhl) if np.isfinite(mdot_bhl) else np.nan
            Mdot_RLO_array[i] = float(m_rlo) if np.isfinite(m_rlo) else np.nan
            Eddington_ratio[i] = float(edd_ratio) if np.isfinite(edd_ratio) else np.nan
            mdot_edd_gps[i] = float(mdot_edd_val) if np.isfinite(mdot_edd_val) else np.nan


        except Exception:
            Lx_Total [i] = Beaming[i] = np.nan   
            Edd_State[i] = None

    if error_count:
        # Succinct error display
        print(f"Compute_Lx: {error_count} error(s) in the luminosity calculation (Indices : {error_indices[:10]})")
    
    
    DF_XRBs['Mdot_BHL_gps'] = Mdot_BHL_array
    DF_XRBs['Mdot_RLO_gps'] = Mdot_RLO_array
    DF_XRBs['Beaming'] = Beaming
    DF_XRBs['Edd_State'] = Edd_State

    DF_XRBs['Eddington_ratio'] = Eddington_ratio
    DF_XRBs['Mdot_Edd_gps'] = mdot_edd_gps
    DF_XRBs['Lx_Total'] = Lx_Total

    return DF_XRBs

#* Checked and DocString updated 
def Compute_IMF_normalization_correction(Population, m_min = 0.01, m_max = 200.0):
    '''
    Compute the IMF-based normalization correction factor used to relate
    a simulated population of massive-star binaries to the total stellar
    mass formed in a star-forming event.

    This correction follows the method described in Appendix A of Bavera et al. (2020, A&A 635, A97, doi:10.1051/0004-6361/201936204).

    Parameters
    ----------
    Population : Population
        A Population object containing the initial parameters of the population synthesis simulation.
    m_min : float, optional
        Minimum stellar mass considered for the normalization [M⊙]. Default is 0.01.
    m_max : float, optional
        Maximum stellar mass considered for the normalization [M⊙]. Default is 200.0. 
        The upper mass limit is restricted to 300 M⊙, corresponding to the maximum ZAMS mass covered by the MESA grids used in POSYDON.

    Returns
    -------
    normalization : float
        The IMF-based normalization correction factor (dimensionless). 
        normalization = 1 / (fraction of total mass formed in stars that is represented by the modeled population).

    References
    ----------
    .. [1]  Bavera, S. S., Fragos, T., Zevin, M., et al. (2020).
            *The origin of spin in binary black holes.*
            Astronomy & Astrophysics, 635, A97.
            DOI: 10.1051/0004-6361/201936204.

    Notes
    -----
    This function assumes that the Population object has an attribute
    `ini_params` which is a dictionary containing the initial parameters
    of the population synthesis simulation, including the primary mass
    scheme and binary fraction scheme. 
    In the appendix A calculation, the factor f_0 is set such as the integral of the IMF (f(m)) over [m_min, m_max] is equal to 1.
    In the final formula, this factor f_0 cancels out, so it is not explicitly calculated here.
    The mass ratio _q_ distribution is assumed to be flat in [0, 1]. 

    '''

    Ini_params = Population.ini_params.copy()

    # Extract relevant parameters
    primary_mass_scheme = Ini_params.get('primary_mass_scheme', 'Kroupa2001')
    m_A = Ini_params.get('primary_mass_min', 5.0)
    m_B = Ini_params.get('primary_mass_max', 150.0)
    sec_mass_scheme = Ini_params.get('secondary_mass_scheme', 'flat_mass_ratio')
    binary_fraction_const = Ini_params.get('binary_fraction_const', 0.7)

    if sec_mass_scheme != 'flat_mass_ratio':
        raise NotImplementedError(f"The secondary_mass_scheme has to be 'flat_mass_ratio' for Compute_IMF_normalization_correction. Got '{sec_mass_scheme}' instead.")
    
    if primary_mass_scheme != 'Kroupa2001':
        raise NotImplementedError(f"Compute_IMF_normalization_correction: primary_mass_scheme '{primary_mass_scheme}' not implemented yet.")
    
    if not (0.01 <= m_min < m_A < m_B < m_max <= 300):
        raise ValueError(f"Compute_IMF_normalization_correction: m_min and m_max must be within [0.01, 300] and cover at least the range of simulated primary masses.\n Primary_mass_min : {m_A}\n Primary_mass_max : {m_B}")
    
    # Kroupa (2001) IMF parameters
    alpha1 = 0.3
    alpha2 = 1.3
    alpha3 = 2.3
    m_break1 = 0.08 # [M⊙] - Transition slope from alpha1 to alpha2
    m_break2 = 0.5  # [M⊙] - Transition slope from alpha2 to alpha3

    # Mean mass of stellar systems in the population : 
    mstar_mean = (1 + binary_fraction_const/2) * (1 / (2 - alpha1) * (m_break1**(2 - alpha1) - m_min**(2 - alpha1)) / m_min**(-alpha1) 
                                                  + 1 / (2 - alpha2) * (m_break1 / m_min)**(-alpha1) * (m_break2**(2 - alpha2) - m_break1**(2 - alpha2)) / m_break1**(-alpha2)
                                                  + 1 / (2 - alpha3) * (m_break1 / m_min)**(-alpha1) * (m_break2 / m_break1)**(-alpha2) * 
                                                    (m_max**(2 - alpha3) - m_break2**(2 - alpha3)) / m_break2**(-alpha3))
    
    # Fraction of systems that are modeled if m_A, m_B > m_break2
    f_model = (binary_fraction_const * 1 / (1 - alpha3) * (m_break1 / m_min)**(-alpha1) * (m_break2/m_break1)**(-alpha2) 
              * (m_B**(1 - alpha3) - m_A**(1 - alpha3)) / (m_break2**(-alpha3)))
    
    # Mean mass of a binary system in the modelled population :
    mstar_model_mean = 3/2 * (1- alpha3) / (2 -alpha3) * (m_B**(2 - alpha3) - m_A**(2 - alpha3)) / (m_B**(1 - alpha3) - m_A**(1 - alpha3))

    # The total modeled mass represnets only a fraction of the total mass formed in stars.
    correction_factor = f_model * mstar_model_mean / mstar_mean

    # Normilization with the inverse of the correction factor
    normalization = 1 / correction_factor

    return float(normalization)

#* Checked and DocString updated
def Stat_XLF(transient_population, Lmin=1e30, SFR = 1.0, stat_kwargs=None):
    '''
    Compute statistics for a given transient population of XRBs, including the weigthed cumulative number of XRBs above a given luminosity.

    Parameters
    ----------
    transient_population : TransientPopulation object
        A transient population of XRBs.
    Lmin : float, optional
        Minimum luminosity threshold for XRBs [erg/s]. Default is 1e30.
    SFR : float, optional
        Star formation rate [M⊙/yr]. Default is 1.0. (E.g. Milky-Way SFR ~ 1.5 M⊙/yr. Starburst galaxies SFR ~ 10 M⊙/yr)
    stat_kwargs : dict, optional
        A dictionary of population parameters to override the default ones used in the transient population in order to weight the population accordingly.
        See Notes for allowed keys. Default is *None*.

    Returns
    -------
    L_sorted : np.ndarray
        Sorted array of X-ray luminosities [erg/s].
    N_weighted : np.ndarray
        Weighted cumulative number of XRBs above each luminosity [number].
    weights : pd.Series
        Weights associated with each XRB in the population.

    References
    ----------
    .. [1] Misra, D., Kovlakas, K., Fragos, T., Lazzarini, M., Bavera, S. S.,  
           Lehmer, B. D., Zezas, A., Zapartas, E., Xing, Z., Andrews, J. J., Dotter, A.,  
           Rocha, K. A., Srivastava, P. M., Sun, M. (2023).  
           *X-ray luminosity function of high-mass X-ray binaries: Studying the signatures of different physical processes using detailed binary evolution calculations*.  
           https://arxiv.org/abs/2209.05505v2
    .. [2] Bavera, S. S., Fragos, T., Zevin, M., et al. (2020). 
           *The origin of spin in binary black holes.* 
           Astronomy & Astrophysics, 635, A97. 
           DOI: 10.1051/0004-6361/201936204.

    Notes
    -----
    The function uses the calculate_model_weights method of the TransientPopulation class to compute weights based on the specified population parameters.
    For that, you can provide a dictionary of population parameters to override the default ones used in the transient population in order to weight the population accordingly.
    Therefore, the allowed keys in the pop_params dictionary are:
    - 'number_of_binaries'
    - 'binary_fraction_scheme', 'binary_fraction_const'
    - 'star_formation'
    - 'max_simulation_time'
    - 'primary_mass_scheme', 'primary_mass_min', 'primary_mass_max'
    - 'secondary_mass_scheme', 'secondary_mass_min', 'secondary_mass_max'
    - 'orbital_scheme', 'orbital_period_scheme', 'orbital_period_min', 'orbital_period_max'
    - 'eccentricity_scheme'
    - 'q_min', 'q_max'
    
    Example
    -------
    >>> stat_kwargs = {'primary_mass_min': 0.01, 'primary_mass_max': 200.0, 'q_min': 0.0, 'q_max': 1.0,}
    >>> L_sorted, N_weighted, weights = Stat_XLF(transient_population, Lmin=1e30, Tot_underlying_mass=1e8, stat_kwargs=stat_kwargs)

    '''

    #===== Statistics Kwargs =====

    # Keys allowed in calculate_model_weights()
    _allowed_pop_keys = {
        'number_of_binaries',
        'binary_fraction_scheme',
        'binary_fraction_const',
        'star_formation',
        'max_simulation_time',
        'primary_mass_scheme',
        'primary_mass_min',
        'primary_mass_max',
        'secondary_mass_scheme',
        'secondary_mass_min',
        'secondary_mass_max',
        'orbital_scheme',
        'orbital_period_scheme',
        'orbital_period_min',
        'orbital_period_max',
        'eccentricity_scheme',
        'q_min', 'q_max'
    }
    Pop_params = transient_population.ini_params.copy()

    if stat_kwargs is not None:
        if not isinstance(stat_kwargs, dict):
            raise TypeError("stat_kwargs must be a dict or None.")
        if not _allowed_pop_keys.issuperset(stat_kwargs.keys()):
            raise ValueError(f"Stat_XLF: Invalid population parameter(s): {sorted(set(stat_kwargs.keys()) - _allowed_pop_keys)}. Allowed keys: {_allowed_pop_keys}")
        # overwrite only allowed keys
        for key, value in stat_kwargs.items():
            Pop_params[key] = value


    # Correction factor from 
    #IMF_corr = Compute_IMF_normalization_correction(transient_population)

    T_window_yr = Pop_params['max_simulation_time'] # Years
    Tot_underlying_mass = SFR * T_window_yr  # Total underlying mass in M⊙

    base_w = transient_population.calculate_model_weights(model_weights_identifier="base_IMF",
                                                          population_parameters=Pop_params)
    
    if isinstance(base_w, pd.DataFrame):
        # prefer the named column if present, else take first column
        if "base_IMF" in base_w.columns:
            weights_series = base_w["base_IMF"].to_numpy(float)
        else:
            weights_series = base_w.iloc[:, 0].to_numpy(float)
    else:
        weights_series = np.asarray(base_w, float)  # assume Series-like
    
    Weights = weights_series * Tot_underlying_mass

    return Weights

#* Checked and DocString updated
def plot_ccdf(transient_population, ax, weights, mask, label, SFR = 1.0, **ccdf_kwargs):
    '''
    Compute the cumulative X-ray luminosity function (CCDF) of a given transient populations. 
    This function returns a *time-averaged* CCDF, i.e. the expected number of X-ray binaries brighter than L at a random time, 
    assuming constant star formation over a time window T_window_Myr.\n
    Masks split the population into different sub-populations to be plotted separately (e.g. RLO, Wind-fed, Be-XRBs, etc.).

    Parameters
    ----------
    transient_population : TransientPopulation object
        The transient population.
    SFR : float, optional
        Star formation rate [M⊙/yr]. Default is 1.0. (E.g. Milky-Way SFR ~ 1.5 M⊙/yr. Starburst galaxies SFR ~ 10 M⊙/yr)
    ax : matplotlib.axes.Axes
        The axes on which to plot the CCDF (_fig , ax = plt.subplots(*args, **kwargs)_).
    weights : array-like
        The weights associated with each value.
    mask : array-like
        A boolean mask to filter the values.
    label : str
        The label for the CCDF plot.
    linestyle : str
        The line style for the CCDF plot.
    **kwargs : dict, optional
        Additional keyword arguments for customization.
        - norm_factor (float, default=0.5) : Normalization factor L_x/L_bolo ~ 0.5 from Misra et al.[1] page 4
        - extend_to_edges (bool, default=True) : Whether to extend the CCDF to the edges of the plot.
    
        Any other keyword arguments valid for matplotlib.pyplot.plot() can be passed.

    Returns
    -------
    None

    References
    ----------
    .. [1] Misra, D., Kovlakas, K., Fragos, T., Lazzarini, M., Bavera, S. S.,  
           Lehmer, B. D., Zezas, A., Zapartas, E., Xing, Z., Andrews, J. J., Dotter, A.,  
           Rocha, K. A., Srivastava, P. M., Sun, M. (2023).  
           *X-ray luminosity function of high-mass X-ray binaries: Studying the signatures of different physical processes using detailed binary evolution calculations*.  
           https://arxiv.org/abs/2209.05505v2
    Notes
    -----
    * Weights are assumed to already include IMF, binary fraction, and population 
    synthesis corrections

    Example
    -------
    >>> fig, ax = plt.subplots()
    >>> plot_ccdf(transient_population, ax, weights, mask, label='Label name', color='blue', linestyle='-')

    '''
    #===== kwargs =====
    controlled_keys = {'norm_factor', 'extend_to_edges'}

    plot_keys = dict(label=label)
    extra_plot_keys = {key: var for key, var in ccdf_kwargs.items() if key not in controlled_keys}
    plot_keys.update(extra_plot_keys)
 
    extend_to_edges = ccdf_kwargs.get('extend_to_edges', True) # Whether to extend the CCDF to the edges of the plot
    norm_factor = ccdf_kwargs.get('norm_factor', 0.5) # Normalization factor L_x/L_bolo ~ 0.5 from Misra et al.[1] page 4

    if norm_factor <= 0.0:
        raise ValueError("plot_ccdf: norm_factor must be > 0.0")

    L = transient_population.population['Lx_Total'].to_numpy(dtype = float) * norm_factor

    T_window_yr = transient_population.ini_params.get('max_simulation_time', 1e8) # Years
    Total_underlying_mass = SFR * T_window_yr  # Total underlying mass in M⊙

    if Total_underlying_mass <= 0.0:
        raise ValueError("plot_ccdf: Total_underlying_mass must be > 0.0")

    b = transient_population.population['Beaming'].to_numpy(dtype = float)
    b = np.where(np.isfinite(b) & (b > 0.0), b, 1.0)

    selection = mask & np.isfinite(L) & (L >= 1e30)

    if not np.any(selection):
        return

    L_selec = L[selection]
    Weight_select = weights[selection] * b[selection]

    order = np.argsort(L_selec)
    L_sorted = L_selec[order]
    Weights_sorted = Weight_select[order]

    N = np.cumsum(Weights_sorted[::-1])[::-1]  # Cumulative sum from the end to the beginning


    if extend_to_edges:
        cur_xlim = ax.get_xlim()
        if cur_xlim == (0.0, 1.0):
            xmin, xmax = float(L_sorted[0]), float(L_sorted[-1])
        else:
            xmin, xmax = cur_xlim
        cur_ylim = ax.get_ylim()
        if cur_ylim == (0.0, 1.0):
            ymin = np.nextafter(0, 1)
        else:
            ymin = cur_ylim[0]

    # Valeur totale au-dessus du plus petit L
        y0 = float(np.sum(Weights_sorted))

        x_ext = np.r_[xmin, L_sorted, xmax]
        y_ext = np.r_[y0, N, ymin]

        ax.plot(x_ext, y_ext, **plot_keys)
    else:
        ax.plot(L_sorted, N, **plot_keys)



#==============================================================================
# Functions not needed for the moment. Kept here for later use if needed.

#* Checked and DocString updated
#? Add a control parameter to avoid to recalculate model wieghts if already done before. (calculate_model_weights() in Stat_XLF()) Might be good ? 
def plot_XLF_models(model_list, L_min = 1e30, L_max = 1e43, Total_underlying_mass=1e8,
                    ncols=2, figsize=(12,6), dpi = 300, **kwargs):
    '''
    Plot the X-ray luminosity function (XLF) for multiple XRB population models in the form of grid. 
    This function uses a cumulative distribution function (CCDF) representation.
    Unified legend and fixed colors/styles across subplots.\n
    
    This function uses two other functions that extract needed data (weight of the population) and plot the CCDF. In stat_kwargs, one can pass keywords
    for Stat_XLF() which calculate the weight of each XRB in the population. Parameters in the **.ini** file can be given in stat_kwargs to extend the weighting scheme. (See notes below for details).\n

    Plots will use consistent colors/styles (Misra et al. 2023 [[1]_]) :
    - Total XRB population : solid blue
    - Be-XRBs : dotted grey
    - RLO / Wind (BH / NS) by donor composition:
        * BH, He-rich (donor contains "stripped_He"): RLO solid purple, Wind dashed purple
        * BH, H-rich (donor contains "H-rich" or "accreted_He"): RLO solid dark red, Wind dashed dark red
        * NS, He-rich (donor contains "stripped_He"): RLO solid yellow, Wind dashed yellow
        * NS, H-rich (donor contains "H-rich" or "accreted_He"): RLO solid red, Wind dashed red

    (Donor composition is inferred from 'Donor_state' containing the keywords above.)

    Parameters
    ----------
    model_list : list of tuples
        List of tuples containing model labels and TransientPopulation objects.
    L_min : float, optional
        Minimum luminosity threshold for XRBs [erg/s]. Default is 1e30.
    L_max : float, optional
        Maximum luminosity threshold for XRBs [erg/s]. Default is 1e43.
    Total_underlying_mass : float, optional
        Total underlying mass for weighting [M⊙]. Default is 1e8 M⊙.
    ncols : int, optional
        Number of columns in the subplot grid. Default is 2.
    figsize : tuple, optional
        Figure size. Default is (12, 6).
    dpi : int, optional
        Dots per inch for the figure. Default is 300.
    ylims : dict or list of tuples, optional
        Y-axis limits for each subplot. If a dict, keys are ~ model labels. If a list, length must match number of models.
    yscale : str, optional
        Y-axis scale ('log' or 'linear'). Default is 'log'.
    **kwargs : dict, optional
        Additional keyword arguments for Stat_XLF and plot_ccdf functions.
        - stat_kwargs (dict, default=None) : Keyword arguments for Stat_XLF().
        - ccdf_kwargs (dict, default=None) : Keyword arguments for plot_ccdf().
        - ylims (dict or list of tuples, default=None) : Y-axis limits for each subplot.
        - yscale (str, default='log') : Y-axis scale ('log' or 'linear').

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.

    References
    ----------
    .. [1] Misra, D., Kovlakas, K., Fragos, T., Lazzarini, M., Bavera, S. S.,  
           Lehmer, B. D., Zezas, A., Zapartas, E., Xing, Z., Andrews, J. J., Dotter, A.,  
           Rocha, K. A., Srivastava, P. M., Sun, M. (2023).  
           *X-ray luminosity function of high-mass X-ray binaries: Studying the signatures of different physical processes using detailed binary evolution calculations*.  
           https://arxiv.org/abs/2209.05505v2

    Notes
    -----
    - Each subplot corresponds to a different XRB population model.
    - Colors and line styles are consistent across models for different XRB components.
    - A unified legend is created for all subplots.
    - The function uses Stat_XLF() to compute XLF statistics and plot_ccdf() to plot the cumulative distribution function. 
    Therefore, kwargs for these functions can be passed via stat_kwargs and ccdf_kwargs.\n
    

    Example
    -------
    Keywords for **Stat_XLF()** includes _Lmin, Tot_underlying_mass and pop_params (dict of population parameters)_ and may be
    passed as follow :\n 
    >>> dict_params = {'primary_mass_min': 0.01, 'primary_mass_max': 200.0, 'q_min': 0.0, 'q_max': 1.0,} 
    >>> stat_kwargs={'L_min': 1e30, 'Tot_underlying_mass':1e8, pop_params = dict_params}\n
    >>> plot_XLF_models(model_list, stat_kwargs=stat_kwargs)\n

    Keywords for **plot_ccdf()** include **color, linestyle, T_window_Myr, Total_underlying_mass, norm_factor** and may be 
    passed as follow :\n 
    >>> ccdf_kwargs={'T_window_Myr':100.0, 'Total_underlying_mass':1e8, 'norm_factor':0.5}
    >>> plot_XLF_models(model_list, ccdf_kwargs=ccdf_kwargs)
    '''

    import matplotlib.lines as mlines # type: ignore

    #===== Extract kwargs =====
    allowed_top = {'stat_kwargs', 'ccdf_kwargs', 'ylims', 'yscale'}

    if not allowed_top.issuperset(kwargs.keys()):
        raise ValueError(f"plot_XLF_models: Invalid keyword argument(s): {sorted(set(kwargs.keys()) - allowed_top)}. Allowed keys: {allowed_top}")

    stat_kwargs = kwargs.get('stat_kwargs', None) # kwargs for Stat_XLF()
    ccdf_kwargs = kwargs.get('ccdf_kwargs', None) # kwargs for plot_ccdf() and for matplotlib.pyplot.plot()
    ylims = kwargs.get('ylims', None)
    yscale = kwargs.get('yscale', 'log')


    if stat_kwargs is None:
        stat_kwargs = dict(Lmin=L_min, Tot_underlying_mass=Total_underlying_mass, 
                           stat_kwargs=None)
    if ccdf_kwargs is None:
        ccdf_kwargs = {}
    
    elif not isinstance(ccdf_kwargs, dict):
        raise TypeError("plot_XLF_models: ccdf_kwargs must be a dict or None.")


    #===== Parameters and plotting =====
    # color / linestyle mapping (keep consistent across models) based on Misra et al. (2023)
    style_map = {
        'Total XRB population': {'color':'#1F77B4', 'linestyle':'-'},
        'Be-XRBs': {'color':'#808080', 'linestyle':':'},
        'RLO BH-He-rich': {'color':'#B760C4', 'linestyle':'-'},
        'RLO BH-H-rich': {'color':'#540B0E', 'linestyle':'-'},
        'RLO NS-He-rich': {'color':'#F9E979', 'linestyle':'-'},
        'RLO NS-H-rich': {'color':'#DA1E37', 'linestyle':'-'},
        'Wind BH-He-rich': {'color':'#B760C4', 'linestyle':'--'},
        'Wind BH-H-rich': {'color':'#540B0E', 'linestyle':'--'},
        'Wind NS-He-rich': {'color':'#F9E979', 'linestyle':'--'},
        'Wind NS-H-rich': {'color':'#DA1E37', 'linestyle':'--'},
    }

    components = list(style_map.keys())  # order for legend

    nmodels = len(model_list)
    nrows = int(np.ceil(nmodels / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi, squeeze=False)
    axes_flat = axes.ravel()

    ylims_list = None
    if ylims is None:
        ylims = {}
    elif isinstance(ylims, (list, tuple)):
        if len(ylims) == nmodels:
            ylims_list = list(ylims)
        else:
            raise ValueError("If ylims is a list/tuple, its length must match the number of models.")
            
    for i, (ax, (label, tp)) in enumerate(zip(axes_flat, model_list)):
        # compute Stat_XLF once (weights returned for later use)
        _, _, weights = Stat_XLF(tp, stat_kwargs=stat_kwargs)

        # precompute masks used for the different components on this population
        pop = tp.population
        Lumino_range = (pop['Lx_Total'] < L_max) & (pop['Lx_Total'] > L_min)
        Be_XRB = pop['Donor_state'].str.contains('Be star', na=False)
        BH = pop['Accretor_state'] == 'BH'
        NS = pop['Accretor_state'] == 'NS'
        H1_rich = (pop['Donor_state'].str.contains('H-rich|accreted_He', regex=True, na=False)
                   & ~pop['Donor_state'].str.contains('Be star', na=False))
        stripped_He = pop['Donor_state'].str.contains('stripped_He', na=False)

        is_RLO = (pop['Mdot_RLO_gps'] > pop['Mdot_BHL_gps']) & np.isfinite(pop['Mdot_RLO_gps']) & np.isfinite(pop['Mdot_BHL_gps'])
        is_Wind = (pop['Mdot_BHL_gps'] >= pop['Mdot_RLO_gps']) & np.isfinite(pop['Mdot_RLO_gps']) & np.isfinite(pop['Mdot_BHL_gps'])

        mask_map = {
            'Total XRB population': Lumino_range,
            'Be-XRBs': Be_XRB,
            'RLO BH-He-rich': is_RLO & BH & stripped_He,
            'RLO BH-H-rich': is_RLO & BH & H1_rich,
            'RLO NS-He-rich': is_RLO & NS & stripped_He,
            'RLO NS-H-rich': is_RLO & NS & H1_rich,
            'Wind BH-He-rich': is_Wind & BH & stripped_He,
            'Wind BH-H-rich': is_Wind & BH & H1_rich,
            'Wind NS-He-rich': is_Wind & NS & stripped_He,
            'Wind NS-H-rich': is_Wind & NS & H1_rich,
        }

        # plot components without setting labels (we'll create a unified legend)
        for comp in components:
            mask = mask_map.get(comp, np.zeros(len(pop), dtype=bool))
            # skip if nothing to plot
            if not np.any(mask):
                continue
            s = style_map[comp]
            # call existing plot function (it accepts color & linestyle)
            plot_ccdf(tp, ax, weights, mask, label=None, linestyle=s['linestyle'], color=s['color'], **ccdf_kwargs)
            
        ax.set_xscale('log')
        ax.set_yscale('log' if yscale == 'log' else 'linear')
        ax.set_xlabel(r'$L_{X,(0.5–8 keV)}$ /erg s$^{-1}$')
        ax.set_ylabel(r'N($> L_{X}$) / (M$_\odot$ yr$^{-1}$)')
        ax.set_xlim(1e35, 1e43)
        
        ylim = (ylims_list[i] if ylims_list is not None else ylims.get(label, None))
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(label, pad=8)
        
    # hide any unused axes
    for ax in axes_flat[nmodels:]:
        ax.set_visible(False)

    # build unified legend using proxy artists (one entry per component)
    proxies = []
    labels = []
    for comp in components:
        s = style_map[comp]
        line = mlines.Line2D([], [], color=s['color'], linestyle=s['linestyle'], linewidth=1.5)
        proxies.append(line)
        labels.append(comp)

    fig.legend(proxies, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=min(5, len(labels)), frameon=True, fontsize=8, edgecolor='k')
    fig.subplots_adjust(top=0.85, hspace=0.3, wspace=0.25)
    
    return fig

#? Not needed as in the paper of Misra et al. (2023), the energy flux fraction is set at 0.5
def energy_flux_fraction(band_min, band_max, gamma=1.7, ref_min=0.3, ref_max=10.0):
    '''
    Compute the fraction of energy flux in a given energy band for a power-law spectrum N(E) ~ E^-gamma.

    Parameters
    ----------
    band_min : float,
        Minimum energy of the band [keV]. Default is 0.5 keV.
    band_max : float
        Maximum energy of the band [keV]. Default is 8.0 keV.
    gamma : float, optional
        Photon index of the power-law spectrum. Default is 1.7.
    ref_min : float, optional
        Minimum energy of the reference band [keV]. Default is 0.3 keV.
    ref_max : float, optional
        Maximum energy of the reference band [keV]. Default is 10.0 keV.

    Returns
    -------
    float
        Fraction of energy flux in the specified band.

    '''
    if band_min >= band_max or band_min < 0.0 or band_max < 0.0:
        raise ValueError("Both band values must be non-negative and band_min must be less than band_max")
    if ref_min >= ref_max or ref_min < 0.0 or ref_max < 0.0:
        raise ValueError("Both reference values must be non-negative and ref_min must be less than ref_max.")
    
    # fraction de flux d'énergie pour N(E) ~ E^-gamma
    a = 2.0 - gamma
    if abs(a) < 1e-8:
        return np.log(band_max/band_min) / np.log(ref_max/ref_min)
    return (band_max**a - band_min**a) / (ref_max**a - ref_min**a)

#* Checked and DocString updated 
#TODO Need to be included in the final data frame
def XRB_types(accretor_state, donor_state, donor_mass, ):
    '''
    Determine the type of X-ray binary (XRB) system based on the mass of the donor star. 
    The classification is as follows:
    - High-Mass X-ray Binary (HMXB): Donor mass ≥ 8 M⊙
    - Low-Mass X-ray Binary (LMXB): Donor mass ≤ 1 M⊙
    - Intermediate-Mass X-ray Binary (IMXB): 1 M⊙ < Donor mass < 8 M⊙
    - Non-X-ray Binary (Non-XRB): If the accretor is not a neutron star (NS) or black hole (BH), 
      or if the donor is a compact object (NS, BH, or white dwarf (WD)).
    
    Parameters
    ----------
    accretor_state : str
        State of the accretor. Can be either 'NS' or 'BH'.
    donor_state : str
        State of the donor star.
    donor_mass : float
        Mass of the donor star in [M⊙].
    
    Returns
    -------
    XRB_type : str
        Type of the X-ray binary system. Can be 'HMXB', 'IMXB', 'LMXB', or 'Non-XRB'.
    
    '''
    _allowed_accretor_states = ['NS', 'BH']
    _avoid_donor_states = ['NS', 'BH', 'WD']
    if (accretor_state in _allowed_accretor_states) and (donor_state not in _avoid_donor_states):

        if donor_mass >= 8.0:
            XRB_type = 'HMXB'  # High-Mass X-ray Binary
        elif donor_mass <= 1.0 :
            XRB_type = 'LMXB'  # Low-Mass X-ray Binary
        else:
            XRB_type = 'IMXB'  # Intermediate-Mass X-ray Binary
    else:
        XRB_type = 'Non-XRB'  # Not an X-ray binary

    return XRB_type
