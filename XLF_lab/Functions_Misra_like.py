import os
import sys

import numpy as np
import matplotlib.pyplot as plt                              # type: ignore
from scipy.optimize import newton                            # type: ignore 
import pandas as pd                                          # type: ignore

import posydon.utils.constants as const                      # type: ignore
from posydon.utils.common_functions import roche_lobe_radius # type: ignore
from posydon.utils.posydonwarning import Pwarn               # type: ignore

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
    
    ## To control that the spin is in the range that gives the maximal efficiency
    #a_star = float(np.clip(spin, -0.998, 0.998))

    ##=== Compute efficiency [1]=== 
    #
    ## Use sign = +1 for spin ≥ 0 (prograde) and -1 otherwise (retrograde)
    #sign = 1 if a_star >= 0 else -1
#
    #Z1 = 1 + (1 - a_star**2)**(1/3) * ((1 + a_star)**(1/3) + (1 - a_star)**(1/3))
    #Z2 = np.sqrt(3 * a_star**2 + Z1**2)
#
    ## Radius at the innermost stable circular orbit (ISCO)
    #r_tilde = 3 + Z2 - sign * np.sqrt((3 - Z1)*(3 + Z1 + 2 * Z2)) # Dimensionless
#
    ## Energy at ISCO in units of mc^2 (i.e E_ISCO < 1, E = 1 means particle energy at infinity)
    #E_ISCO = (r_tilde**(3/2) - 2 * r_tilde**(1/2) + a_star)/(r_tilde**(3/4) * np.sqrt(r_tilde**(3/2) - 3 * r_tilde**(1/2) + 2 * a_star))
    #
    ## Radiative efficiency
    #eta = 1 - E_ISCO

    E_ISCO = 0.0 
    r_tilde = 6.0
    m_ini = float(mass) * np.sqrt(r_tilde/6.0)
    eta = 1- np.sqrt(1 - (min(mass, np.sqrt(6) * m_ini) / (3 * m_ini))**2)

    return E_ISCO, r_tilde, eta


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
    IMF_corr = Compute_IMF_normalization_correction(transient_population)

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
    
    Weights = weights_series * IMF_corr * Tot_underlying_mass

    return Weights, IMF_corr


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
    #? Modified for Functions_Misra_like.py
    L = transient_population.population['Lx_Total'].to_numpy(dtype = float) * norm_factor
    #? End modification =====
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


def XRB_selection_function1(history_chunk, oneline_chunk, formation_channel_chunk, **kwargs):

    observe_wind_XRB = kwargs.get('observe_wind_XRB', False)

    # iterate over each system in the chunk and compute Lx/Beaming per row
    rows = oneline_chunk if isinstance(oneline_chunk, pd.DataFrame) else pd.DataFrame([oneline_chunk])

    Compact_types = {'NS','BH'}
    Objects_to_avoid = {'NS','BH','massless_remnant', 'WD'} # Avoid double CO, WD, massless remnants companions
    bad_systems_states = {'disrupted', 'ERR'}


    state1, state2, state_sys = oneline_chunk['S1_state_f'], oneline_chunk['S2_state_f'], oneline_chunk['state_f']

    mask = ((state1.isin(Compact_types) & ~state2.isin(Objects_to_avoid)) & ~state_sys.isin(bad_systems_states)) | ((state2.isin(Compact_types) & ~state1.isin(Objects_to_avoid)) & ~state_sys.isin(bad_systems_states))
    
    oneline_selection = oneline_chunk.loc[mask].copy()
    indices = oneline_selection.index.to_numpy()
    DataFrame = pd.DataFrame(index = indices)

    S1_radius_Rsun = 10**oneline_selection['S1_log_R_f']
    S2_radius_Rsun = 10**oneline_selection['S2_log_R_f']

    # Compactness mask aligned to selection
    is_S1_compact = oneline_selection['S1_state_f'].isin(Compact_types)

    DataFrame['state_sys'] = oneline_selection['state_f']
    DataFrame['Orbital_period'] = oneline_selection['orbital_period_f']  # [days]
    DataFrame['Log_Mtransfer_rate'] = oneline_selection['lg_mtransfer_rate_f'] # [M⊙/yr]
    DataFrame['time'] = oneline_selection['time_f'] - oneline_selection['time_i'] # [yr]
    DataFrame['metallicity'] = oneline_selection['metallicity']

    DataFrame['Accretor_state'] = np.where(is_S1_compact, oneline_selection['S1_state_f'], oneline_selection['S2_state_f'])
    DataFrame['Accretor_mass'] = np.where(is_S1_compact, oneline_selection['S1_mass_f'], oneline_selection['S2_mass_f'])
    DataFrame['Accretor_Radius'] = np.where(is_S1_compact, S1_radius_Rsun, S2_radius_Rsun)
    DataFrame['Accretor_spin'] = np.where(is_S1_compact, oneline_selection['S1_spin_f'], oneline_selection['S2_spin_f'])
    DataFrame['Accretor_logMdot'] = np.where(is_S1_compact, oneline_selection['S1_lg_mdot_f'], oneline_selection['S2_lg_mdot_f'])

    DataFrame['Donor_mass'] = np.where(is_S1_compact, oneline_selection['S2_mass_f'], oneline_selection['S1_mass_f'])
    DataFrame['Donor_Radius'] = np.where(is_S1_compact, S2_radius_Rsun, S1_radius_Rsun)
    DataFrame['Donor_state'] = np.where(is_S1_compact, oneline_selection['S2_state_f'], oneline_selection['S1_state_f'])
    DataFrame['Donor_surf_H1'] = np.where(is_S1_compact, oneline_selection['S2_surface_h1_f'], oneline_selection['S1_surface_h1_f'])
    DataFrame['Donor_surf_velo'] = np.where(is_S1_compact, oneline_selection['S2_surf_avg_omega_div_omega_crit_f'], oneline_selection['S1_surf_avg_omega_div_omega_crit_f'])

    
    a_cm = np.zeros(len(DataFrame))
    Donor_RL = np.zeros(len(DataFrame))
    # Store string labels, initialize with NaN using object dtype
    Mechanism_Xray = np.empty(len(DataFrame), dtype=object)
    Mechanism_Xray[:] = np.nan
    Lx = np.zeros(len(DataFrame))
    b = np.ones(len(DataFrame))
    Donor_radius_cm = DataFrame['Donor_Radius'] * const.Rsun # For Be-XRB conditions

    Donor_relative_RLO = np.zeros(len(DataFrame)) 

    RLO_systems = ['initial_RLOF', 'RLO1', 'RLO2']
    Wind_systems = ['detached']
    # Loop row-wise to compute per-system values safely


    for i, idx in enumerate(DataFrame.index):

        a_cm[i] = Orbital_sep_and_velo_from_period(DataFrame.loc[idx, 'Accretor_mass'], DataFrame.loc[idx, 'Donor_mass'], DataFrame.loc[idx, 'Orbital_period'])[0]
        Donor_RL[i] = roche_lobe_radius(DataFrame.loc[idx, 'Donor_mass'], DataFrame.loc[idx, 'Accretor_mass'], a_cm[i])
        Donor_relative_RLO[i] = (Donor_radius_cm[idx] - Donor_RL[i]) / Donor_RL[i] if Donor_RL[i] > 0 else 0 # Relative Roche lobe filling factor    
        
        if ((DataFrame.loc[idx, 'Donor_state'] == 'H-rich_Core_H_burning') and (DataFrame.loc[idx, 'Donor_mass'] >= 3.0) and (DataFrame.loc[idx, 'state_sys'] == 'detached')
            and (Donor_RL[i] <= (100.0 * Donor_radius_cm[idx])) and (DataFrame.loc[idx, 'Donor_surf_velo'] >= 0.7) and (DataFrame.loc[idx, 'Orbital_period'] >= 10.0)
            and (DataFrame.loc[idx, 'Orbital_period'] <= 300.0)):
            # Be-XRBs
            Lx[i] = 10.0**35 * 10.0**(4.53 - (1.50 * np.log10(DataFrame.loc[idx, 'Orbital_period'])) )
            b[i] = 0.1 # assuming 10% duty cycle
            donor_val = DataFrame.loc[idx, 'Donor_state']
            donor_str = '' if pd.isna(donor_val) else str(donor_val)
            DataFrame.loc[idx, 'Donor_state'] = f"{donor_str} (Be star)"
        else: 
            # Choose accretion rate per system
            log_mtransfer = DataFrame.loc[idx, 'Log_Mtransfer_rate']
            acc_log_mdot = DataFrame.loc[idx, 'Accretor_logMdot']
            if pd.isna(log_mtransfer) and pd.isna(acc_log_mdot):
                Lx[i] = 0.0
                b[i] = 1.0
            else:
                mdot_i = 10**log_mtransfer if not pd.isna(log_mtransfer) else 10**acc_log_mdot
                mdot_i = mdot_i * const.Msun / const.secyer # Convert from M⊙/yr to g/s for the Eddington comparison

                # Efficiency per accretor type
                if DataFrame.loc[idx, 'Accretor_state'] == 'BH':
                    eta = BH_efficiency(DataFrame.loc[idx, 'Accretor_mass'], DataFrame.loc[idx, 'Accretor_spin'])[2]
                elif DataFrame.loc[idx, 'Accretor_state'] == 'NS':
                    acc_radius = DataFrame.loc[idx, 'Accretor_Radius']
                    acc_logR = np.log10(acc_radius) if np.isfinite(acc_radius) and acc_radius > 0 else None
                    eta = NS_efficiency(DataFrame.loc[idx, 'Accretor_mass'], acc_logR, B_surf=None, accretor_logMdot=acc_log_mdot)[0]
                else:
                    eta = 0.0

                # Mdot_edd converted from solar to cgs units
                mdot_edd_gps = Eddington_limit(DataFrame.loc[idx, 'Accretor_mass'], DataFrame.loc[idx, 'Donor_surf_H1'], eta)[0]
                mdot = mdot_i / mdot_edd_gps if mdot_edd_gps > 0 else 0.0

                # RLO and wind regimes with beaming
                if mdot <= 1:
                    Lx[i] = mdot_i * eta * const.clight**2
                    b[i] = 1.0
                else:
                    if mdot > 150.0:
                        b[i] = 3.2e-3
                    elif mdot > 8.5:
                        b[i] = 73. / (mdot)**2
                    else:
                        b[i] = 1.
                    Lx[i] = mdot_edd_gps * eta * const.clight**2 * (1 + np.log(mdot)) / b[i]

        # Determine X-ray mechanism label, excluding Be-star donors from RLO/Wind
        donor_state_val = DataFrame.loc[idx, 'Donor_state']
        donor_state_str = '' if pd.isna(donor_state_val) else str(donor_state_val)
        is_be_star = '(Be star)' in donor_state_str
        if not is_be_star:
            if DataFrame.loc[idx, 'state_sys'] in RLO_systems:
                Mechanism_Xray[i] = 'RLO'
            elif DataFrame.loc[idx, 'state_sys'] in Wind_systems:
                Mechanism_Xray[i] = 'Wind'
            else:
                Mechanism_Xray[i] = np.nan
        else:
            Mechanism_Xray[i] = 'Be'

        # If observe_wind_XRB == True, check if wind accretion rate is high enough to make accretion disc
        # bychecking the donor is filling 80% of its Roche lobe.
        # Hirai & Mandel (2021)
        if (observe_wind_XRB == True) & ((DataFrame.loc[idx, 'state_sys'] == 'detached') & (DataFrame.loc[idx, 'Accretor_state'] == 'BH')):
            if (( Donor_radius_cm[idx] / Donor_RL[i]) < 0.8):
                Lx[i] = 10**-54
                b[i] = 1.0
        
        if Donor_relative_RLO[i] == 0.0:
            Lx[i] = 0.0
            b[i] = 1.0
            Mechanism_Xray[i] = np.nan

    DataFrame['Donor_RL'] = Donor_RL
    DataFrame['Mechanism_Xray'] = Mechanism_Xray
    DataFrame['Beaming'] = b
    DataFrame['Lx_Total'] = Lx
        
    return DataFrame
    