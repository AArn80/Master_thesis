import matplotlib.pyplot as plt                                                              # type : ignore
import numpy as np
import posydon                                                                               # type : ignore
from posydon.utils.constants import const                                                    # type : ignore 
from posydon.utils.population import Population                                              # type : ignore
from posydon.utils.common_functions import orbital_separation_from_period, roche_lobe_radius # type : ignore
from posydon.utils.posydonwarning import Pwarn                                               # type : ignore

#! A version of the file New_Functions.py on the 20.02.26 containing the code that allows to create the histogram

def Systems_selection(synthetic_population, filename):
    """
    Selects compact-object binaries with non-compact companions and export.

    Keeps systems that have one compact object (NS/BH) and a companion that is
    not in the avoided set, and that are not in excluded system states.

    The outputed file can be later use in create_transient_population() with X-ray binary selection function. 

    Parameters
    ----------
    synthetic_population : posydon.Population object
        Population object containing binary systems.

    filename : str
        Name for the file that will be exported (without the extension).

    Returns
    -------
    exported_file : posydon.Population object
        The exported_file will be called "filename.h5".

    """

    if not isinstance(synthetic_population, Population):
        Pwarn("synthetic_population must be a posydon Population", category="InappropriateValueWarning")
        return None
    if not isinstance(filename, str) or filename.strip() == "":
        Pwarn("filename must be a non-empty string", category="InappropriateValueWarning")
        return None
    
    compact_types = {'NS', 'BH'}
    objects_to_avoid = {'NS', 'BH', 'massless_remnant', 'WD'}  # Avoid double CO, WD, massless remnants companions
    bad_system_states = {'disrupted', 'ERR', 'initial_RLOF'}

    tmp_oneline = synthetic_population.oneline[['state_f', 'S1_state_f', 'S2_state_f', 'event_f']]

    state1, state2, state_sys = tmp_oneline['S1_state_f'], tmp_oneline['S2_state_f'], tmp_oneline['state_f']

    mask = (
        (state1.isin(compact_types) & ~state2.isin(objects_to_avoid) & ~state_sys.isin(bad_system_states))
        | (state2.isin(compact_types) & ~state1.isin(objects_to_avoid) & ~state_sys.isin(bad_system_states))
    )
    idx = tmp_oneline.index[mask]

    exported_file = synthetic_population.export_selection(list(idx), f"{filename}.h5")

    saved_path = exported_file if exported_file is not None else f"{filename}.h5"
    print(f"The selected systems are stored in {saved_path}.")

    return exported_file
    


#============================================================================================
selection = Population('BHNS_only.h5')
selection_oneline =selection.oneline[:]

orbital_separation = orbital_separation_from_period(selection_oneline['orbital_period_f'], selection_oneline['S1_mass_f'], selection_oneline['S2_mass_f'])

Donor_RL_f = roche_lobe_radius(selection_oneline['S2_mass_f'].to_numpy(), selection_oneline['S1_mass_f'].to_numpy(), orbital_separation)

mask_be = ((selection_oneline['S2_state_f'] == 'H-rich_Core_H_burning')
            & (selection_oneline['S2_mass_f'] >= 6.0)
            & (selection_oneline['state_f'] == 'detached')
            &  (selection_oneline['S2_surf_avg_omega_div_omega_crit_f'] >= 0.7)
            & (selection_oneline['orbital_period_f'] >= 10.0)
            & (selection_oneline['orbital_period_f'] <= 300.0)
            & (Donor_RL_f <= (100.0 * 10**(selection_oneline['S2_log_R_f'])))
            )


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

def BH_efficiency(spin): 
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

    return eta

def Stefan_Boltzmann_law(Luminosity, Radius):
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

@np.vectorize
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

def Wind_velocity_BHL(accretor_type, donor_mass, donor_logR, donor_logL, Mdot_wind, 
                      donor_surface_H1, donor_He_core_mass, Donor_RL_radius,
                      Wind_fed_scheme = None, Wind_velo_scheme ='Kudritzki+2000'):
  
    # In [1] Misra et al. A&A 672. A99 (2023), based on [4] Hirai & Mandel 2021
    if (accretor_type=='BH' and 10**donor_logR < 0.8 * Donor_RL_radius and Wind_fed_scheme == 'HiraiMandel21'): # Must fill at least 80% of its Roche lobe to get winds
        V_wind = 0.0 # No wind
    else:

        Teff = Stefan_Boltzmann_law(10.0**donor_logL, 10.0**donor_logR) # [K]

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
            if np.log10(Mdot_wind) >= -5.25:
                slope = (3.7 - 3.25) / (-2.5 + 5.25)
            else:
                slope = (3.25 - 3.75) / (-5.25 + 7.25)
            V_wind = 10 ** (slope * np.log10(Mdot_wind) + 3.25 + 5.25 * slope) * 1e5  # Convert from km/s to cm/s
        else:
            pass

    return V_wind


def BHL_mdot_acc(Macc_Msun, donor_mass_Msun, accretor_type, 
                 Mdot_w_Msun_per_yr, a_cm, ecc, V_wind, **kwargs):
    
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


def Be_XLF(period):

    Lx = (10.0**35) * 10.0**(4.53 - (1.50 * np.log10(period))) 
    return Lx


@np.vectorize
def L_bol(bin_state, lg_mtransfer_rate, acc_log_mdot, donor_log_mdot, acc_state, donor_log_L, 
          Macc, Mdonor, Acc_log_radius, Donor_log_radius, Donor_surface_H1, donor_He_core_mass, 
          Donor_RL_radius, a_cm, ecc):
    

    # Mass-transfer rate (donor -> vicinity of accretor) through RLO
    Mdot_RLO_Msunyr = (0.0 if lg_mtransfer_rate is None or not np.isfinite(lg_mtransfer_rate) else 10.0**float(lg_mtransfer_rate))
    Mdot_RLO_gps = Mdot_RLO_Msunyr * const.Msun / const.secyer  # [g/s]

    # Donor mass loss rate (total, including RLO + winds) [Msun/yr]
    Mdot_donor_total_Msunyr = (0.0 if donor_log_mdot is None or not np.isfinite(donor_log_mdot) else 10.0**float(donor_log_mdot))
    mdot_wind_only_Msunyr = max(Mdot_donor_total_Msunyr - Mdot_RLO_Msunyr, 0.0) # [Msun/yr]


    if bin_state == 'RLO1' or bin_state == 'RLO2':
        Mdot = 10**lg_mtransfer_rate * const.Msun/const.secyer
        #print(f"State : {bin_state}, Mdot_RLO = {Mdot:.2e} g/s")
    elif bin_state == 'detached':
        if acc_log_mdot <= -60.0:
            Vwind = Wind_velocity_BHL(acc_state, Mdonor, Donor_log_radius, donor_log_L, mdot_wind_only_Msunyr, Donor_surface_H1, donor_He_core_mass, Donor_RL_radius)
            Mdot = BHL_mdot_acc(Macc, Mdonor, acc_state, mdot_wind_only_Msunyr, a_cm, ecc, Vwind)
        else:
            Mdot = 10**acc_log_mdot * const.Msun/const.secyer
        #print(f"State : {bin_state}, Mdot_BHL = {Mdot:.2e} g/s")

    if acc_state == 'BH':
        eta = BH_efficiency(Macc)
    elif acc_state == 'NS':
        eta = NS_efficiency(Macc, Acc_log_radius)
    else:
        raise ValueError(f"Unknown accretor state: {acc_state}. Expected 'BH' or 'NS'.")

    Medd, L_edd = Eddington_limit(Macc, Donor_surface_H1, eta)
    
    if Mdot/Medd <= 1:
        Lx = Mdot * eta * const.clight**2
    else:
        if Mdot/Medd > 8.5:
            b = 73. / (Mdot/Medd)**2
            if b < 3.2e-3:
                b = 3.2e-3
        else:
            b = 1.
        Lx = (L_edd/b) * (1 + np.log(Mdot/Medd))

    return Lx


def XRB_selection_function(history_chunk, oneline_chunk, formation_channels_chunk = None):

    

    return



Df_Be = selection_oneline[mask_be].copy()

Df_Be['Be_XLF'] = Be_XLF(selection_oneline.loc[mask_be, 'orbital_period_f'].to_numpy())

Other_systems = selection_oneline[~selection_oneline.index.isin(Df_Be.index)]
RLO_sys = Other_systems[(Other_systems['state_f'] == 'RLO2')]
Wind_sys = Other_systems[(Other_systems['state_f'] == 'detached')]



state1 = RLO_sys['S1_state_f']
compact_objects = ['NS', 'BH']
is_S1_compact = state1.isin(compact_objects)

Macc = np.where(is_S1_compact, RLO_sys['S1_mass_f'], RLO_sys['S2_mass_f'])
Mdonor = np.where(is_S1_compact, RLO_sys['S2_mass_f'], RLO_sys['S1_mass_f'])
Donor_surface_H1 = np.where(is_S1_compact, RLO_sys['S2_surface_h1_f'], RLO_sys['S1_surface_h1_f'])
Donor_log_radius = np.where(is_S1_compact, RLO_sys['S2_log_R_f'], RLO_sys['S1_log_R_f'])
Donor_log_lum = np.where(is_S1_compact, RLO_sys['S2_log_L_f'], RLO_sys['S1_log_L_f'])
Acc_log_radius = np.where(is_S1_compact, RLO_sys['S1_log_R_f'], RLO_sys['S2_log_R_f'])
acc_state = np.where(is_S1_compact, RLO_sys['S1_state_f'], RLO_sys['S2_state_f'])
donor_state = np.where(is_S1_compact, RLO_sys['S2_state_f'], RLO_sys['S1_state_f'])
donor_log_mdot = np.where(is_S1_compact, RLO_sys['S2_lg_mdot_f'], RLO_sys['S1_lg_mdot_f'])
acc_log_mdot = np.where(is_S1_compact, RLO_sys['S1_lg_mdot_f'], RLO_sys['S2_lg_mdot_f'])
donor_He_core_mass = np.where(is_S1_compact, RLO_sys['S2_he_core_mass_f'], RLO_sys['S1_he_core_mass_f'])
bin_state = RLO_sys['state_f']
lg_mtransfer_rate = RLO_sys['lg_mtransfer_rate_f']
a_cm = const.Rsun * orbital_separation_from_period(RLO_sys['orbital_period_f'], RLO_sys['S1_mass_f'], RLO_sys['S2_mass_f'])
Donor_RL_radius = roche_lobe_radius(Mdonor, Macc, a_cm)

ecc = RLO_sys['eccentricity_f']


RLO_sys['Lx'] = L_bol(bin_state, lg_mtransfer_rate, acc_log_mdot, donor_log_mdot, acc_state, Donor_log_lum,
                     Macc, Mdonor, Acc_log_radius, Donor_log_radius, Donor_surface_H1,
                     donor_He_core_mass, Donor_RL_radius, a_cm, ecc)



state1 = Wind_sys['S1_state_f']
compact_objects = ['NS', 'BH']
is_S1_compact = state1.isin(compact_objects)

Macc = np.where(is_S1_compact, Wind_sys['S1_mass_f'], Wind_sys['S2_mass_f'])
Mdonor = np.where(is_S1_compact, Wind_sys['S2_mass_f'], Wind_sys['S1_mass_f'])
Donor_surface_H1 = np.where(is_S1_compact, Wind_sys['S2_surface_h1_f'], Wind_sys['S1_surface_h1_f'])
Donor_log_radius = np.where(is_S1_compact, Wind_sys['S2_log_R_f'], Wind_sys['S1_log_R_f'])
Donor_log_lum = np.where(is_S1_compact, Wind_sys['S2_log_L_f'], Wind_sys['S1_log_L_f'])
Acc_log_radius = np.where(is_S1_compact, Wind_sys['S1_log_R_f'], Wind_sys['S2_log_R_f'])
acc_state = np.where(is_S1_compact, Wind_sys['S1_state_f'], Wind_sys['S2_state_f'])
donor_state = np.where(is_S1_compact, Wind_sys['S2_state_f'], Wind_sys['S1_state_f'])
donor_log_mdot = np.where(is_S1_compact, Wind_sys['S2_lg_mdot_f'], Wind_sys['S1_lg_mdot_f'])
acc_log_mdot = np.where(is_S1_compact, Wind_sys['S1_lg_mdot_f'], Wind_sys['S2_lg_mdot_f'])
donor_He_core_mass = np.where(is_S1_compact, Wind_sys['S2_he_core_mass_f'], Wind_sys['S1_he_core_mass_f'])
bin_state = Wind_sys['state_f']
lg_mtransfer_rate = Wind_sys['lg_mtransfer_rate_f']
a_cm = const.Rsun * orbital_separation_from_period(Wind_sys['orbital_period_f'], Wind_sys['S1_mass_f'], Wind_sys['S2_mass_f'])
Donor_RL_radius = roche_lobe_radius(Mdonor, Macc, a_cm)

ecc = Wind_sys['eccentricity_f']

Wind_sys['Lx'] = L_bol(bin_state, lg_mtransfer_rate, acc_log_mdot, donor_log_mdot, acc_state, Donor_log_lum,
                      Macc, Mdonor, Acc_log_radius, Donor_log_radius, Donor_surface_H1,
                      donor_He_core_mass, Donor_RL_radius, a_cm, ecc)



H1_rich_RLO = RLO_sys['S2_state_f'].str.contains('H-rich|accreted_He', regex=True, na=False)
stripped_He_RLO = RLO_sys['S2_state_f'].str.contains('stripped_He', na=False)
RLO_BH = RLO_sys['S1_state_f'] == 'BH'
RLO_NS = RLO_sys['S1_state_f'] == 'NS'

H1_rich_Wind = Wind_sys['S2_state_f'].str.contains('H-rich|accreted_He', regex=True, na=False)
stripped_He_Wind = Wind_sys['S2_state_f'].str.contains('stripped_He', na=False)
Wind_BH = Wind_sys['S1_state_f'] == 'BH'
Wind_NS = Wind_sys['S1_state_f'] == 'NS'

Lx_Total = np.concatenate([RLO_sys['Lx'].to_numpy(), Wind_sys['Lx'].to_numpy(), Df_Be['Be_XLF'].to_numpy()])






plt.figure(figsize=(6,5), dpi = 150)

h, bins = np.histogram(Df_Be['Be_XLF'], bins=np.logspace(35, 45, 100))
h = np.cumsum(h[::-1])[::-1]
plt.step(bins[:-1], h, where='post', color = '#808080', label='Be-XRBs')

h, bins = np.histogram(RLO_sys['Lx'][H1_rich_RLO & RLO_BH], bins=np.logspace(35, 45, 100))
h = np.cumsum(h[::-1])[::-1]
plt.step(bins[:-1], h, where='post', label = 'RLO-BH-H-rich', color ='#540B0E')

h, bins = np.histogram(RLO_sys['Lx'][H1_rich_RLO & RLO_NS], bins=np.logspace(35, 45, 100))
h = np.cumsum(h[::-1])[::-1]
plt.step(bins[:-1], h, where='post', label = 'RLO-NS-H-rich', color ='#DA1E37')

h, bins = np.histogram(RLO_sys['Lx'][stripped_He_RLO & RLO_BH], bins=np.logspace(35, 45, 100))
h = np.cumsum(h[::-1])[::-1]
plt.step(bins[:-1], h, where='post', label = 'RLO-BH-stripped_He', color='#B760C4')
h, bins = np.histogram(RLO_sys['Lx'][stripped_He_RLO & RLO_NS], bins=np.logspace(35, 45, 100))
h = np.cumsum(h[::-1])[::-1]
plt.step(bins[:-1], h, where='post', label = 'RLO-NS-stripped_He', color='#F9E979')


h, bins = np.histogram(Wind_sys['Lx'][H1_rich_Wind & Wind_BH], bins=np.logspace(35, 45, 100))
h = np.cumsum(h[::-1])[::-1]
plt.step(bins[:-1], h, where='post', label = 'Wind-BH-H-rich', linestyle='--', color='#540B0E')

h, bins = np.histogram(Wind_sys['Lx'][H1_rich_Wind & Wind_NS], bins=np.logspace(35, 45, 100))
h = np.cumsum(h[::-1])[::-1]
plt.step(bins[:-1], h, where='post', label = 'Wind-NS-H-rich', linestyle='--', color='#DA1E37')

h, bins = np.histogram(Wind_sys['Lx'][stripped_He_Wind & Wind_BH], bins=np.logspace(35, 45, 100))
h = np.cumsum(h[::-1])[::-1]
plt.step(bins[:-1], h, where='post', label = 'Wind-BH-stripped_He', linestyle='--', color='#B760C4')

h, bins = np.histogram(Wind_sys['Lx'][stripped_He_Wind & Wind_NS], bins=np.logspace(35, 45, 100))
h = np.cumsum(h[::-1])[::-1]
plt.step(bins[:-1], h, where='post', label = 'Wind-NS-stripped_He', linestyle='--', color='#F9E979')


h, bins = np.histogram(Lx_Total, bins=np.logspace(35, 45, 100))
h = np.cumsum(h[::-1])[::-1]
plt.step(bins[:-1], h, where='post', label = 'Total', color = '#1F77B4')


plt.xscale('log')
plt.yscale('log')
plt.xlim(1e35, 1e44)
plt.legend()





