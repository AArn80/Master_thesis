from posydon.popsyn.binarypopulation import BinaryPopulation
from posydon.popsyn.io import binarypop_kwargs_from_ini
from posydon.utils.common_functions import convert_metallicity_to_string
from posydon.binary_evol.simulationproperties import SimulationProperties
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('metallicity', type=float)
    args = parser.parse_args()
    ini_kw = binarypop_kwargs_from_ini('./Pop_params_GRMHD_Bis.ini')
    ini_kw['metallicity'] = args.metallicity
    str_met = convert_metallicity_to_string(args.metallicity)
    ini_kw['temp_directory'] = str_met+'_Zsun_' + ini_kw['temp_directory']
    sim_props = SimulationProperties.from_ini('./Pop_params_GRMHD_Bis.ini')
    synpop = BinaryPopulation(population_properties=sim_props, **ini_kw)
    synpop.evolve()
