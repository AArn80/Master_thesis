from posydon.popsyn.binarypopulation import BinaryPopulation
from posydon.popsyn.io import binarypop_kwargs_from_ini
from posydon.utils.common_functions import convert_metallicity_to_string
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('metallicity', type=float)
    args = parser.parse_args()
    ini_kw = binarypop_kwargs_from_ini('population_params_XLF.ini')
    ini_kw['metallicity'] = args.metallicity
    str_met = convert_metallicity_to_string(args.metallicity)
    ini_kw['temp_directory'] = str_met+'_Zsun_' + ini_kw['temp_directory']
    synpop = BinaryPopulation(**ini_kw)
    synpop.evolve()