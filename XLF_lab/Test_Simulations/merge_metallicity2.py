from posydon.popsyn.binarypopulation import BinaryPopulation
from posydon.popsyn.io import binarypop_kwargs_from_ini
from posydon.utils.common_functions import convert_metallicity_to_string
import argparse
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metallicity", type=float)
    args = parser.parse_args()
    ini_kw = binarypop_kwargs_from_ini("Test_Data_Misra2.ini")
    ini_kw["metallicity"] = args.metallicity
    str_met = convert_metallicity_to_string(args.metallicity)
    ini_kw["temp_directory"] = str_met+"_Zsun_" + ini_kw["temp_directory"]
    synpop = BinaryPopulation(**ini_kw)
    path_to_batch = ini_kw["temp_directory"]
    tmp_files = [os.path.join(path_to_batch, f) for f in os.listdir(path_to_batch) if os.path.isfile(os.path.join(path_to_batch, f))]
    tmp_files = sorted(tmp_files, key=lambda x: int(x.split(".h5")[0].split(".")[-1]))
    synpop.combine_saved_files(str_met+ "_Zsun_population2.h5", tmp_files)
    print("done")
    if len(os.listdir(path_to_batch)) == 0:
        os.rmdir(path_to_batch)