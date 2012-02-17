"""Grow the beta lookup table

Command line usage:
Call the following command from the directory with the pickled lookup table
python beta_dictionary_builder.py S_min S_max N_min N_max

"""

import mete
import sys

if __name__ == "__main__":
    S_min = int(sys.argv[1])
    S_max = int(sys.argv[2])
    N_min = int(sys.argv[3])
    N_max = int(sys.argv[4])
    mete.build_beta_dict(S_min, S_max, N_max, N_min=N_min)