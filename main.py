# -*- coding: utf-8 -*-
#
# Written by Adel Sohbi (https://github.com/adelshb).
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Testing script for Hidden Shift Problem (HSP) with quadratic boolean function. """

from argparse import ArgumentParser
import numpy as np

from src.qbhsp_gav11 import HSPOracle

def main(args) -> None:

    hsp = HSPOracle(N=args.N, A=args.A, b=args.b, c=args.c, s=args.s)
    hsp.build_oracle_circuit()
    
    from qiskit import BasicAer
    backend = BasicAer.get_backend('qasm_simulator')

    hsp.run(backend=backend, shots=100)
    print(hsp.counts)
    print("Real shift is : ", hsp.s)
    print("Guessed shift is: ", hsp.candidate_shift)

    # from IPython import embed; embed()

if __name__ == "__main__":

    args = ArgumentParser()
    N = 5

    # Quadratic form parameters
    args.N = N 
    A = np.random.randint(0, 2, (N, N))
    A ^= A.T
    args.A = A
    args.b = np.random.randint(0,2,N)
    # args.c = np.random.randint(0,2,1)
    args.c = 0

    # Hidden shift
    args.s = np.random.randint(0,2,N)

    main(args) 