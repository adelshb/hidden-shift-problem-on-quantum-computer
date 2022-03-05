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

""" Qiskit implementation of the Hidden Shift Problem (HSP) for quadratic boolean functions. Algorithm described in arXiv:1103.3017. """

from typing import List, Optional

import numpy as np
from numpy import ndarray
from scipy.optimize import fsolve

from qiskit import QuantumCircuit, execute, transpile
from qiskit.providers import Backend
from qiskit.circuit.library.arithmetic import QuadraticForm

class HSPOracle():
    """ Class for the quantum Oracle for the HSP problem for quadratic boolean function. """

    def __init__(self,
        N: int,
        A: Optional[ndarray] = None,
        b: Optional[ndarray] = None,
        c: Optional[ndarray] = None,
        s: Optional[ndarray] = None,
        backend: Optional[Backend] = None,
        ) -> None:
        """ Initialize the HSP Oracle. If parameters are not provided, the oracle will be initialized with random values.
        Args:
            N: Number of variables/qubits.
            A: Matrix of coefficients of the quadratic part of the objective function.
            b: Vector of coefficients of the linear part of the objective function.
            c: Vector of coefficients of the constant part of the objective function.
            s: Hidden shift vector.
            backend: The backend to use for quantum experiment.
        """

        # Get parameters
        self.N = N
        self.oracle = None

        if A is None:
            A = np.random.randint(0, 2, (N, N))
            A ^= A.T
            self.A = A
        else:
            self.A = A

        if b is None:
            self.b = np.random.randint(0,2,N)
        else:
            self.b = b

        if c is None:
            self.c = c = np.random.randint(0,2,1)
        else:
            self.c = c

        # Get the quadratic form for f
        self.f = QuadraticForm(num_result_qubits=1, quadratic=A, linear=b, offset=c)

        if s is None:
            s = np.random.randint(0,2,N)
        else:
            self.s = s

        if backend is None:
            self.backend = None
        else:
            self.backend = backend

        # Get the quadratic form for g (f with the hidden shift s)
        self.g= QuadraticForm(num_result_qubits=1, quadratic=A, linear=(2*A @ s + b) % 2, offset=(s.T @ b + c) % 2)


    def build_oracle_circuit(self,
        ) -> None:
        """ Build the quantum circuit for the HSP oracle. """

        self.oracle = QuantumCircuit(self.N + 1)

        for qubit in range(1, self.N+1):
            self.oracle.h(qubit) 

        self.oracle.append(self.f, [i for i in range(1, self.N+1)] + [0])     

        self.oracle.z(0)   

        self.oracle.append(self.g, [i for i in range(1, self.N+1)] + [0])

        for qubit in range(1, self.N+1):
            self.oracle.h(qubit)

        self.oracle.measure_all()

    def run(self,
        backend: Optional[Backend] = None,
        shots: Optional[int] = 1000,
        ) -> None:
        """ Run the algorithm. Proceed to the experiment and solve equations to make a guess on the shift. """

        if backend is None and self.backend is None:
            raise ValueError("No backend provided.")
        
        if self.oracle is None:
            print("Building the oracle...")
            self.build_oracle_circuit()

        qc = transpile(self.oracle, backend=backend)
        job = execute(qc, backend, shots=shots)
        self.counts = job.result().get_counts()

        self.outcomes = [[int(o) for o in outcome] for outcome in list(self.counts.keys())]
        self.candidate_shift = fsolve(self.func, np.random.randint(0, 2, self.N))
    
    def func(self, x: ndarray)-> List[int]:
        """ Function to be solved by fsolve. Correspond to the equations in  Step 4 in arXiv:1103.3017."""
        equations = []
        for outcome in self.outcomes:
            o = outcome[:-1]
            b = outcome[-1]
            prod = np.dot(x[::-1], o) % 2
            equations.append((prod - b) % 2)
        return equations