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

""" Qiskit implementation of the Hidden Shift Problem (HSP) for bent function. Algorithm described in Theorem 7 in arXiv:0811.3208. """

from typing import Optional

import numpy as np
from numpy import ndarray

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

        self.oracle = QuantumCircuit(2*self.N + 3, self.N+1)

        for qubit in range(1 + 2*self.N):
            self.oracle.h(qubit)

        self.oracle.h(2*self.N+1 + 1)
        self.oracle.p(np.pi, 2*self.N+1 + 1)
        self.oracle.h(2*self.N+1 + 2)
        self.oracle.p(np.pi, 2*self.N+1 + 2)  

        for qubit in range(self.N):
            self.oracle.cz(qubit+1, qubit+self.N+1)    

        self.oracle.append(self.g.control(1), [0] + [i for i in range(self.N+1, 2*self.N+1)] + [2*self.N+1])     

        self.oracle.x(0)   

        self.oracle.append(self.f.control(1), [0] + [i for i in range(self.N+1, 2*self.N+1)] + [2*self.N+2])   

        self.oracle.x(0)   

        for qubit in range(self.N):
            self.oracle.cz(qubit+1, qubit+self.N+1)  
            
        for qubit in range(1 + 2*self.N):
            self.oracle.h(qubit) 

        self.oracle.measure(range(self.N+1), range(self.N+1))

    def run(self,
        backend: Optional[Backend] = None,
        shots: Optional[int] = 1000,
        ) -> None:
        """ Run the algorithm. """

        if backend is None and self.backend is None:
            raise ValueError("No backend provided.")
        
        if self.oracle is None:
            print("Building the oracle...")
            self.build_oracle_circuit()

        # from IPython import embed; embed()
        qc = transpile(self.oracle, backend=backend)
        job = execute(qc, backend, shots=shots)
        self.counts = job.result().get_counts()