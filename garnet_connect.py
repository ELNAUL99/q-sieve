"""No-cost connection check to IQM Garnet (20-qubit, IQM Resonance).

Reads the device name, qubit count and gate set. Submits NO circuit, books NO
timeslot, uses NO quota. Token read from .env (never printed).

Run in the IQM venv:  .venv-iqm/Scripts/python.exe garnet_connect.py
"""

import os
import sys
import statistics

SERVER = "https://resonance.iqm.tech/"
ALIAS = "garnet"


def load_token():
    env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    for line in open(env, encoding="utf-8"):
        line = line.strip()
        if line.startswith("IQM_RESONANCE_TOKEN="):
            tok = line.split("=", 1)[1].strip()
            if tok and tok != "your_iqm_token_here":
                return tok
    sys.exit("IQM_RESONANCE_TOKEN not set in .env")


def main():
    from iqm.qiskit_iqm import IQMProvider
    provider = IQMProvider(SERVER, quantum_computer=ALIAS, token=load_token())
    backend = provider.get_backend()
    print(f"connected to: {backend.name}")
    print(f"qubits:       {backend.num_qubits}")
    try:
        print(f"gate set:     {sorted(backend.target.operation_names)}")
    except Exception as e:
        print(f"gate set:     (n/a: {e})")
    try:
        tgt = backend.target
        ro = [p.error for q in range(backend.num_qubits)
              for p in [tgt["measure"].get((q,))] if p and p.error is not None]
        if ro:
            print(f"readout error: median {statistics.median(ro):.3f}")
    except Exception as e:
        print(f"readout error: (n/a: {e})")
    print("\nOK — connection works. NO circuit submitted, NO quota used.")


if __name__ == "__main__":
    main()
