"""No-cost connection check to IBM Quantum (newer Cloud platform).

Needs IBM_QUANTUM_TOKEN and IBM_QUANTUM_CRN (the instance) in .env. Lists the
real backends and your open-plan time; submits NO job, uses NO QPU time.

Run in the IBM venv:
    .venv-ibm/Scripts/python.exe ibm_connect.py
"""

import os
import sys


def env(key):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith(key + "="):
            v = line.split("=", 1)[1].strip()
            if v and not v.startswith("your_"):
                return v
    return None


def main():
    token = env("IBM_QUANTUM_TOKEN")
    crn = env("IBM_QUANTUM_CRN")
    if not token:
        sys.exit("IBM_QUANTUM_TOKEN not set in .env")
    if not crn:
        sys.exit("IBM_QUANTUM_CRN not set in .env — paste your instance CRN there "
                 "(crn:v1:bluemix:...).")

    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=crn)

    backends = service.backends(operational=True, simulator=False)
    print(f"connected. {len(backends)} real backend(s):")
    for b in backends:
        mc = getattr(b, "max_circuits", None)
        print(f"  {b.name:22} qubits={b.num_qubits}  max_circuits/job={mc}")
    try:
        lb = service.least_busy(operational=True, simulator=False)
        print(f"least busy: {lb.name} ({lb.num_qubits} qubits)")
    except Exception as e:
        print(f"least_busy: n/a ({e})")
    print("\nOK — connection works. NO job submitted, NO QPU time used.")


if __name__ == "__main__":
    main()
