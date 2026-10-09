"""Print the shape of a finished Q50 job. No new circuits."""

import os

import requests

JOBS = ["67303a05-3949-4788-a63f-4aec646a4acf"]
base = os.environ["Q50_CORTEX_URL"].rstrip("/")
headers = {
    "Authorization": "Bearer " + os.environ["IQM_TOKEN"],
    "Accept": "application/json",
}

for job in JOBS:
    response = requests.get(f"{base}/api/jobs/{job}", headers=headers, timeout=60)
    print("job", job, response.status_code, flush=True)
    if response.status_code != 200:
        print(" ".join(response.text.split())[:500], flush=True)
        continue
    data = response.json()
    results = data.get("results", data)
    print("job", job, response.status_code, "status", results.get("status"), flush=True)
    print("results_keys", sorted(results.keys()), flush=True)
    meas = results.get("measurements")
    print("n_meas", len(meas) if isinstance(meas, list) else type(meas).__name__, flush=True)
    if not isinstance(meas, list) or not meas:
        continue
    first = meas[0]
    if not isinstance(first, dict):
        print("first", type(first).__name__, flush=True)
        continue
    keys = list(first.keys())
    print("n_keys", len(keys), "keys", keys[:12], flush=True)
    value = first[keys[0]]
    print("value_type", type(value).__name__, "len", len(value), "shot0", value[0], flush=True)
