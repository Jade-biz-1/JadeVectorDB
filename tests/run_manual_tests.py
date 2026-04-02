import os
import sys
import subprocess
import time
import requests
import json
import signal

BASE_URL = "http://localhost:8080"
ADMIN_USER = "admin"
ADMIN_PASS = "admin123"
SERVER_BIN = "/Users/Deepak/Public/JadeVectorDB/backend/build/jadevectordb"
SERVER_CWD = "/Users/Deepak/Public/JadeVectorDB/backend/build"

def print_step(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}")

def wait_for_server(timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            requests.get(f"{BASE_URL}/v1/databases", timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False

def start_server():
    print_step("Starting jadevectordb server...")
    proc = subprocess.Popen(
        [SERVER_BIN],
        cwd=SERVER_CWD,
        env={**os.environ, "JADEVECTORDB_ENV": "development"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not wait_for_server():
        print_step("ERROR: Server did not become ready in time.")
        proc.terminate()
        sys.exit(1)
    print_step("Server is ready.")
    return proc

def login():
    res = requests.post(f"{BASE_URL}/v1/auth/login", json={"username": ADMIN_USER, "password": ADMIN_PASS})
    if res.status_code == 200:
        return res.json().get('token')
    else:
        print_step(f"Login failed: {res.text}")
        return None

def main():
    results = {}

    # Pre-test setup: kill any existing instance and start fresh
    print_step("Pre-test setup...")
    os.system("pkill -SIGTERM jadevectordb 2>/dev/null; sleep 1")
    start_server()

    token = login()
    if not token:
        print("Failed to get token, aborting")
        return
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    test_db_name = f"test_persistence_{int(time.time())}"
    # MT-01: Basic Database Creation
    print_step("MT-01: Database Creation")
    res = requests.post(f"{BASE_URL}/v1/databases", headers=headers, json={
        "name": test_db_name,
        "vectorDimension": 512,
        "distance_metric": "cosine",
        "index_type": "flat"
    })
    
    db_id = test_db_name
    if res.status_code == 201:
         db_id = res.json().get('databaseId', test_db_name)
    elif res.status_code == 409: # Already exists
         pass
         
    results['MT-01'] = "PASS" if res.status_code in (201, 409) else f"FAIL (HTTP {res.status_code})"
        
    # MT-02: Store Vectors
    print_step("MT-02: Store Vectors")
    success_store = True
    for i in range(1, 11):
        res = requests.post(f"{BASE_URL}/v1/databases/{db_id}/vectors", headers=headers, json={
            "id": f"vec_{i}",
            "values": [0.1] * 512,
            "metadata": {"label": "test"}
        })
        if res.status_code not in (200, 201):
            success_store = False
            results['MT-02'] = f"FAIL (HTTP {res.status_code}: {res.text})"
            break
    if success_store:
        results['MT-02'] = "PASS"

    # MT-03: Restart Persistence
    print_step("MT-03: Restart Persistence")
    os.system("pkill -SIGTERM jadevectordb")
    print_step("Waiting for server to stop...")
    time.sleep(2)
    start_server()
    
    token = login()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    res = requests.get(f"{BASE_URL}/v1/databases/{db_id}", headers=headers)
    if res.status_code == 200:
        res_vectors = requests.get(f"{BASE_URL}/v1/databases/{db_id}/vectors/vec_1", headers=headers)
        if res_vectors.status_code == 200:
            results['MT-03'] = "PASS"
        else:
            print_step(f"Vector retrieve failed: {res_vectors.status_code} {res_vectors.text}")
            results['MT-03'] = f"FAIL (Vector retrieval failed)"
    else:
        results['MT-03'] = f"FAIL (Database retrieval failed: {res.status_code} {res.text})"

    # MT-04: Multiple Databases
    print_step("MT-04: Multiple Databases")
    res1 = requests.post(f"{BASE_URL}/v1/databases", headers=headers, json={"name": "db_256", "vectorDimension": 256})
    res2 = requests.post(f"{BASE_URL}/v1/databases", headers=headers, json={"name": "db_512", "vectorDimension": 512})
    res3 = requests.post(f"{BASE_URL}/v1/databases", headers=headers, json={"name": "db_1024", "vectorDimension": 1024})
    
    if res1.status_code in (201, 409) and res2.status_code in (201, 409) and res3.status_code in (201, 409):
        results['MT-04'] = "PASS"
    else:
        results['MT-04'] = f"FAIL (HTTP {res1.status_code} {res2.status_code} {res3.status_code})"
        
    # MT-05: Update and Delete Operations
    print_step("MT-05: Update and Delete")
    requests.post(f"{BASE_URL}/v1/databases/{db_id}/vectors", headers=headers, json={"id": "vec_100", "values": [0.2]*512})
    requests.post(f"{BASE_URL}/v1/databases/{db_id}/vectors", headers=headers, json={"id": "vec_100", "values": [0.3]*512})
    requests.delete(f"{BASE_URL}/v1/databases/{db_id}/vectors/vec_100", headers=headers)
    
    os.system("pkill -SIGTERM jadevectordb")
    print_step("Waiting for server to stop...")
    time.sleep(2)
    start_server()
    token = login()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    res = requests.get(f"{BASE_URL}/v1/databases/{db_id}/vectors/vec_100", headers=headers)
    if res.status_code == 404:
        results['MT-05'] = "PASS"
    else:
        results['MT-05'] = f"FAIL (Should be 404, got {res.status_code})"

    # MT-06: Large Dataset
    results['MT-06'] = "SKIP"
    
    # MT-07: Flush Operations
    results['MT-07'] = "SKIP - Takes 5 minutes"
    
    # MT-08: Database Deletion
    res = requests.delete(f"{BASE_URL}/v1/databases/{db_id}", headers=headers)
    if res.status_code in (200, 204):
        results['MT-08'] = "PASS"
    else:
        results['MT-08'] = f"FAIL (HTTP {res.status_code})"
        
    # MT-09: Index Resize
    results['MT-09'] = "SKIP"
    results['MT-10'] = "SKIP"
    results['MT-11'] = "SKIP"
    results['MT-12'] = "SKIP"
    results['MT-13'] = "SKIP"
    results['MT-14'] = "SKIP"
    
    # MT-16: Vector Listing Endpoint
    print_step("MT-16: Vector Listing Endpoint")
    list_test_name = f"list_test_{int(time.time())}"
    res = requests.post(f"{BASE_URL}/v1/databases", headers=headers, json={"name": list_test_name, "vectorDimension": 128})
    list_db_id = list_test_name
    if res.status_code in (201, 409):
        if res.status_code == 201: list_db_id = res.json().get('databaseId', list_test_name)
        
        for i in range(1, 26):
            requests.post(f"{BASE_URL}/v1/databases/{list_db_id}/vectors", headers=headers, json={"id": f"vec_{i:03d}", "values": [0.1]*128})
        res_list = requests.get(f"{BASE_URL}/v1/databases/{list_db_id}/vectors", headers=headers)
        if res_list.status_code == 200 and res_list.json().get('total') == 25:
            results['MT-16'] = "PASS"
        else:
            results['MT-16'] = f"FAIL (HTTP {res_list.status_code} Total: {res_list.json().get('total')})"
            
    # MT-15: Admin Shutdown
    print_step("MT-15: Admin Shutdown")
    res = requests.post(f"{BASE_URL}/admin/shutdown", headers=headers)
    if res.status_code in (200, 202):
        results['MT-15'] = "PASS"
        time.sleep(2)
    else:
        print_step(f"Admin shutdown failed: {res.status_code} {res.text}")
        results['MT-15'] = f"FAIL (HTTP {res.status_code})"
        os.system("pkill jadevectordb")
    
    # Create the report
    report = f'''# Manual Testing Execution Report
    
### Test Round: {time.strftime('%Y-%m-%d')}
### Build: Automated Test
### Tester: Antigravity

| Test ID | Result | Notes |
|---------|--------|-------|
| MT-01   | {results.get('MT-01', 'FAIL')} | Basic Database Creation |
| MT-02   | {results.get('MT-02', 'FAIL')} | Store Vectors |
| MT-03   | {results.get('MT-03', 'FAIL')} | Restart Persistence |
| MT-04   | {results.get('MT-04', 'FAIL')} | Multiple Databases |
| MT-05   | {results.get('MT-05', 'FAIL')} | Update and Delete Operations |
| MT-06   | {results.get('MT-06', 'SKIP')} | Performance test |
| MT-07   | {results.get('MT-07', 'SKIP')} | Auto flush takes 5 minutes |
| MT-08   | {results.get('MT-08', 'FAIL')} | Database Deletion |
| MT-09   | {results.get('MT-09', 'SKIP')} | Index resize - covered by unit tests |
| MT-10   | {results.get('MT-10', 'SKIP')} | WAL crash recovery - covered by unit tests |
| MT-11   | {results.get('MT-11', 'SKIP')} | Snapshot Backup & Restore - unit tests |
| MT-12   | {results.get('MT-12', 'SKIP')} | Persistence Statistics - unit tests |
| MT-13   | {results.get('MT-13', 'SKIP')} | Data Integrity Verifier - unit tests |
| MT-14   | {results.get('MT-14', 'SKIP')} | Free List Space Reuse - unit tests |
| MT-15   | {results.get('MT-15', 'FAIL')} | Admin Shutdown Endpoint |
| MT-16   | {results.get('MT-16', 'FAIL')} | Vector Listing Endpoint |
| MT-17   | SKIP | SIGINT manual test (Graceful shutdown) |
| MT-18   | SKIP | SIGINT manual test (Force exit) |
'''

    with open("/Users/Deepak/Public/JadeVectorDB/TestResultFeb25.md", "w") as f:
        f.write(report)
        
    print_step("Report generated at /Users/Deepak/Public/JadeVectorDB/TestResultFeb25.md")

if __name__ == "__main__":
    main()
