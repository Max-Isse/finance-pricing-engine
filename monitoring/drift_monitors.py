import numpy as np
import scipy.stats as stats
import redis  # <--- Import our cache client
import os

def calculate_psi(baseline, target, num_bins=10):
    baseline_counts, bins = np.histogram(baseline, bins=num_bins)
    target_counts, _ = np.histogram(target, bins=bins)
    
    baseline_probs = np.where(baseline_counts == 0, 0.0001, baseline_counts) / len(baseline)
    target_probs = np.where(target_counts == 0, 0.0001, target_counts) / len(target)
    
    psi_value = np.sum((target_probs - baseline_probs) * np.log(target_probs / baseline_probs))
    return psi_value

if __name__ == "__main__":
    np.random.seed(42)
    historical_yields = np.random.normal(loc=0.041, scale=0.002, size=1000)
    live_yields_drifted = np.random.normal(loc=0.049, scale=0.002, size=1000)
    
    score = calculate_psi(historical_yields, live_yields_drifted)
    print(f"DRIFT_CALCULATION_SCORE: {score:.4f}")
    
    # --- NEW: Pipe metric straight into our live Redis container ---
    try:
        # Use the service name defined in your docker-compose file as the host
        r = redis.Redis(host='redis-cache', port=6379, db=0, socket_timeout=5)
        r.set('latest_drift_score', f"{score:.4f}")
        r.set('pipeline_status', 'HEALTHY')
        print("[REDIS LOG] Successfully updated metrics cache layer.")
    except Exception as e:
        print(f"[REDIS ERROR] Failed to connect to caching layer: {e}")