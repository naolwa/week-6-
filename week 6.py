import random
import statistics
import time
from typing import Callable, List, Tuple
import tensorflow as TFlite
class SmartIntrusionDetector:
    """
    Simple intrusion detector that 'learns' normal sensor readings (numeric),
    then flags anomalous readings (potential intruders) using a z-score threshold.

    Learning input: list of numeric samples representing normal activity (e.g., motion counts).
    Detection output: (intruder: bool, score: float) where score is the z-score distance.
    """

    def __init__(self, sensitivity: float = 3.0, epsilon: float = 1e-4):
        # sensitivity: z-score threshold above which we consider an anomaly
        self.sensitivity = float(sensitivity)
        # learned distribution parameters
        self._mean = None
        self._std = None
        self._epsilon = float(epsilon)

    def learn_normal(self, samples: List[float]) -> None:
        """Learn mean/std from given normal-operation samples."""
        if not samples:
            raise ValueError("samples must be a non-empty list of numeric values")
        self._mean = statistics.mean(samples)
        # use population stdev; fallback to small epsilon if zero
        std = statistics.pstdev(samples)
        self._std = std if std > 0 else self._epsilon

    def detect(self, reading: float) -> Tuple[bool, float]:
        """
        Return (intruder_flag, z_score).
        intruder_flag is True if |reading - mean| / std > sensitivity.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError("Detector has not been trained. Call learn_normal(samples) first.")
        z = abs(reading - self._mean) / self._std
        intruder = z > self.sensitivity
        return intruder, z

    def monitor(self,
                get_reading: Callable[[], float],
                interval_seconds: float = 1.0,
                iterations: int = 30,
                alert_callback: Callable[[float, float], None] = None) -> None:
        """
        Run a simple monitoring loop that reads values from get_reading(),
        performs detection, and optionally calls alert_callback(reading, z).
        """
        for i in range(iterations):
            reading = get_reading()
            try:
                intruder, z = self.detect(reading)
            except RuntimeError as e:
                print("Error:", e)
                return

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            if intruder:
                msg = f"[{timestamp}] ALERT: Intruder detected! reading={reading:.2f} z={z:.2f}"
                print(msg)
                if alert_callback:
                    try:
                        alert_callback(reading, z)
                    except Exception as cb_err:
                        print("Alert callback error:", cb_err)
            else:
                print(f"[{timestamp}] OK: reading={reading:.2f} z={z:.2f}")

            time.sleep(interval_seconds)


# ----------------------
# Example usage / simulation
# ----------------------
if __name__ == "__main__":
    # Simulate training data: normal motion counts (e.g., number of motion sensor triggers per minute)
    normal_samples = [random.gauss(5, 1) for _ in range(100)]  # mean ~5, std ~1

    detector = SmartIntrusionDetector(sensitivity=3.0)
    detector.learn_normal(normal_samples)
    print(f"Learned mean={detector._mean:.2f}, std={detector._std:.2f}")

    # Simulated live-read function: usually normal, occasionally large spike representing an intruder
    def simulated_sensor():
        # 95% of the time normal; 5% chance of intruder spike
        if random.random() < 0.05:
            return random.uniform(12, 20)  # intruder spike
        return random.gauss(5, 1)

    # Optional alert callback (e.g., send notification or save event)
    def alert_callback(reading, z):
        # For demo, just print that we'd notify:
        print(f"--> (callback) notify: reading={reading:.2f}, z={z:.2f}")

    # Run monitoring for 30 iterations, 0.5s between samples
    detector.monitor(get_reading=simulated_sensor, interval_seconds=0.5, iterations=30, alert_callback=alert_callback)