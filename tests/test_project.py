import unittest

from main import (
    MachineReading,
    evaluate_model,
    extract_features,
    health_score,
    maintenance_risk,
    train_centroid_model,
)


class MaintenanceTest(unittest.TestCase):
    def test_feature_engineering_and_health_score(self):
        reading = MachineReading(92, 7.5, 3.0, 1600, 1)

        self.assertIn("thermal_load", extract_features(reading))
        self.assertLess(health_score(reading), 55)
        self.assertEqual(maintenance_risk(reading), "high")

    def test_training_and_evaluation(self):
        readings = [
            MachineReading(60, 1.0, 2.1, 200, 0),
            MachineReading(64, 1.4, 2.2, 400, 0),
            MachineReading(95, 8.0, 3.2, 1800, 1),
            MachineReading(90, 7.0, 3.0, 1500, 1),
        ]

        model = train_centroid_model(readings)
        evaluation = evaluate_model(readings)

        self.assertIn("healthy_center", model)
        self.assertIn("confusion_matrix", evaluation)
        self.assertGreaterEqual(evaluation["accuracy"], 0)


if __name__ == "__main__":
    unittest.main()
