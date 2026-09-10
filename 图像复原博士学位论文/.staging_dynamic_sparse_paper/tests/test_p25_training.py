import unittest
from scripts.train_p25_restormer import StepBatches,choose


class TrainingTests(unittest.TestCase):
    def test_resume_sampling(self):
        whole=list(StepBatches(17,4,0,10))
        self.assertEqual(whole[4:],list(StepBatches(17,4,4,10)))
        keys=[key for batch in whole for key in batch]
        self.assertEqual(len(set(k[0] for k in keys[:17])),17)
        self.assertEqual(len(set(k[1] for k in keys)),40)

    def test_safe_choice(self):
        rows=[dict(passed=True,allocated_mib=4000,reserved_mib=5000,images_per_second=9.8),
              dict(passed=True,allocated_mib=8000,reserved_mib=9000,images_per_second=10.),
              dict(passed=True,allocated_mib=9000,reserved_mib=10000,images_per_second=20.)]
        self.assertEqual(choose(rows),rows[0])
