import numpy as np

from training.trainer import Trainer


trainer = Trainer(
    episodes=10000,
    max_steps=200
)

results = trainer.train()

np.save(
    "models/q_table.npy",
    results["q_table"]
)

np.save(
    "results/rewards.npy",
    results["rewards"]
)

np.save(
    "results/steps.npy",
    results["steps"]
)

np.save(
    "results/epsilon.npy",
    results["epsilon"]
)

print("\nTraining completed successfully.")
print("Q-table saved to: models/q_table.npy")
print("Training results saved to: results/")