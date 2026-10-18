"""Run the complete pipeline: data -> models -> optimisation (about 3-5 minutes on 1 CPU)."""
import time

import generate_data
import optimize
import train_models

if __name__ == "__main__":
    t0 = time.time()
    print("=== Step 1/3: generate simulated dataset ===")
    generate_data.main()
    print("=== Step 2/3: train and evaluate models ===")
    train_models.main()
    print("=== Step 3/3: optimisation modules ===")
    optimize.main()
    print(f"\nPipeline finished in {time.time() - t0:.0f} s")
