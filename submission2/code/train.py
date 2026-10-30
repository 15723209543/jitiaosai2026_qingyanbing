"""New submission2: official-train full-pool joint head tuning with margin AUC loss."""
import sys
from joint_train import parse_args, run

if __name__ == "__main__":
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "full"])
    if "--method" not in sys.argv:
        sys.argv.extend(["--method", "marginauc"])
    run(parse_args())
