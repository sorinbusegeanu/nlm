"""Evaluate fixed E31 checkpoint rosters on the same v0_group scenarios."""
from src.e31_roster_evaluation import evaluate


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true",
                        help="write available roster results while the 40 roster is pending")
    args = parser.parse_args()
    result = evaluate(allow_partial=args.allow_partial)
    print("\nMean balanced accuracy across the 50 identical scenarios:")
    for count in sorted(map(int, result["summary"])):
        row = result["summary"][str(count)]
        print(count, " | ".join(f"{method}: {row[method]['mean_balanced_accuracy']:.6f}"
                                for method in ("Independent", "E19-F-A", "E19-F-C", "E19-D")))
    if result["pending_checkpoint_counts"]:
        print("Pending roster counts:", result["pending_checkpoint_counts"])
