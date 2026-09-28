"""Plot one result CSV or compare a directory of compatible portfolio results."""

import argparse

from btlight.ml.metrics.portfolio import plot_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="One model CSV, or a directory of model CSVs to compare")
    parser.add_argument("--output", help="Figure path; defaults beside the input")
    args = parser.parse_args()
    print(f"Saved {plot_results(args.path, args.output)}")


if __name__ == "__main__":
    main()
