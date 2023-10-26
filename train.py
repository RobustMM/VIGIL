import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=str,
        default="0"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--dataset",
        type=str
    )
    parser.add_argument(
        "--source_domain",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--target_domain",
        type=str
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./data/"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
