import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
