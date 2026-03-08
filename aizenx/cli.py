import argparse


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--info", action="store_true")

    args = parser.parse_args()

    if args.info:

        print("AizenX Explainable AI Toolkit")


if __name__ == "__main__":
    main()