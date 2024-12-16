import argparse
from ma_sh.Demo.Convertor.mash import demo as demo_convert_mash


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is a sample program")
    parser.add_argument('device', default='cpu', type=str, help="Input file")
    args = parser.parse_args()

    demo_convert_mash(args.device)
