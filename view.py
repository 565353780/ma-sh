import argparse

from ma_sh.Demo.view import demo_view_mash

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="view mash")

    parser.add_argument("mash_file_path", help="mash file path")
    args = parser.parse_args()

    demo_view_mash(args.mash_file_path)
