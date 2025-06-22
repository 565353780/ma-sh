import os
import argparse

from ma_sh.Demo.view import demo_view_mash, demo_view_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="view mash")

    parser.add_argument("mash_data_path", help="mash data path")
    args = parser.parse_args()

    if os.path.isdir(args.mash_data_path):
        demo_view_training(args.mash_data_path, 1)
    elif args.mash_data_path.endswith(".npy"):
        demo_view_mash(args.mash_data_path)
    else:
        print("[ERROR][view::__main__]")
        print("\t mash data path not valid!")
        print("\t mash_data_path:", args.mash_data_path)
