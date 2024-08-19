import os
import shutil
import argparse

def remove_cache_dirs(root_dir, is_remove=False):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'cache' in dirnames:
            cache_dir = os.path.join(dirpath, 'cache')

            if is_remove:
                print(f"Removing directory: {cache_dir}")
                shutil.rmtree(cache_dir)
            elif not is_remove:
                print(f"Cache directory: {cache_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove all 'cache' directories recursively.")
    parser.add_argument('--target',required=True, type=str, help='The root directory to start the search from.')
    parser.add_argument("is_remove",action="store_true",help="本当に消すかどうか. フラグを立てないと, cacheディレクトリ名が表示されるだけ")
    args = parser.parse_args()

    remove_cache_dirs(args.target,args.is_remove)