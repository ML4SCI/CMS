import os
import argparse

from src.utils.data import download_jetclass_data, extract_tar

DATA_DIR = 'data'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the JetClass dataset")

    parser.add_argument('--timeout', type=int, default=7200, help="Timeout for downloading files (in seconds)")
    parser.add_argument('--chunk-size', type=int, default=1024 * 1024 * 1024, help="Chunk size for downloading files (in bytes)")
    parser.add_argument('--remove-tar', action='store_true', help="Remove .tar files after extraction")

    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_args()

    # Links to download the JetClass dataset
    train_data_urls = [
        'https://zenodo.org/records/6619768/files/JetClass_Pythia_train_100M_part0.tar?download=1',
        'https://zenodo.org/records/6619768/files/JetClass_Pythia_train_100M_part1.tar?download=1',
        'https://zenodo.org/records/6619768/files/JetClass_Pythia_train_100M_part2.tar?download=1',
        'https://zenodo.org/records/6619768/files/JetClass_Pythia_train_100M_part3.tar?download=1',
        'https://zenodo.org/records/6619768/files/JetClass_Pythia_train_100M_part4.tar?download=1',
        'https://zenodo.org/records/6619768/files/JetClass_Pythia_train_100M_part5.tar?download=1',
        'https://zenodo.org/records/6619768/files/JetClass_Pythia_train_100M_part6.tar?download=1',
        'https://zenodo.org/records/6619768/files/JetClass_Pythia_train_100M_part7.tar?download=1',
        'https://zenodo.org/records/6619768/files/JetClass_Pythia_train_100M_part8.tar?download=1',
        'https://zenodo.org/records/6619768/files/JetClass_Pythia_train_100M_part9.tar?download=1'
    ]  # 15.2 GB each
    val_data_url = 'https://zenodo.org/records/6619768/files/JetClass_Pythia_val_5M.tar?download=1'  # 7.6 GB
    test_data_url = 'https://zenodo.org/records/6619768/files/JetClass_Pythia_test_20M.tar?download=1'  # 30.4 GB

    # Create the data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    # Create the train directory for partial downloads
    train_data_dir = os.path.join(DATA_DIR, 'train_100M')
    os.makedirs(train_data_dir, exist_ok=True)

    # Download and extract the training data
    for url in train_data_urls:
        tar_path = download_jetclass_data(url, train_data_dir, args.timeout,args.chunk_size)
        extract_tar(tar_path, train_data_dir, remove_tar=args.remove_tar)

    # Download the validation and test data
    val_tar_path = download_jetclass_data(val_data_url, DATA_DIR, args.timeout, args.chunk_size)
    test_tar_path = download_jetclass_data(test_data_url, DATA_DIR, args.timeout, args.chunk_size)

    # Extract the validation and test data
    extract_tar(val_tar_path, DATA_DIR, remove_tar=args.remove_tar)
    extract_tar(test_tar_path, DATA_DIR, remove_tar=args.remove_tar)


if __name__ == '__main__':
    main()