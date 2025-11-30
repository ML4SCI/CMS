import os
import tarfile
import requests
from urllib.parse import urlsplit


def filename_from_url(url: str) -> str:
    # Try Content-Disposition; fall back to URL path
    with requests.get(url, stream=True, timeout=15) as r:
        cd = r.headers.get('content-disposition', '')

        if 'filename=' in cd:
            return cd.split('filename=')[-1].strip(' ";')
        
    # Fallback to path segment without query
    return os.path.basename(urlsplit(url).path)


def download_jetclass_data(url: str, folder: str, timeout: int, chunk_size: int) -> str:
    os.makedirs(folder, exist_ok=True)
    fname = filename_from_url(url)
    dst = os.path.join(folder, fname)

    # Ensure continuous full download (no resume)
    if os.path.exists(dst):
        os.remove(dst)

    print(f"[start] {fname} -> {folder}")

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()

        with open(dst, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

    print(f"[done] {dst}")

    return dst


def extract_tar(file_path: str, folder: str, remove_tar: bool = True):
    print(f"[extract] {file_path} -> {folder}")

    with tarfile.open(file_path, 'r') as tar:
        tar.extractall(path=folder)

    print(f"[done] extracted {file_path}")

    # Delete the archive to save disk space
    if remove_tar:
        os.remove(file_path)
        print(f"[cleanup] removed {file_path}")