from importlib.metadata import version, PackageNotFoundError


LGATR_VERSION = '1.4.4'
INSTALL_COMMAND = 'python -m pip install --user "lgatr==1.4.4" uproot awkward tqdm vector'


def check_lgatr_version(required: str = LGATR_VERSION) -> str:
    """Fail fast, with the install command, unless the lgatr version the PAG models were built with is installed."""
    try:
        installed = version('lgatr')
    except PackageNotFoundError:
        installed = None

    if installed != required:
        raise RuntimeError(
            f"lgatr=={required} is required, found {installed or 'no lgatr'}. Install it with:\n"
            f"    {INSTALL_COMMAND}\n"
            f"or run: source jobs/setup_env.sh"
        )

    return installed
