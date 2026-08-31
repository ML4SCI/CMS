"""
prepare_cifar100.py  --  fetch and verify CIFAR-100 on a LOGIN node.

Perlmutter compute nodes have no outbound network, so the download has to
happen here, once, into $SCRATCH.

    module load pytorch/2.6.0
    source /pscratch/sd/k/krish_m/venvs/depthvit/bin/activate
    python3 scripts/prepare_cifar100.py --root /pscratch/sd/k/krish_m/datasets/cifar100
"""

import argparse
import os

from torchvision.datasets import CIFAR100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.root, exist_ok=True)

    train = CIFAR100(root=args.root, train=True, download=True)
    val = CIFAR100(root=args.root, train=False, download=True)

    n_train, n_val, n_classes = len(train), len(val), len(train.classes)
    print(f"root:     {args.root}")
    print(f"train:    {n_train}")
    print(f"val:      {n_val}")
    print(f"classes:  {n_classes}")

    img, label = train[0]
    print(f"sample:   size={img.size} mode={img.mode} label={label}")

    ok = (n_train == 50_000 and n_val == 10_000 and n_classes == 100
          and img.size == (32, 32) and img.mode == "RGB")
    print()
    print("VERIFY:   " + ("PASS" if ok else "FAIL"))
    if not ok:
        raise SystemExit(1)
    print()
    print("Set these in the config:")
    print(f'  data.root            = "{args.root}"')
    print( '  data.n_channels      = 3')
    print( '  data.num_classes     = 100')
    print( '  data.dataset_size    = 50000')


if __name__ == "__main__":
    main()
