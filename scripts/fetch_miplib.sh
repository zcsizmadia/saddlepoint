#!/usr/bin/env sh
# Downloads the benchmark instances from MIPLIB 2017 into data/miplib/ and
# decompresses them. saddlepoint solves their LP relaxations (the MPS reader
# ignores integrality markers). Files already present are kept.
#
#     sh scripts/fetch_miplib.sh [DIR]
set -eu
dir=${1:-data/miplib}
mkdir -p "$dir"
for name in rail507 irish-electricity supportcase10 ex10 physiciansched3-3 s100 savsched1 neos-5052403-cygnet square41; do
    if [ -f "$dir/$name.mps" ]; then continue; fi
    echo "fetching $name"
    curl -fsSL -o "$dir/$name.mps.gz" "https://miplib.zib.de/WebData/instances/$name.mps.gz"
    gunzip -f "$dir/$name.mps.gz"
done
