# ns-3 scenario and 5G-LENA patches

| Component | Version |
|---|---|
| ns-3 | 3.48 (`https://www.nsnam.org/releases/ns-3.48.tar.bz2`, SHA-256 `5700ceecef2c9bc862502914b2237fe798d4c7ae07652247050e300e49407c4d`) |
| 5G-LENA (`contrib/nr`) | v5.1, commit `cedceadda17392c90587fb9400eb9b1f8c236713` (`https://gitlab.com/cttc-lena/nr.git`) |
| Eigen | 3.4.0, commit `3147391d946bb4b6c68edd901f2add6ac1f31f8c` (`https://gitlab.com/libeigen/eigen.git`) |

## Files

| File | Purpose |
|---|---|
| `ran-closed-loop-c2-production.cc` | The scenario every C2 production run was built from (SHA-256 `f661966d...`) |
| `ran-closed-loop.cc` | The same scenario with the RQ5 hook added (SHA-256 `74b15d90...`). Without `--rq5Socket` the hook is inactive, and a short check produced the same outputs as the production binary. The RQ5 runs and their references used this build |
| `run_c2.py` | Runs one simulation: copies the inputs into the run directory, records the build facts and hashes, and writes `manifest.json` last |
| `c2-default.json` | Default configuration (the C2 runs use `configs/ranbench/c2-base-config.json`) |
| `patches/lena-v5.1-d7-idle-connecting-ra.patch` | D-7: in RRC state IDLE_CONNECTING a second random-access success no longer aborts the run. The UE keeps waiting, bounded by T300 |
| `patches/lena-v5.1-d7b-stale-bsr.patch` | D-7b: a buffer status report for a logical channel whose context was removed is ignored instead of aborting the run |

Each patched branch prints one stderr line (`EXP2026003_PATCH_D7` or `EXP2026003_PATCH_D7B`) when it fires, so every run
reports how often a guard was used.

NOTICE: the files in `patches/` modify 5G-LENA, which is GPL-2.0-only, and are distributed under GPL-2.0-only
(`patches/LICENSE`, copied from the 5G-LENA v5.1 tree), not under the repository's MIT license. The two scenario
sources carry no code copied from ns-3 or 5G-LENA examples. They use the simulator APIs and stay under MIT.

## Build

```bash
curl -LO https://www.nsnam.org/releases/ns-3.48.tar.bz2 && tar xjf ns-3.48.tar.bz2
git clone https://gitlab.com/cttc-lena/nr.git ns-3.48/contrib/nr && git -C ns-3.48/contrib/nr checkout cedceadda17392c90587fb9400eb9b1f8c236713
git clone --branch 3.4.0 https://gitlab.com/libeigen/eigen.git
cmake -S eigen -B eigen-build -DCMAKE_INSTALL_PREFIX="$PWD/eigen-install" -DBUILD_TESTING=OFF && cmake --install eigen-build
# optional, for the patched rerun builds (D-7 alone, or D-7 and D-7b)
patch -d ns-3.48/contrib/nr -p1 < src/ranbench/ns3/patches/lena-v5.1-d7-idle-connecting-ra.patch
patch -d ns-3.48/contrib/nr -p1 < src/ranbench/ns3/patches/lena-v5.1-d7b-stale-bsr.patch
cp src/ranbench/ns3/ran-closed-loop-c2-production.cc ns-3.48/scratch/ran-closed-loop.cc
cd ns-3.48
CMAKE_PREFIX_PATH="$PWD/../eigen-install" ./ns3 configure --build-profile=optimized --enable-examples --disable-tests
./ns3 build ran-closed-loop                 # -> build/scratch/ns3.48-ran-closed-loop-optimized
```

The Apple M4 builds used the same commands with `--enable-eigen` added to `configure`.

For the RQ5 build, copy `ran-closed-loop.cc` to `ns-3.48/scratch/ran-closed-loop-rq5.cc` and build
`ran-closed-loop-rq5`. `run_c2.py` finds the build tree through the `NS3_ROOT` environment variable (default
`ns-3.48`). The optimized profile compiles with `-march=native`, so a binary belongs to one compiler and CPU type.

## Builds used by the paper

`experiments/c2-closed-loop/results/analysis/runs.csv` records the binary of every run. The first eight hex digits of
the SHA-256 identify the build:

| Platform | Unpatched | D-7 | D-7 and D-7b | RQ5 hook |
|---|---|---|---|---|
| cluster A (x86-64, gcc 14.2) | `7e2268c1` | `025d3dc4` | `74d32e46` | |
| cluster B (x86-64, gcc) | `bd1738b0` | `a4b41532` | | |
| L40 server (x86-64, gcc 13.3) | `a59ba72a` | | | |
| Apple M4 Max (arm64, clang) | `70abfed6` | | `32976335` | `4d46bc2e` |

The combined D-7 and D-7b patch (the two files applied in order) has SHA-256 `9dfac9e5...`.
