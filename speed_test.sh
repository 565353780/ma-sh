perf record -F 10000 -g python train.py
perf script | ../FlameGraph/stackcollapse-perf.pl >./output/out.folded
../FlameGraph/flamegraph.pl ./output/out.folded >./output/flamegraph.svg
