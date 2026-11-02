#!/bin/bash
# pack.sh RESULTS ZIP : check against private-test.txt order, then zip as results.txt
RES=$(realpath $1); ZIP=$(realpath -m $2); T=$SRED_DATA/private_test/private-test.txt; export LC_ALL=C
n=$(awk 'NF>=2' $T | wc -l)
bad_order=$(paste -d' ' <(cut -d' ' -f1,2 "$RES") <(awk 'NF>=2{print $1, $2}' $T) | awk '$1!=$3||$2!=$4{b++} END{print b+0}')
bad_lines=$(awk 'NF!=3 || $3!=$3+0 || $3=="nan" || $3=="inf" || $3=="-inf" {b++} END{print b+0}' "$RES")
echo "$(basename $RES): lines $(wc -l < $RES)/$n mismatched $bad_order bad $bad_lines"
[ "$(wc -l < "$RES")" -eq "$n" ] && [ "$bad_order" -eq 0 ] && [ "$bad_lines" -eq 0 ] || { echo "abort: checks failed"; exit 1; }
tmp=$(mktemp -d); cp "$RES" "$tmp/results.txt"; rm -f "$ZIP"; (cd "$tmp" && zip -q -j "$ZIP" results.txt) || { echo "abort: zip failed"; exit 1; }; rm -r "$tmp"
echo "ready: $ZIP ($(sha256sum "$ZIP" | cut -c1-16))"
