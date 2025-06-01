#!/usr/bin/env bash
set -e
OUT=data/raw_txt
mkdir -p "$OUT"
for Y in {2010..2024}; do
  curl --globoff -L "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info?op=exp_request&ds=hmi.sharp_cea_720s[$Y.01.01_00:00:00_TAI-$Y.12.31_23:59:00_TAI]%{PATCH,MTOT,USFLUX,R_VALUE}" \
  -o "$OUT/sharp_${Y}.tar"
done
