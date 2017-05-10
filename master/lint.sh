#!/bin/sh

echo "ChkTeX"
echo "======"

ChkTex main.tex

echo ""
echo "lacheck"
echo "======="

lacheck main.tex
