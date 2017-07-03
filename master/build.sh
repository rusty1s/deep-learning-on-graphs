#!/bin/sh

pdflatex main.tex
bibtex main
makeindex -s main.ist -o main.gls main.glo
pdflatex main.tex
pdflatex main.tex
