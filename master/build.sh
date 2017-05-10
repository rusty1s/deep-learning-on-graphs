#!/bin/sh

bibtex main
makeglossaries main
pdflatex main.tex
