#!/bin/bash

pdflatex main.tex
bibtex main
makeglossaries main
pdflatex main.tex
