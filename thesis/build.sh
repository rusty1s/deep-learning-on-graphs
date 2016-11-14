#!/bin/bash

pdflatex main.tex
makeglossaries main
pdflatex main.tex
cp main.pdf main2.pdf
mv main2.pdf "$HOME/shared/main.pdf"
