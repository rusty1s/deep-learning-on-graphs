#!/bin/sh

pdflatex -draftmode main
biber main
makeglossaries main
pdflatex -draftmode main
pdflatex main
