#!/bin/sh

pdflatex main
biber main
makeglossaries main
pdflatex main
