# Superpixel

## Paper

### Neural Networks

* [SuperCNN: A Superpixelwise Convolutional Neural Network for Salient Object Detection](http://www.shengfenghe.com/uploads/1/5/1/3/15132160/supercnn_ijcv2015.pdf)
  * erkennt auffällige Objekte in einem Bild
  * Bild wird in verschiedene Superpixelrepräsentationen umgewandelt (Multi-Scale)
  * auf diesen Superpixeln werden Sequenzen für Farbkontraste und Farbverteilung erstellt (Graph Kernel)
  * Sequenzen sind die Eingaben für das CNN
* [Superpixel Convolutional Networks using Bilateral Inceptions](https://arxiv.org/pdf/1511.06739v5.pdf)
* [Feedforward semantic segmentation with zoom-out features](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Mostajabi_Feedforward_Semantic_Segmentation_2015_CVPR_paper.pdf)
* [Harmony Potentials for Joint Classification and Segmentation](http://www.cat.uab.es/~joost/papers/cvpr2010.pdf)
* [On Parameter Learning in CRF-based Approaches to Object Class Image Segmentation](https://www.vision.ee.ethz.ch/en/publications/papers/proceedings/eth_biwi_00742.pdf)
* [Learning hierarchical features for scene labeling](http://yann.lecun.com/exdb/publis/pdf/farabet-pami-13.pdf)
  * [Convolutional Networks in Scene Labelling](http://cs231n.stanford.edu/reports/ashwinpp_final_report.pdf)
* [Recurrent convolutional neural networks for scene labeling](http://jmlr.org/proceedings/papers/v32/pinheiro14.pdf)
* [Deep Convolutional Neural Fields for Depth Estimation from a Single Image](https://arxiv.org/pdf/1411.6387v2.pdf) ([Github](https://github.com/slundqui/superpixelDepth))

## Region adjacency graphes

* [Region connectivity graphs in
  Python](http://peekaboo-vision.blogspot.de/2011/08/region-connectivity-graphs-in-python.html)
* [Drawing Region Adjacency
  Graphs](https://vcansimplify.wordpress.com/2014/08/15/604/)

### Non Neural Networks

* [Object Detection by Labeling Superpixels](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Yan_Object_Detection_by_2015_CVPR_paper.pdf)
  * Objekterkennung auf Superpixeln
  * auf den erkannten Objekten kann dann eine Bildklassifizierung angewendet werden (z.B. CNN)
  * Objekterkennung über Minimierung der Kosten, dass benachbarte Superpixel zum gleichen Objekt gehören
* [Class Segmentation and Object Localization with Superpixel Neighborhoods](http://www.vision.cs.ucla.edu/papers/fulkersonVS09.pdf)
* [Superpixel lattices](https://pdfs.semanticscholar.org/1328/880541640d3c9aa1ce7b5201f90d6c4e0925.pdf)
* [Superpixel graph label transfer with learned distance metric](http://users.cecs.anu.edu.au/~sgould/papers/eccv14-spgraph.pdf)
* [PatchMatchGraph: Building a Graph of Dense Patch Correspondences for Label Transfer](https://www.vision.ee.ethz.ch/en/publications/papers/proceedings/eth_biwi_00742.pdf)
* Image labeling with a classical CRF model, constructed on the superpixels

## Comparisons

* [Superpixel Benchmark and Comparison](https://www.tu-chemnitz.de/etit/proaut/rsrc/neubert_protzel_superpixel.pdf)
* [Superpixel Algorithms: Overview and Comparison](http://davidstutz.de/superpixel-algorithms-overview-comparison/)
* [Normalized
  Cut](http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_ncut.html)

## Algorithms

* [SLIC superpixels compared to state-of-the-art superpixel
  methods](https://infoscience.epfl.ch/record/177415/files/Superpixel_PAMI2011-2.pdf)
  * runtime efficient
  * **Implementations:**
    * [scikit-image: Image processing in Python](http://scikit-image.org/) ([Github](https://github.com/scikit-image/scikit-image))
    * OpenCV >= 3.0.0
* Quick Shift in [Quick Shift and Kernel Methods for Mode 
  Seeking](http://vision.cs.ucla.edu/papers/vedaldiS08quick.pdf)
  * superpixels are not fixed in size or number (a complex image with many fine scale image structures may have many more superpixels than a simple one)
* [Felzenszwalb: Efficient Graph-Based Image 
  Segmentation](http://cs.brown.edu/~pff/papers/seg-ijcv.pdf)
  * Segmentierung über Graphstruktur, dessen Kanten die Ähnlichkeit zwischen zwei Knoten beschreiben
  * Segmentierungsalgorithmus ergibt Superpixel-Repräsentation
  * wird oft benutzt
* [SEEDS: Superpixels Extracted via Energy-Driven
  Sampling](http://www.mvdblive.org/seeds/)
* [Structured forests for fast edge 
  detection](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/12/DollarICCV13edges.pdf)
  * low quantization error
* [Selective Search for Object Recognition](https://ivi.fnwi.uva.nl/isis/publications/bibtexbrowser.php?key=UijlingsIJCV2013&bib=all.bib)
  * Hierarchische Segmentierung
* [Convolutional Networks in Scene 
  Labelling](http://cs231n.stanford.edu/reports/ashwinpp_final_report.pdf)

### Skimage

* [Segmentation Algorithms in
  scikits-image](http://peekaboo-vision.blogspot.de/2012/09/segmentation-algorithms-in-scikits-image.html)

## Tutorials

* [Segmentation: A SLIC Superpixel Tutorial using Python](http://www.pyimagesearch.com/2014/07/28/a-slic-superpixel-tutorial-using-python/)
* [Accessing Individual Superpixel Segmentations with Python](http://www.pyimagesearch.com/2014/12/29/accessing-individual-superpixel-segmentations-python/)
  * [OpenCV center of contour](http://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/)

## Formfeatures

* [A Parameter-Optimizing Model-Based Approch to the Analysis of Low-SNR Image
  Sequences for Biological Virus
  Detection](https://eldorado.tu-dortmund.de/handle/2003/35229)
  * Kapitel 6.2
* [Skimage Region
  Properties](http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops)
* [Merkmale von Bildregionen, Einfuehrung in 
  Spektraltechniken](http://www-home.fh-konstanz.de/~mfranz/ibv_files/lect09_spectr.pdf)
* [Design and FPGA Implementation of a Perimeter 
  Estimator](http://www.maa.org/sites/default/files/images/upload_library/applets/CirclesRedistrict/perimeter.doc)
