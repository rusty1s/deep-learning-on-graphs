# Superpixel

## Paper

### Neural Networks

* [SuperCNN: A Superpixelwise Convolutional Neural Network for Salient Object Detection](http://www.shengfenghe.com/uploads/1/5/1/3/15132160/supercnn_ijcv2015.pdf)
  * erkennt auffällige Objekte in einem Bild
  * Bild wird in verschiedene Superpixelrepräsentationen umgewandelt (Multi-Scale)
  * auf diesen Superpixeln werden Sequenzen für Farbkontraste und Farbverteilung erstellt (Graph Kernel)
  * Sequenzen sind die Eingaben für das CNN
* [Object Detection by Labeling Superpixels](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Yan_Object_Detection_by_2015_CVPR_paper.pdf)
* [Superpixel Convolutional Networks using Bilateral Inceptions](https://arxiv.org/pdf/1511.06739v5.pdf)
* [Convolutional Networks in Scene Labelling](http://cs231n.stanford.edu/reports/ashwinpp_final_report.pdf)
* [Feedforward semantic segmentation with zoom-out features](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Mostajabi_Feedforward_Semantic_Segmentation_2015_CVPR_paper.pdf)
* [Harmony Potentials for Joint Classification and Segmentation](http://www.cat.uab.es/~joost/papers/cvpr2010.pdf)
* [On Parameter Learning in CRF-based Approaches to Object Class Image Segmentation](https://www.vision.ee.ethz.ch/en/publications/papers/proceedings/eth_biwi_00742.pdf)
* [Learning hierarchical features for scene labeling](http://yann.lecun.com/exdb/publis/pdf/farabet-pami-13.pdf)
* [Recurrent convolutional neural networks for scene labeling](http://jmlr.org/proceedings/papers/v32/pinheiro14.pdf)
* [Deep Convolutional Neural Fields for Depth Estimation from a Single Image](https://arxiv.org/pdf/1411.6387v2.pdf) ([Github](https://github.com/slundqui/superpixelDepth))

### Representation

* [Superpixel lattices](https://pdfs.semanticscholar.org/1328/880541640d3c9aa1ce7b5201f90d6c4e0925.pdf)
* [Superpixel graph label transfer with learned distance metric](http://users.cecs.anu.edu.au/~sgould/papers/eccv14-spgraph.pdf)
* [PatchMatchGraph: Building a Graph of Dense Patch Correspondences for Label Transfer](https://www.vision.ee.ethz.ch/en/publications/papers/proceedings/eth_biwi_00742.pdf)

## Algorithms

* [SLIC superpixels compared to state-of-the-art superpixel 
  methods](https://infoscience.epfl.ch/record/177415/files/Superpixel_PAMI2011-2.pdf)
  * runtime efficient
  * **Implementations:**
    * [scikit-image: Image processing in Python](http://scikit-image.org/) ([Github](https://github.com/scikit-image/scikit-image))
    * OpenCV
* [Structured forests for fast edge detection](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/12/DollarICCV13edges.pdf)
  * low quantization error

## Examples

* [Segmentation: A SLIC Superpixel Tutorial using Python](http://www.pyimagesearch.com/2014/07/28/a-slic-superpixel-tutorial-using-python/)
* [Accessing Individual Superpixel Segmentations with Python](http://www.pyimagesearch.com/2014/12/29/accessing-individual-superpixel-segmentations-python/)
  * [OpenCV center of contour](http://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/)
