---
layout:     post
title:      Object Detection
subtitle:   A review of object detection
date:       2017-11-24
author:     Xiya Lv
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - Machine Learning
    - Deep Learning
    - Object Detection
---

# Abstract

​	Object detection as a core role of computer vision, ideals with detecting instances of semantic objects of a certain class in digital images and videos, involving not only recognizing and classifying every object, but localizing each one by drawing the appropriate bounding box around it. With the development of deep Learning, which has recently shown outstanding performance  on many fields, many successful approaches to object detection are proposed, making the object detection systems faster and more accurate.
	In this paper, I review some basic knowledges of object detection, such as the datasets using by most of the papers,  the criteria for determing performance, non-maximal suppression for fixing multiple detections. And review the main successful methods in the recent years: R-CNN, Fast R-CNN, Faster R-CNN, R-FCN, YOLO and SSD.  By the end of this review, we will hopefully have gained an understanding of how deep learning is applied to object detection, and how these object detection models inspire and diverge from others.

**Keywords:** Object Detection, Deep Learning, CNN, SSD, YOLO

## Introduction

		Humans glance at an image and instantly know what the objects are, where they are and how they interact. Many researchers are committed to make computers to do the same things as human, which leads to many subdomains of computer vision, such as object recognition to konw what the object are, object detection to know what they are and where they are, semantic segmentation to know both above and to know how the images interact, and so on. As we move towrads more complete image undestanding, having more precise and detailed object recognition becomes crucial. Detecing instances of semantic objects of a certain class in digital images, asks us not only to classify images, but also precisely estimatine the classes and locations of the objects, a problem known as objetc detection.
		The last decade of progress on object detection has been based on the use of SIFT and HOG, which is considered as a traditional method. But it is generally acknowledged that the gains of performance has been small during 2010-2012, on the canonical visual recognition task, PASCAL VOC object detection. Most advance is obtained by building ensemble system and emplying minor fariants of successful methods. Before the advent of convolutional nerual network, the state of the art for those two approaches - Deformable Part Model(DOM) and Selective Search, had comparable performance. In recent years, with the development of deep learning, object detection develop rapidly since convolution neural networks(CNNs) are  heavy used to get big processes. Especially R-CNN are proposaled in 2014 achieving a dramaic improvement, which combines selective search region proposals and convolution neural network based post-claddification. From then on,  a series of improvement measures are proposed to get better preformances and faster speed on object detection. 
		
		Fast R-CNN trains networks using a multi-task loss in a single training stage based on the original R-CNN. Faster R-CNN replace the slow selective search algorithm with a fast neural net based on Fast R-CNN. R-FCN shares 100\% of the computations across every single output based on Faster R-CNN, getting a faster speed. YOLO, a new approach to object detection presented in CVPR 2016, framing object detection as a regression problem to spatially separated bounding boxes and associated class probabilities, is very different form the family of R-CNN. SSD combines the core ideas of Faster R-CNN and YOLO, as one-stage method, getting a faster speed than YOLO and almost the same accuracy as Faster R-CNN. All above algorithms are introduced in detail in chapter 3. 
	
		Current state-of-the-art object detection systems are variants of two strategies: one-stage method and two-stage method. The representatives of two-stage method are R-CNN family, SPPnet et al. The approaches use region proposal methods to first generate potential bounding boxes in an image and then run a high-quality classifier or a convolutional neural network on those proposed boxes.  The first step generating potential bounding boxes is generally implemented in two ways, one based on sliding windows and the other based on region proposal network(RPN). Selective search as a traditional method is fading away since RPN proposaled in Faster R-CNN, but still an useful method. While accurate, these approaches have been too computationally intensive for embedded systems and, even with high-end hardware, too slow for real-time applications. YOLO and SSD speed up the progress of object detection significantly, known as the one-stage method. YOLO make up for this shortcoming using a single nerual network to predict  bounding boxes and class probabilities directly from full images in one evaluation, while SSD uses different scales from feature maps of different scales and considers multiple aspect ratios .
## Basic knowledge

		In this chapeter, I introduce some basic knowledge of object detection in brief, which appearing in many papers for convenience of explanation, including data set, intersection-over-union(IoU), mean average precision(mAP) and non-maximum suppression(NMS). 
		
		**DataSet:** Most of papers train and test their models based on PASCAL VOC challenge. The PASCAL VOC project provides standardised image data sets for object class recognition. Organised annually from 2005 to 2012, the PASCAL VOC challenge and its associated dataset has become accepted as the benchmark for object detection. VOC 2007 having 9,963 images, containing 24,640 annotated objects is the most popular data set. It has 20 classes: person, bird, cat, cow, dog, horse, sheep, aeroplane, bicycle, boat, bus, car, motorbike, train, bottle, chair, dining table, potted plant, sofa, tv/monitor.  The later dataset 2010 and 2012 are similar with 2007. In recent years, the dataset COCO is also used to train detectors.COCO is a large-scale object detection, segmentation, and captioning dataset, having several features described as the website: Object segmentation, Recognition in context, Superpixel stuff segmentation, 330K images (>200K labeled), 1.5 million object instances, 80 object categories, 91 stuff categories, 5 captions per image and 250,000 people with keypoints.
		
		**IoU** Intersection over Union is an evaluation metric used to measure the accuracy of an object detector on a particular dataset.We can typically find it used to evaluate the performance of object detectors, such as R-CNN, R-FCN, YOLO, etc. More formally, in order to use the IoU, we need to know two sets of values. One is the ground-truth bounding boxes, and the other is the predicted bounding boxes outputed from an objetc detector.  A simple IoU is simply a ratio, can be determined as figure 1.In the numerator we compute the area of overlap between the predicted bounding box and the ground-truth bounding box. The denominator is the area of union, or more simply, the area encompassed by both the predicted bounding box and the ground-truth bounding box. Dividing the area of overlap by the area of union yields gets the final score -- IoU.
		\begin{figure}[H]
			\centering			
			\includegraphics[width=100pt]{/Users/lxy/Desktop/ReviewofObjectDetection/picture/IoU.jpg}
			\caption{Computing the IoU is as simple as dividing the area of overlap between the bounding boxes by the area of union.}
			\label{fig:myphoto}
		\end{figure}
		**mAP**: Mean average precision is a criterion for judging the performance of a objetc detector. We have to understand average precision before introducing mean average precision. Precision and recall are single-value metrics based on the whole list of documents returned by the system. For systems that return a ranked sequence of documents, it is desirable to also consider the order in which the returned documents are presented. By computing a precision and recall at every position in the ranked sequence of documents, one can plot a precision-recall curve, plotting precision $ p(\gamma) $ over the interval from $ \gamma = 0 $ to $ \gamma = 1 $:
		\[ AP = \int_{0}^{1} p(\gamma) d\gamma \]
		That is the area under the precision-recall curve. Normally, to reduce the impact of "wiggles" in the curve in the PASCAL Visual Object Classes challenge, researchers computes average precision by averaging the precision over a set of evenly spaced recall levels {0, 0.1, 0.2, ... 1.0}:
		\[ AP = \frac{1}{11} \sum_{\gamma \in \{0,0.1,...,1.0\}} p_{interp}(\gamma) \]
		where $ p_{interp}(\gamma) $ is an interpolated precision that takes the maximum precision over all recalls greater than $ \gamma $: $ p_{interp}(\gamma) = max_{\tilde{\gamma}:\tilde{\gamma}\geq\gamma}p(\tilde{\gamma}) $.
	
		Mean average precision for a set of classes is the mean of the average precision scores for each class.
		 \[ mAP = \frac{\sum_{q=1}^{Q} AP(q)}{Q} \]
		where Q is the number of queries.
		
		**NMS**: The application of non-maximum supression in the field of object detection is very wide. The main purpose is to remove the redundant frames and, produce the final and hope the best location of detections, since no matter what object detection method you choose, you will detect multiple bounding boxes surrounding the object in the image. NMS finds the bounding box of highest confidence, according to the score matrixes and the coordinate information of regions. The main steps are following: First of all, NMS calculate the class-specific confidence score of each bounding box, sort them accroading to the scores, and then take the one with highest score as a candidate. Go through the rest of the boxes, and if the IoU with the candidate is greater than a certain threshold, delete it. Selecting a candidate with the highest score from the untreated boxes, Repeat the above steps until there is no other box.
		\begin{figure}[H]
			\centering			
			\includegraphics[width=100pt]{/Users/lxy/Desktop/ReviewofObjectDetection/picture/nms.jpg}
			\includegraphics[width=100pt]{/Users/lxy/Desktop/ReviewofObjectDetection/picture/nms-1.jpg}
			\includegraphics[width=100pt]{/Users/lxy/Desktop/ReviewofObjectDetection/picture/nms-12.jpg}
			\includegraphics[width=100pt]{/Users/lxy/Desktop/ReviewofObjectDetection/picture/nms-final.jpg}
			\caption{NMS removes the redundant frames and produces the final location of detections.}
			\label{fig:myphoto}
		\end{figure}

## Approaches

​	As we all know, the evolution of object detection algorithm is divided into two stages, one is based on the traditional features, the other is based on deep learning. Traditional feature-based optimization methods are still the mainstream until 2013. However, after 2013, the whole academia and industry are increasingly using the deep learning, because compared with traditional method, the deep learning improve the performance a lot. It is worth noting that the traditional detection method will become saturated with the increase of data volume, which means the detection performance will gradually increase as the data volume increases, but to a certain extent, the perforance gains little with the data volume increasing. However, the methods of deep learning are different. When the data conforming to the actual scene distribution is more and more, the detection performance will be better and better.

### R-CNN

The R-CNN, or regions with proposals with CNNs, was presented by Ross Girshick, etc in CVPR 2014. 
		\begin{figure}[H]
			\centering			
			\includegraphics[width=200pt]{/Users/lxy/Desktop/ReviewofObjectDetection/picture/rcnn.jpg}
			\caption{Object detection system overview. The system (1) takes an input image, (2) extracts around 2000 bottom-up region proposals, (3) computes features for each proposal using a large convolutional neural network (CNN), and then (4) classifies each region using class-specific linear SVMs.}
			\label{fig:myphoto}
		\end{figure}
		The object detection system consists of three modules. The first generates category-independent region proposals using selective search. These proposals define the set of candidate detections available to the detector. The second module is a large convolutional nerual network that extracts a fixed-length feature vector from each region. The third module is a set of class-specific linear SVMs.The three modules can be also summarized in three steps illustrated by figure 4: (1) region proposals generate. Scan the input image for possible objects using an algorithm called Selective Search, generating ~2000 region proposals. (2)feature extraction. Run a convolutional neural net (CNN) on top of each of these region proposals. (3)object classification and localization detection. Take the output of each CNN and feed it into a) an SVM to classify the region and b) a linear regressor to tighten the bounding box of the object, if such an object exists.
		\begin{figure}[H]
			\centering			
			\includegraphics[width=120pt]{/Users/lxy/Desktop/ReviewofObjectDetection/picture/rcnn-2.jpg}
			\caption{An example of R-CNN detector.}
			\label{fig:myphoto}
		\end{figure}
		In the first step, each region proposal's size is different, while the CNN requires fixed size(227 x 227 pixel size) images.We must first convert the image data in thar region into a form that is compatible with the CNN. Of the many possible transformations, the authors opt for the simplest, which regardless of the size or aspect ratio of the candidate region, warp all pixels in a tight bounding box around it to the required size.
		

		The CNN architecture is Alexnet.  To adapt the CNN to the new task and the new domain, the paper continues stochastic gradient descent training of the parameters using only warped region proposals form VOC, with treating all region proposals with $ \geq 0.5 $ IoU overlap with a ground-truth box as positices for the box's class adn the rest as negatives. The learning rate of stochastic gradient desecnt is 0.001, which allows fine-tuning to make progress while not clobbering the initialization.
		Once features are extracted and training labels are applied, optiminzing one linear SVM per class.
		
		The R-CNN achieves a performance of 53.3\%mAP on VOC 2012 test, which is proves mAP by more than 30\% relative to the best result at that time. The speed is 13s/image on a GPU or 53s/image on a CPU.
		
		 As a groundwork of the later researches, R-CNN has excellent object detection accuracy. However, it also has notable drawbacks.(1) Training is a multi-stage pipelines. one first time fine-tunes a ConvNet for detection using cross-entropy loss. Then linear SVMs are fit to ConvNet features computed on warped object proposals. In the third training stage, bouding-box regressors are learned. (2) Traing is expensive in space and time. For SVM and regressor training, features are extracted from each warped object proposal in each image and written to disk. These features require hundreds of gigabytes of storage. (3) Test-time detection is slow. At test-time, features are extract from each warped object proposal in each test image. R-CNN is slow because it warps and then processes each object proposal indepently. There is much repetitive work, the selective search, convolutional neural network for per region proposal progress. Even for one region, we must classify it by using 21 SVMs. Fortunately, the later work improves every aspect to get a faster speed and a more accurate performance.










