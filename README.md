# MMLL-demo

Demonstrations of the [Musketeer Machine Learning Library (MMLL)](https://github.com/Musketeer-H2020/MMLL) using [Musketeer's pycloudmessenger](https://github.com/IBM/pycloudmessenger/)

## Installation

You may install pycloudmessenger and other library dependencies as follows:

`pip install -r requirements.txt`

Or if you only require pycloudmessenger, then:

`pip install http://github.com/IBM/pycloudmessenger/archive/v0.7.0.tar.gz`

The pycloudmessenger package requires a credentials file to access the cloud service. Please, place the `musketeer.json` credentials at the `demos/demo_pycloudmessenger/` folder.

## Content (available demos):

### POM1:

* **Kmeans**: Clustering demo on the PIMA dataset.
* **Neural Networks**: Multiclass Classification demo on the MNIST dataset.
* **Suport Vector Machines**: Binary classification demo on the MNIST dataset.
* **Preprocessing**: Different preprocessing demos.

### POM2:

* **Kmeans**: Clustering demo on the PIMA dataset.
* **Neural Networks**: Multiclass Classification demo on the MNIST dataset.
* **Suport Vector Machines**: Binary classification demo on the MNIST dataset.
* **Preprocessing**: Different preprocessing demos.

### POM3:

* **Kmeans**: Clustering demo on the PIMA dataset.
* **Neural Networks**: Multiclass Classification demo on the MNIST dataset.
* **Suport Vector Machines**: Binary classification demo on the MNIST dataset.
* **Preprocessing**: Different preprocessing demos.

### POM4: 

* **LR**: Linear Regression demo on the redwine dataset.
* **Kmeans_pm**: Clustering demo on a synthetic dataset.
* **Preprocessing**: Different preprocessing demos.

### POM5: 

* **LR**: Linear Regression demo on the redwine dataset.
* **Kmeans_pm**: Clustering demo on a synthetic dataset.
* **Preprocessing**: Different preprocessing demos.

### POM6: 

* **RR**: Ridge Regression demo on the redwine dataset.
* **Kmeans_pm**: Clustering demo on a synthetic dataset (public model).
* **LC_pm**: Binary Logistic Classifier: Binary Classification demo on the MNIST dataset (public model). Includes **normalization** example.
* **MLC_pm**: Multiclass Logistic Classifier: Multiclass Classification demo on the MNIST dataset (public model).
* **KR_pm**: Kernel Regression demo on a synthetic dataset.
* **Preprocessing**: Different preprocessing demos.

### Preprocessing:
* **Check data**: Check that the provided data at workers is in accordance with the data type description.
* **Conversion to numeric**: Conversion of categorical features to numeric.
* **Data alignment estimation**: Obtain the degree of alignment of each worker with respect to a reference dataset at MasterNode using different metrics.
* **Ad-hoc preprocessing**: Preprocess data at workers using logarithmic scale transformation and outlier clipping.
* **Image reshape**: Preprocess data at workers changing the height and width of images and vectorization.
* **Normalization**: Normalize data at workers to either have zero mean and stardard deviation one, or to be within zero and one.
* **Principal component analysis**: Dimensionality reduction at workers using PCA algorithm.
* **Greedy feature selection**: Greedy feature selection at workers using a linear model.
* **Frequency feature selection**: Remove some features at workers based on frequency of appearance.
* **Random projection**: Sparse random projection to a lower dimensional space.
* **Deep learning**: Transformation of any image into a vector of size 1000 using AlexNet.
* **Natural language processing**: NLP transformation using TF-IDF.
* **Record linkage**: Linkage of samples for vertical partitioning datasets.
* **Missing data imputation**: Missing data imputation for vertical partitioning datasets.
* **Data value estimation (a priori)**: Data value estimation for every worker using different metrics.
* **Data value estimation (a posteriori)**: Data value estimation for every worker using AUC with a greedy incremental approach.


The output files are stored in the corresponding `results/` folder.

## Usage

Please visit every subfolder in `demo/demo_pycloudmessenger/` for a detailed explanation about how to run the demos.

If you encounter any difficulties in the installation of some of the dependencies, please refer to the `install` instructions in this repository.

* Requirements Note: if you do not use conda to install and use pip instead, you may need to run `pip install -r requirements.txt` 

## Input data

The datasets needed to run these demos are located at [IBM Box](https://ibm.box.com/s/l8yzdbdb40j499o513hygx5q85xyoz6v). Please, download and place them in your local `input_data/` folder. 

**IMPORTANT NOTE**: These datasets have been modified with respect to their original versions and are only intended to be used in these demos. Use them for other purposes under your own responsability.

## Acknowledgement 

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 824988. https://musketeer.eu/

![](./EU.png)
