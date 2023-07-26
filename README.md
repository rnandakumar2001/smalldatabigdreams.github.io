# Small Data Big Dreams
## Intro/Background

Data augmentation and transfer learning are vital for leveraging machine learning on small datasets. These techniques mitigate overfitting and improve model generalizability by artificially expanding the dataset and leveraging pre-existing knowledge, respectively. 

In this exploration, we'll delve into the application of these and similar methodologies for image classification and tabular regression problems on small datasets, presenting practical insights and challenges encountered in implementing these techniques.

## Problem Statement

The motivation of this project is to experiment with various dataset sizes and methodologies for data augmentation and transfer learning processes. Overfitting, limited dataset sizes, and overall accuracy in machine learning are still areas rife with potential for new development today, and the ability to artificially expand training sets from existing data offers an efficient breakthrough in the field. However, data augmentation today is still limited by the effectiveness of current approaches.

## Methods

We are going to explore different ways to deal with small datasets.
### Data Cleaning
#### Tabular Data
Using the uber dataset, there are many different comparisons that can be leveraged for training a deep learning model. Cleaning the uber dataset involved parsing
through all the data available and removing illfitting data(null data, outliers, irrelevant data).Additionally, only one days worth of data from the uber dataset was
used, this will serve as our training data and the results from the model will be compared to the actual values recorded in the uber dataset. The data was plotted on
scatter plots and a heat map to determine what sort of regression should be used to fit to the data. In the future we will apply the best fit regression model and
begin to train our diffusion model using TabDDPM.The metrics are availible lower in the document. See below for the resulting scatter plots of the cleaned data.

<img src='heatmapuber.PNG' width='200'> <img src='counthour.PNG' width='200'> <img src='countday.PNG' width='200'>

#### Image Data
For our image-based dataset, we used the well-known CIFAR-10 dataset accessible as part of the PyTorch library. We pared down the 60,000 images to a smaller subset of
5,000.

### Data augmentation:
Data augmentation methods, such as random croppings, rotations, and changing perspectives, are ways to deal with small datasets. We utilized these methods for data augmentation and will work to find which combinations of them are the optimal combo.<br><br>
Control Image:
- No augmentation was performed on these images<br>
   <img src='Control.png' width='300'>


Random Crop:
- Zooms into and crops a random portion of the image <br>
   <img src='Random Crop.png' width='300'>


Grayscale:
- Changes RGB values to shades of gray <br>
   <img src='GrayScale.png' width='300'>


Rotations (90, 180, and 270 degrees respectively): <br>
- This helps the model recognize all sorts of rotated images <br>
<img src='Rotate 90.png' width='300'> <img src='Rotate 180.png' width='300'> <img src='Rotate 270.png' width='300'>


Perspective Changes (0.3 and 0.6 respectively): <br>
- This is used to train the model for stretched and tilted images <br>
<img src='Weak Perspective.png' width='300'> <img src='Stronger Perspective.png' width='300'>

### Generative models for data synthesis:
Synthetic data as a supplement to real data is another way to deal with small datasets. Specifically, we will test the GAN and Diffusion models on their efficacy in generating synthetic data.
- GAN: We’ll use GAN on existing datasets to generate synthetic data.
- Diffusion model: We’ll use textual inversion on pre-trained diffusion models to engineer a prompt for our dataset, and use this prompt as a condition to generate synthetic data from the pre-trained diffusion model. We will use TabDPPM as a our diffusion model for tabular regression [2, 5].
### Transfer learning: 
We’ll further explore how we can fine-tune existing large models for small dataset classification tasks.
- Changing all parameters of the model is probably unrealistic given our current devices.
- Use LoRA to fine-tune the classification model.
- Add layers on top of the pre-trained model and freeze other layers.
- Use of Big Transfer model for image classification [1].
## Potential Results/Discussion

We'll use several metrics to gauge the success of our data augmentation techniques. The Fowlkes-Mallows index gauges the similarity between synthetic and original data, with a higher score signaling better augmentation. The AUC-ROC, an evaluation measure for classification problems, plots the True Positive Rate against the False Positive Rate. We anticipate improved scores with synthetic data. For multi-class models, multiple AUC-ROC curves will be generated. In tabular regression tasks, we'll use RMSE and MAE, metrics that quantify prediction deviations from actual values, thus offering a holistic view of our prediction accuracy. We aim for these scores to also improve with the use of synthetic data [3, 4].
  After the use of data augmentation, we will utilize two main scoring metrics to determine the effectiveness of the synthetic data. First, the Fowlkess-Mallows Measure utilizes the following equation:
![TP FN](/eq1.png)

  We expect a score between 0 and 1 as well as the FM measure being higher for the data augmented set. 
  The second method that we will use is the “Area Under Curve” of the “Receiver Operating Characteristic” or AUC-ROC. This plots True Positive Rate (TPR) vs False Positive Rate (FPR) where:

![TPR FPR](/eq2.png)

  In the AUC-ROC curve, a higher value of X signifies more False positives than True negatives and a higher Y means more True positives than False negatives. The values of the AUC range from 0 to 1, where:
  - 0.7 - 0.8 is acceptable
  - 0.8 - 0.9 is excellent
  - 0.9+ is outstanding [3]

Similarly to the FM measure, we expect the AUC-ROC to be higher for the synthetic dataset. 
  
  When using the AUC-ROC for multi-class models with N number of classes, we will plot N number of AUC-ROC curves. For example, if there are three dog breeds (A, B, and C), then we will have 1 ROC for A classified against B and C, 1 ROC for B against A and C, and 1 ROC for C against A and B.

![ROC](/eq3.png)

## Results

![RMSE](/RMSE.PNG)
![MAE](/MAE.PNG)
![ValidationvSample](/ValidationvSample.PNG)

  
## Timeline:
![Timeline](/Timeline.png)

## Contribution Table:

| Name            | Contribution                              |
|-----------------|-------------------------------------------|
| Gabe Graves     | XGBoost, GAN |
| Lucy Xing       | Data Cleaning                          |
| Hyuk Huang      | Data Augmentation                     |
| Hannah Lee      | CNN, F1 score                                   |
| Rohan Nandakumar| Tabular Preprocessing, Diffusion                      |




## References:
[1] A. Kolesnikov et al., “Big transfer (BIT): General Visual Representation Learning,” Computer Vision – ECCV 2020, pp. 491–507, 2020. doi:10.1007/978-3-030-58558-7_29  
[2] A. Kotelnikov, D. Baranchuk, I. Rubachev, and A. Babenko, TabDDPM: Modelling Tabular Data with Diffusion Models. doi: https://doi.org/10.48550/arXiv.2209.15421 Focus to learn more  
[3] J. N. Mandrekar, “Receiver operating characteristic curve in diagnostic test assessment,” Journal of Thoracic Oncology, vol. 5, no. 9, pp. 1315–1316, 2010. doi:10.1097/jto.0b013e3181ec173d  
[4] S. Narkhede, “Understanding AUC - roc curve,” Medium, https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5 (accessed Jun. 16, 2023).   
[5] “Textual inversion,” Textual Inversion, https://huggingface.co/docs/diffusers/training/text_inversion#:~:text=Textual%20Inversion%20is%20a%20technique,model%20variants%20like%20Stable%20Diffusion (accessed Jun. 16, 2023).
