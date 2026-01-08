## Technologies Used

The following tools and libraries were used in this project:

- **Python**
- **Jupyter Notebook**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **scikit-learn**
- **PyTorch**


## Training Data Description

The dataset used in this project is called *“Breast Histopathology Images”* and comes from [Kaggle.com](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)
. It was created based on **163** breast histopathology whole-slide images stained with **H&E** and scanned using a **WSI scanner** at **40× magnification**. By splitting these images, **RGB patches** of size **50×50 pixels** were obtained. In total, this resulted in **277,524** samples, of which **198,738** were labeled as **healthy**, while **78,786** were labeled as **cancerous**.

Additionally, each patch has a strictly defined filename in the format: `U_xX_yY_classC.png`.  
Here:
- **U** encodes the **patient identifier**,
- **X** and **Y** refer to the **coordinates** from which the patch was extracted,
- **C** indicates the **class** of the patch, where **0** represents a **healthy** sample and **1** represents a **cancerous** sample.

The patch generation procedure begins with transferring the scanned image to a pathologist, who performs the analysis and then marks regions affected by the tumor. Next, a grid is applied to the image, based on which patches are generated. Moreover, the patch creation process assumes the removal of regions that contain mostly adipose tissue, and a given patch is labeled as cancerous if at least **80%** of its area lies within the diseased region marked by the pathologist. 

Examples of the resulting dataset samples are shown below:

- **(a)** ![Healthy Samples](/resources/healthy_samples.png) 
- **(b)** ![Sick Samples](/resources/sick_samples.png) 

*Randomly selected examples of patches: (a) healthy samples, (b) sick samples.*


Based on the previously described filename structure, a dataframe was created containing information about the **patient identifier**, **X and Y coordinates**, **sample class**, and the **file path** to each patch. Storing the available information in this way made it possible to build a complete view of the histopathological sample by reconstructing it from its patches. This approach allows healthy and cancerous tissue regions to be analyzed from a broader perspective. 

The reconstructions created in this way are shown below:

- **(a)** ![Tissue images without disease information](/resources/sample_reconstruction.png) 
- **(b)** ![Tissue images with tumor regions marked](/resources/sample_reconstruction_marked.png) 


*Tissue images after merging their component patches: (a) tissues without disease information, (b) tissues with marked tumor regions.* 

## Data Analysis

### Sample distribution

As mentioned earlier, the data comes from the *“Breast Histopathology Images”* dataset. It contains data from **279 patients**. In total, there are **277,524** samples available, among which **198,738** are classified as **healthy**, and **78,786** samples are classified as **cancerous**. 
The distribution of samples is shown below:

![Sample distribution in the full dataset](/resources/samples_distibution.png) 

*Distribution of samples in the entire dataset. *


The histogram shows that healthy samples make up **71.6%** of the entire dataset, while cancerous samples account for **28.4%**. Such a disproportion—where healthy samples significantly outnumber cancerous samples—leads to a problem known as **class imbalance**. This characteristic must be taken into account when building a machine learning model and during its later evaluation, because the algorithm may prefer predicting the more frequent class.

### Percentage of sick samples

The next step in the analysis was to verify how large a portion of each patient’s tissue is covered by cancerous areas. The results are presented below:

![Percentage of sick samples per patient](/resources/sick_samples_per_patient.png)

*Percentage of sick samples per patient*

The histogram indicates that most patient tissues contain about **20–30%** cancerous patches. The mean value is **30.8%**, and the median is approximately **26.6%**. It is also worth noting that some patients have tissues where more than **80%** of patches are cancerous. This may indicate a very late stage of the disease or inaccurate data acquisition, for example, if the WSI image was taken only for a part of the tissue.

### Samples per patient

Another element analyzed was the number of samples per patient. This is shown here:

![Number of samples per patient](/resources/samples_per_patient.png)

*Number of samples per patient*

It can be seen that most patients have around **1,000** samples. The average number of samples is **994.7**, and the median is **967** samples. Moreover, more than **60** patients have fewer than **500** samples, which may result from differences in image resolution or partial data loss. Another reason may be the earlier assumption about removing adipose tissue regions, which could suggest that patients with a small number of samples had a high amount of fat in the tissue.

### Pixel distribution

The next stage of the analysis was to examine the distribution of pixel intensities in randomly selected healthy and cancerous samples. 

- **(a)** ![Random healthy samples and pixel distribution analysis](/resources/healthy_sample_hist1.png) 
- **(b)** ![Random healthy samples and pixel distribution analysis](/resources/healthy_sample_hist2.png)

*Random samples classified as healthy and pixel distribution analysis.*

- **(a)** ![Random sick samples and pixel distribution analysis](/resources/sick_sample_hist1.png) 
- **(b)** ![Random sick samples and pixel distribution analysis](/resources/sick_sample_hist2.png)

*Random samples classified as sick and pixel distribution analysis.*



A comparison of both sample types shows that healthy samples have a significant dominance of **red-channel pixels** compared to the **blue** and **green** channels. It can be seen that the number of red pixels is roughly **twice** as large as the number of blue and green pixels. In contrast, in cancerous samples, red pixels are comparable to green and blue pixels, and they may even become the minority. The described pixel distribution results in healthy tissues having a colour ranging from light red to pink, while diseased tissues have a purple hue.

## Machine Learning Model

From the perspective of correctly designing a system for tumor detection in histopathological images, it was necessary to split the dataset. The dataset was divided in a **60:20:20** ratio, where **60%** of the data was used for **training**, **20%** for **model validation**, and **20%** for the **final testing** of the system. 

![Class distribution across train, validation, and test sets](/resources/samples_split_distribution.png) 

As shown ,the class distribution in each subset is close to the distribution of the full dataset, which equals **71.6%** to **28.4%**.

To increase the amount of data in the training set, **data augmentation** was applied. Due to the nature of the data—patches extracted from larger images—standard augmentation techniques were used instead of deformation-based methods and synthetic image generation. In this dataset, augmentation consisted of a **random vertical flip** and a **random horizontal flip**, each applied with a **50% probability**. An example of augmented samples is shown below:

![Class distribution across train, validation, and test sets](/resources/augmentation.png) 

*Example of the applied data augmentation.*


In the presented figure, the previously discussed randomness of flips can be observed: samples in columns **2, 3, and 5** contain **both flips**, while columns **1 and 4** present only a **vertical flip**. In the shown set of samples, no case of a standalone **horizontal flip** was observed. It should also be emphasized that data augmentation was used **only for training purposes**. Validation samples should reflect real examples as closely as possible, therefore they must not undergo any processes that alter their appearance.


### Network Architecture

In the described tumor change detection system, the **ResNet-18** model was used, which is the baseline network within the ResNet family. The main argument for choosing a relatively shallow network was that, in this project, the model does not need to recognize highly abstract, high-level features. In practice, the network is expected to analyze the **cellular structure** of tissue and its **color**, which largely relies on detecting **image contours and local patterns**. In addition, a smaller network requires less computational power.

The structure of ResNet-18 is presented below:

![ResNet-18 Architecture](/resources/resnet18.png) 

*ResNet-18 Architecture: X. Ou, P. Yan, Y. Zhang, B. Tu, G. Zhang, J. Wu, W. Li, Moving object detection method via ResNet-18*


ResNet-18 consists of **16 convolutional layers**, **2 sampling (pooling) layers**, and several **fully connected layers**. The network input consists of images sized **224×224 pixels**, while the convolutional kernel sizes are **7×7** and **3×3**, with only the very first convolution using the **7×7** kernel.

At the end of the network, a **reduction layer** performs averaging of the outputs from the final convolutional layer and passes the result to the fully connected layers. This produces an output vector that is then fed into the **Softmax** normalization function, resulting in a probability vector representing membership in a given class.

The presented architecture includes both types of skip connections described earlier: **solid lines** indicate direct shortcut connections, while **dashed lines** represent shortcut connections that include an additional convolutional layer.


## Results Analysis

### Confusion matrix

The final verification of the classifier was performed on the previously separated **test set**, which was not used for either training or validation. This makes it possible to evaluate the model on unseen data and assess its ability to generalize. The primary tool used for model validation is the **confusion matrix**, and its values for the tumor tissue classification model are shown here:


![Confusion matrix](/resources/confusion_matrix.png) 

*Confusion matrix*


From the confusion matrix, the following information can be derived:

- The number of **true positives** (*TP – true positives*), i.e., samples that are cancerous in reality and classified as cancerous, was **13,893**.
- The number of **false positives** (*FP – false positives*), i.e., samples that are healthy in reality but classified as cancerous, was **1,501**.
- The number of **true negatives** (*TN – true negatives*), i.e., samples that are healthy in reality and classified as healthy, was **38,096**.
- The number of **false negatives** (*FN – false negatives*), i.e., samples that are cancerous in reality but classified as healthy, was **2,015**.

Based on the values obtained from the confusion matrix, it is possible to compute the basic metrics that characterize the model’s performance. The baseline metric, **accuracy**, reached **93.6%**.

$$
\text{Accuracy} = \frac{TP + TN}{TP + FP + FN + TN}
= \frac{13893 + 38096}{13893 + 1501 + 2015 + 38096}
= 93.6\%
$$


It should also be remembered that the analyzed dataset had an **imbalanced class distribution**, where healthy samples constituted about **70%**, while cancerous samples only about **30%**. For this reason, accuracy cannot be the only metric describing the model, and **precision** should also be considered. In this case, the classifier precision for the minority class was **90.2%**.

$$
\text{Precision} = \frac{TP}{TP + FP}
= \frac{13893}{13893 + 1501}
= 90.2\%
$$

From the medical perspective of this classification problem, it is important to analyze **sensitivity** and **specificity**. Among these metrics, **sensitivity** is more important for the problem at hand because it describes how well the model recognizes cancerous tissues. Based on the confusion matrix results, out of **15,908** cancerous samples, the classifier correctly identified **13,893** samples, achieving a sensitivity of **87.3%**.

$$
\text{Sensitivity} = \frac{TP}{TP + FN}
= \frac{13893}{13893 + 2015}
= 87.3\%
$$

**Specificity**, on the other hand, enables analysis of the classifier behavior for healthy tissues. With **39,597** healthy samples available, the algorithm correctly classified **38,096** of them, which translated into a specificity of **96.2%**.

$$
\text{Specificity} = \frac{TN}{TN + FP}
= \frac{38096}{38096 + 1501}
= 96.2\%
$$

Two separate indicators—precision and sensitivity—can be replaced by a single metric: the **F1-score**. This is a more reliable metric that can account for class imbalance in its value. In this project, the F1-score reached **0.887**, where the best possible value is **1**.

$$
\text{F1-score} = \frac{2TP}{2TP + FP + FN}
= \frac{2 \cdot 13893}{2 \cdot 13893 + 1501 + 2015}
= 0.887
$$

### ROC curve

A popular way to present model performance is the **ROC curve** (*ROC – Receiver Operating Characteristic*), which visualizes the classifier quality by showing the relationship between **TPR** (true positive rate: correctly classifying a truly cancerous sample) and **FPR** (false positive rate: incorrectly classifying a truly healthy sample as cancerous). Figure below presents the ROC curve for the proposed model and its comparison with a random classifier.

![ROC curve](/resources/roc.png)

*ROC curve.*

In theory, the more convex the ROC curve is and the closer it runs to the point **(0.0, 1.0)**, the better the classifier. The presented plot shows that the designed model lies relatively close to **(0.0, 1.0)**, which indicates good performance. Closely related to the ROC curve is the **AUC** (*AUC – Area Under the ROC Curve*), which represents the area under the ROC curve. Its value ranges from **0** to **1**, where **1** corresponds to a perfect classifier. The built classifier achieved an **AUC of 0.92**, which further supports its effective performance in classifying tissue samples.


## Conclusions

To summarize the obtained results, several conclusions can be drawn regarding the constructed classifier. Due to the medical nature of the task, analyzing **sensitivity** is essential, because it shows whether cancerous cases will indeed be classified as cancerous. In theory, it is better for sensitivity to be higher than specificity, because in such a scenario the model is more likely to classify hard-to-decide samples as cancerous. This approach has a lower cost of errors, because if a healthy sample is classified as cancerous, the patient will undergo additional examinations that will ultimately confirm a false diagnosis. In the second case—when a cancerous sample is classified as healthy—the patient will most likely leave the healthcare facility and return only at an advanced stage of the disease, which would lead to poor prognosis and reduced chances of recovery.

In the obtained results, there is a noticeable difference between sensitivity and specificity equal to **8.9%**. This means that in situations where the algorithm was uncertain about the sample type, it preferred labeling it as healthy, which results in a more frequent occurrence of the second (more dangerous) scenario described above. This behavior is caused by the repeatedly mentioned **class imbalance** in the dataset, and the phenomenon of favoring the majority class was already anticipated during the dataset analysis stage. To improve the model’s performance, it would be necessary to collect more data—especially data representing cancerous samples—so that the class imbalance effect could be reduced.

Another aspect characterizing the built model is its ability to **support humans**. Considering all metrics (accuracy of **93.6%**, precision of **90.2%**, and an F1-score of **0.887**) and the ROC curve shape, it can be assumed that the presented algorithm could successfully assist physicians in clinical diagnostics of histopathological tissues. Introducing such a solution would reduce human workload by relieving specialists from the tedious task of analyzing images produced by WSI scanners. The time saved could be invested in improving the quality of patient care or expanding professional competencies. Nevertheless, the final decision regarding diagnosis should remain with the physician. At the same time, it is worth recalling the positive impact of high-performing and reliable decision-support systems, which can improve the correctness of decisions made by humans.

A major advantage of the proposed model is also its ability to make predictions based on **smaller parts** of a histopathological image. This approach enables the detection of small disease foci, unlike a human expert who analyzes the entire image as a whole and may overlook smaller regions.
