## Linear Discriminant Analysis (LDA) & Quadratic Discriminant Analysis (QDA) Classifier:

The Iris Species dataset includes three types of iris species with 50 samples each as well as some properties about each flower. There are 4 features/attributes and three classes; Setosa, Versicolor and Virginica. This multi-class classification evaluation will involve performing a 5-fold cross-validation to assess the classification performance of Linear Discriminant Analysis and Quadratic Discriminant Analysis classifiers. The identification of the optimal classifier will be based on an evaluation of overall model performance, including an examination of individual metrics such as Confusion Matrix, sensitivity, specificity, total accuracy, and F1-score for each classifier as well as an ROC & AUC curve for whichever classifier that performs the best.

## 🛠️ Technologies Used:

Python: Leveraging the power of Pandas, NumPy, and Scikit-Learn.

Sublime Text: Efficient, versatile code editor with minimalist interface.

## 📈 Results:

**LDA Classifier**
* The classifier demonstrated outstanding predictive accuracy, achieving scores of 100%, 96.6%, 96.6%, 100%, and 96.6% across the 5 folds. Furthermore, the sensitivity scores exhibited remarkable precision, with 100% for Class 0 (Setosa), 92% for Class 1 (Versicolor), and 100% for Class 2 (Virginica). The specificity scores were equally impressive, recording 100% for class 0, 100% for class 1, and 95.2% for class 2. Additionally, the LDA classifier displayed robust F1 scores, achieving 100% for class 0, 96% for class 1, and 95% for class 2. The overall model accuracy reached a commendable 98%, affirming the classifier's effectiveness in accurately predicting and classifying the 3 Iris species. In summary, the high sensitivity, specificity, and F1 scores collectively suggest that the LDA classifier performs well on the Iris species dataset. It effectively identifies instances of each class, minimizes misclassifications, and maintains a balanced trade-off between precision and recall.

**QDA Classifier**
* The Quadratic Discriminant Analysis classifier, evaluated on the Iris species dataset through 5-fold cross-validation, exhibited strong predictive capabilities. With sensitivity scores of 100%, 92%, and 100% for Classes 0, 1, and 2, and specificity scores of 100%, 100%, and 95.2%, respectively, the classifier demonstrated accuracy in identifying positive and negative instances. The cross-validation accuracy remained consistent at 97.3%, with scores of 100%, 96.6%, 93.3%, 100%, and 96.6% across different folds, reflecting stability and effectiveness. Notably, F1 scores of 100%, 96%, and 95% for Classes 0, 1, and 2 highlighted a balanced trade-off between precision and recall. Overall, the QDA classifier proved proficient in accurately categorizing Iris species, showcasing high sensitivity, specificity, and consistent accuracy.

**ROC & AUC**
* In the evaluation of the Linear Discriminant Analysis (LDA) classifier on the Iris species dataset, an ROC curve and AUC graph were generated using a one-vs-rest classifier approach to plot the three distinct classes: Setosa, Versicolor, and Virginia. The ROC curve visually illustrates the trade-off between true positive rate (sensitivity) and false positive rate across different classification thresholds. For the Setosa class, the AUC achieved a perfect score of 1.0, indicating flawless discriminatory ability for this particular category. On the other hand, the AUC values for the Versicolor and Virginia classes were observed to be similar and close. This suggests that the LDA classifier exhibits comparable performance in distinguishing between Versicolor and Virginia, with their ROC curves demonstrating overlapping characteristics. This is in line with the data distribution as the Setosa species is separated from the other two species across all feature combinations.

**Conclusion** 
*  Both the Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA) classifiers performed exceptionally well in predicting the Iris species. Both classifiers achieved high sensitivity, specificity, and F1 scores, and their 5-fold cross-validation accuracy scores were comparable. In terms of overall model accuracy, both classifiers performed admirably, with LDA achieving an overall accuracy of 98% and QDA achieving an overall accuracy of 97.3%. Therefore, based on the calculated performance metrics, it's challenging to definitively choose one classifier as outperforming another. Given that the Iris Species dataset is relatively small, there is a concern about overfitting, making the adoption of LDA preferable due to its simpler model structure.

## 🔗 How to Use:

Each project/code along with their dataset has been uploaded for your review or observation. Please feel free to reach out if you have questions, suggestions, or if you're interested in collaboration!

## 🌐 Connect with Me:

LinkedIn: (https://www.linkedin.com/in/faridatlawal/)

##### I'm continuously learning and expanding my skill set. Join me on this exciting journey through the world of machine learning! 🤖✨
