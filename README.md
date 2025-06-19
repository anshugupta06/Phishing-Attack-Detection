# Phishing-Attack-Detection
Features:

1.Develop an ML-based system for phishing detection: To create a machine learning system capable of identifying phishing websites and emails. 

2.Improve detection accuracy: To achieve a higher level of accuracy in identifying phishing attacks compared to traditional rule-based methods. 

3.Reduce false positives: To minimise the number of legitimate websites and emails incorrectly flagged as phishing. 

4.Enable real-time phishing detection: To implement a system that can identify phishing attempts as they occur, preventing associated cyber fraud. 

5.Enhance cybersecurity: To contribute to a stronger overall cyber security posture by providing more effective protection against phishing threats.

Project Architecture

![image](https://github.com/user-attachments/assets/8183e74b-1f97-4955-b17a-00d04b7d8b3b)

Tasks Completed

![image](https://github.com/user-attachments/assets/d58d3847-2b4e-47de-97c0-601a9b514dbd)

Challenges faced:
 1. Computational Challenges with CNN and BiLSTM Models: 
We faced significant delays while running deep learning models such as CNN and BiLSTM due to 
their high computational demands. To address this, we reduced the dataset size to an optimal level that 
allowed for efficient model training without compromising performance substantially. 
2. Difficulty in Implementing Ensemble Learning: 
Integrating ensemble learning techniques to combine the outputs of previously applied models posed a 
challenge, as it was a new concept for us. To overcome this, we undertook a thorough study of 
ensemble methods to identify suitable algorithms and understand their implementation. 
3. Issues Importing the ‘TensorFlow’ Library for GUI Integration: 
While working on the GUI integration, we encountered difficulties with importing the TensorFlow 
library. To resolve this, we explored its structure, including various modules, layers, and utility 
functions. Additionally, we used TensorFlow functionalities to set the random seed and ensure 
reproducibility during model training

Future Scope

•Advanced Model Integration: The project can further enhance its results by incorporating a wider 
array of machine learning models beyond the existing ones, exploring more sophisticated algorithms 
and architectures. 
•Dynamic Learning and Adaptation: Implement capabilities for the system to learn and adapt in 
real-time, allowing it to continuously improve its detection accuracy as new phishing techniques 
emerge. 
•API for Broader Integration: Develop a robust Application Programming Interface (API) to enable 
seamless integration of the phishing detection system with other security tools and platforms. 
•Holistic Data Processing: Expand the system's analytical capabilities to process multi-modal data, 
moving beyond traditional text-based analysis to include different data types that could indicate 
phishing attempts

Project Outcome

The expected outcome is a highly accurate, a phishing detection system capable of identifying and 
preventing phishing attacks in real time. It will efficiently distinguish between phishing and legitimate 
websites or emails with high precision and recall, minimising false positives and false negatives for 
enhanced cybersecurity. Furthermore, the system is expected to provide comprehensive logging and 
reporting capabilities for security analysis and incident response. Its modular design should facilitate 
future updates and integration of new detection techniques. Ultimately, this project aims to deliver a 
robust and scalable solution that significantly reduces the risk and impact of phishing attacks.

Progress Overview 
•Completed the data collection and cleaning. 

•Completed the feature extraction using SelectKBest method. 

•Implemented Logistic Regression, Naive Bayes, Random Forest XGBoost, SVM , CNN and BiLSTM. 

•Combined the output of all these algorithms using ensemble learning. 

•Visualised the results using confusion metrics and bar plot for result comparison. 

•Created a GUI application

Deliverables Progress

• The outcome is a highly accurate, AI-driven phishing detection system capable of identifying and preventing phishing attacks in real time. It will efficiently distinguish between phishing and legitimate websites or emails with high precision and recall, minimising false positives and false negatives for enhanced cybersecurity. 

• We were able to present the performance metrics with high accuracy, f1-score, recall and precision. 

• We also plotted the confusion metrics for all the models used here. 

• We also compared the results the of the various algorithms. 

• It was able to distinguish between legitimate and phishing URLs.

