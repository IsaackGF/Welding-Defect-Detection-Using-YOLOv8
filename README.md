## WELDING DETECTION USING YOLOv8

<p>
BACKGROUND:

</p>

Laser welding is a material  joining process that uses a high-energy laser beam to melt and fuse parts together. It is widely used in industries such as automotive, aerospace, electronics, and manufacturing due to its precision, speed, and ability to weld difficult-to-join materials. 

<img width="485" height="332" alt="image" src="https://github.com/user-attachments/assets/faf7ef19-32a3-42d0-ab30-ec6f87d5b03c" />

<p>
EXPERIMENT:

</p>

First, I collected the data by downloading the dataset, which contained 176 images and 209 object instances to detect, divided into two groups: images of good welds and bad welds. 
Next step, I successfully trained the model with YOLOv8. These results show the summary of the model's training and validation. The training completed 100 epochs in just 8.46 minutes. The best model was evaluated on the validation set (best.pt):

<img width="1502" height="522" alt="image" src="https://github.com/user-attachments/assets/1947cd9d-9568-4ca9-a329-c86f3aa48a4e" />

Key Metrics:

Precision (P): Accuracy of positive predictions. Good Weld has the highest precision (67.3%).
Recall (R): Ability to detect all positive cases. Bad Weld has the highest recall (79.4%).
mAP50: Average precision at IoU=50%. Good Weld has the highest mAP50 (75.6%).
mAP50-95: Average precision at IoU from 50% to 95%. Good Weld also leads here (48.9%).


And to verify the model's performance, the next step is to run the test.py file, which executes an evaluation and prediction script using a YOLOv8 model for detecting welding defects.

<img width="1500" height="541" alt="image" src="https://github.com/user-attachments/assets/78992fd6-50b7-4780-b29b-e396a9c4f4e3" />
"Bad Weld" has higher recall (35.4%) but low precision (22.6%)."Good Weld" has higher precision (34.1%) but low recall (18.4%)."Defect" has perfect precision (1.0) but zero recall (0%), suggesting that the model almost never detects this class.

<p>RESULTS 
</p>
This is the label distribution of the training/validation set. With the Class Histogram (Good Weld, Bad Weld, Defect). The existing imbalance (e.g., 'Defect' with few samples) explains its low mAP,  theb Bounding Box Size Distribution and the Bounding Box Location: Heatmap of object positions in the images.
<img width="937" height="1004" alt="image" src="https://github.com/user-attachments/assets/1a51d6bf-599d-4592-9836-33b1c114ef9c" />

Model Evaluation
Low Overall Performance:
mAP50-95: 0.051 (Very low, ideal >0.5)
Recall: 17.9% (Detects few real defects)
Precision: 52.2% (When it predicts, it's right about half the time)
By Class:
Bad Weld: mAP50=0.169 (Acceptable)
Good Weld: mAP50=0.139 (Poor)
Defect: mAP50=0.0258 (Very bad)

<img width="1500" height="768" alt="image" src="https://github.com/user-attachments/assets/7fbb0ae6-eb65-488a-9768-9c66411c56f2" />

<img width="1500" height="1125" alt="image" src="https://github.com/user-attachments/assets/40f7bf82-4fe0-4c23-bf31-a2012e5dce62" />

<p>CONCLUSION</p>
Laser welding is an advanced technology that offers numerous advantages in terms of precision, speed, and quality.However, it requires in-depth knowledge of process parameters and inspection techniques to ensure optimal results.
The model’s performance is not adequate for real-world use. Specifically, in the "Defect" class, it fails to detect any cases correctly, with a prediction rate of only 50%. One possible solution could be to increase training epochs, for exemple 200. 

<img width="1824" height="81" alt="image" src="https://github.com/user-attachments/assets/28f44ab6-fb7b-4757-a601-4fced3182e8d" />

