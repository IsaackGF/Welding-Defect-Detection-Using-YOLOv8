## WELDING DETECTION USING YOLOv8

<p>
BACKGROUND
</p>

Laser welding is a material  joining process that uses a high-energy laser beam to melt and fuse parts together. It is widely used in industries such as automotive, aerospace, electronics, and manufacturing due to its precision, speed, and ability to weld difficult-to-join materials.

<p>
  EXPERIMENT
</p>

First, I collected the data by downloading the dataset, which contained 176 images and 209 object instances to detect, divided into two groups: images of good welds and bad welds. 


Next step, I successfully trained the model with YOLOv8. These results show the summary of the model's training and validation. The training completed 100 epochs in just 8.46 minutes. The best model was evaluated on the validation set (best.pt):

<img width="1502" height="522" alt="image" src="https://github.com/user-attachments/assets/1947cd9d-9568-4ca9-a329-c86f3aa48a4e" />

Key Metrics:

Precision (P): Accuracy of positive predictions. Good Weld has the highest precision (67.3%).
Recall (R): Ability to detect all positive cases. Bad Weld has the highest recall (79.4%).
mAP50: Average precision at IoU=50%. Good Weld has the highest mAP50 (75.6%).
mAP50-95: Average precision at IoU from 50% to 95%. Good Weld also leads here (48.9%).



