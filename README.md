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
<img width="3801" height="81" alt="image" src="https://github.com/user-attachments/assets/498d3b19-2adf-4f84-8647-a205215b53e4" />


