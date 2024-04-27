## Project Description
The purpose of the project is processing of ultrasonic heart examination videos in order to segment parts of a heart (valves, ventricles, atriums) and to calculate the ejection fraction.
The dataset was really very uncomfortable for processing: neither COCO format, nor Pascal/VOC. Some kind of custom marks, a lot of defects, cyrillic symbols...
So the DataSet class in the notebook looks like Frankenstein’s Monster👻
____

## Models and Experiments

There were several experiments with instance segmentation models:
- MaskRCNN - unsuccessful (too less data, probably)
- Yolo8 - successful - MaP95 up to 0,9

Also there was an idea to start with pretrained U-Net a s semantic segmentation model and then train a classifier model, but less than 100 images in training data made this idea unrealizable.
____
## Video Processing

First of all, the video frames are cropped to leave only the area with a heart (usually, it is the same part of the frame for different videos).
The trained model is used for video processing, drawing masks.
The visible area of the left ventricle mask ($LVA$) is used to calculate an important vital parameter - ejection fraction.
The formula is
$\dfrac{LVA_{max}  -LVA_{min}}{LVA_{max}}$.

Rolling mean function can be used to smooth the artifacts of segmentation.

Then a list of frames is saved as a video file.
___
## Results

The main results:
- instance segmentation model for heart ultrasonic videos
- calculation of the ejection fraction as a helper function for the doctor
____
## Interface

The scripts for Gradio interface are available in the /app folder.

[The application is deployed at HuggingFace Spaces](https://huggingface.co/spaces/fkonovalenko/hertz).
_____
## References

- [MaskRCNN PyTorch Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [One more CAMUS heart dataset](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8)
- [Yolo8 Instance segmentation](https://towardsdatascience.com/trian-yolov8-instance-segmentation-on-your-data-6ffa04b2debd)
- [Yolo train settings docs](https://docs.ultralytics.com/modes/train/#train-settings)
