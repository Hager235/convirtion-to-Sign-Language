#Sign language detection
collect_imgs.py:when it run it open the camera then capture the video and devide it to frames (each one 30 ms in multiple positions)
then press "q" and start new one 
each video extract 100 images 

from images :
create_dataset.py: model learn from it 
train_classifier:train the model 
20% from data test and 80% train 
no layer model 
inference_classifier:run in real time