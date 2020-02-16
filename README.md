# Image-Classifiaction-Using-Resnet-
This Model Runs on Free Heroku server .Static page is inside app.py . Model is loaded by keras application 




If you train your model and try to make pridiction using it on free srever 

Here is the whole crush, heroku gives only 30s for one request and a total of 500MB of RAM. Both of them is insufficient for deep learning models as evaluation and processing takes more time as well as more memory .


So in this code with the help of keras pretrain model i imported the pre trained model so that at time op prediction on server i donot have to load the model and also i created static page inside the app.py to reduce time complixity and memory use on server 


i local system you just run python app.py 
