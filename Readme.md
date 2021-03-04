### Basic Tensor Converting API

STATUS: Functioning but still needs tweaking on reshaping the tensor array

A flask API that talks to an exported Tensorflow Model.  It will convert Images to Tensors for the requests, and convert Tensors to Strings for the response.
Tested with a trained model on the Fashion MNIST dataset.  The Jupyter Notebook used to train the model is in the repository.  

Instructions:

1. Build the docker image  ```docker build -t fashion:0.02 . --no-cache```
2. Run the docker image and forward port 5000 ```docker run -p 5000:5000 fashion:0.02```
3. Make a post request to this URL ```http://127.0.0.1:5000/api/recognize_image``` with a JSON body like this 

```
{
   "img_url" : "https://cf1.s3.souqcdn.com/item/2020/04/15/12/28/80/17/1/item_L_122880171_82d16380e5c8c.jpg"
}
```
