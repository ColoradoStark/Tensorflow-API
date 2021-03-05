### Tensorflow REST API

This is a basic demo for putting a machine learning model that has been trained with a neural net into production. It addresses the basic challenge that the inputs and outputs of a tensorflow model are not human readable.

The source code will build a flask API that runs tensorflow, numpy, scikit and will talk to a pre-trained saved Tensorflow Model. It will convert Images to Tensors for the requests, and convert Tensors to Strings for the response.

Tested with a trained model on the Fashion MNIST dataset. The Jupyter Notebook used to train the model, and the exported trained model are in the repository.

Instructions:

1. Run the docker image and forward port 5000 `docker run -p 5000:5000 coloradostark/fashion:0.08`
2. Make a post request to this URL `http://127.0.0.1:5000/api/recognize_image` with a JSON body like this

```
{
   "img_url" : "https://cf1.s3.souqcdn.com/item/2020/04/15/12/28/80/17/1/item_L_122880171_82d16380e5c8c.jpg"
}
```
