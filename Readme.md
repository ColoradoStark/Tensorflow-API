### Basic Tensor Converting API

A flask API that talks to an exported Tensorflow Model.  It will convert Images to Tensors and Tensors to Strings.
Tested with a trained model on the Fashion MNIST dataset.  To use:

1. Build the docker image
2. Run the docker image and forward port 5000
3. Make a post request with a JSON body like this

```
{
   "img_url" : "https://cf1.s3.souqcdn.com/item/2020/04/15/12/28/80/17/1/item_L_122880171_82d16380e5c8c.jpg"
}
```
