FROM tensorflow/tensorflow:2.4.1

EXPOSE 5000

COPY ./ /tf

RUN pip install flask
RUN pip install scikit-image
RUN pip install sklearn

CMD python /tf/API/api.py

