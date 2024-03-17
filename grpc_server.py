import json
import logging
import requests
import io

from concurrent import futures
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import ToTensor
from PIL import Image

import grpc
import numpy as np
import torch
import inference_pb2
import inference_pb2_grpc


class InferenceClassifier(inference_pb2_grpc.InstanceDetectorServicer):
    def __init__(self):
        self.model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
        self.names = MaskRCNN_ResNet50_FPN_Weights.COCO_V1.meta['categories']
        self.model.eval()
        self.to_tensor = ToTensor()

    @torch.inference_mode()
    def Predict(self, request, context):
        image_url = request.url
        image_data = requests.get(image_url).content

        image = Image.open(io.BytesIO(image_data))
        image = self.to_tensor(image)
        ans = self.model([image])
        ans = ans[0]

        scores = ans['scores']
        labels = ans['labels']

        named_labels = []

        for i in range(len(scores)):
            if scores[i] < 0.75:
                continue
            named_labels.append(self.names[labels[i]])

        named_labels.sort()

        return inference_pb2.InstanceDetectorOutput(objects=named_labels)


def serve():
    # to use processes - https://github.com/grpc/grpc/blob/master/examples/python/multiprocessing/server.py
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_InstanceDetectorServicer_to_server(InferenceClassifier(), server)
    server.add_insecure_port('[::]:9090')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    print("start serving...")
    serve()
