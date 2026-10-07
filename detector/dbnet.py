from doctr.models import detection_predictor
import numpy as np

class DBNetDetector:

    def __init__(self):

        # self.device = device

        self.model = detection_predictor(
            arch="db_mobilenet_v3_large",
            pretrained=True
        )

    def predict(self, image):

        image = np.array(image)
        result = self.model([image])

        return result