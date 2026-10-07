from .model import OCRModel, OCRModelV0_1
from .decoder import greedy_decode
import torch
from PIL import Image
from torchvision import transforms

class OCRRecognizer:

    def __init__(self, model_state, device):
        
        self.model_state = model_state
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = OCRModel()
        self.model.load_state_dict(self.model_state)
        self.model.to(self.device)
        self.model.eval()

    def forward(self, x, lenghts):

        x = x.to(self.device)

        with torch.no_grad():

            logits = self.model(x, lenghts)

        decoded_text = greedy_decode(logits, lenghts)

        return decoded_text
    