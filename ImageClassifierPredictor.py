import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import requests
from io import BytesIO


class ImageClassifier:
    def __init__(self, model_path="model/inceptionv3_multi_label_model_All.pth", num_labels=40, threshold=0.25):
        # Define data transformations
        self.transform = transforms.Compose([
            # InceptionV3 expects 299x299 images
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

        # Define the model architecture
        self.model = self.load_model(model_path, num_labels)

        # Set threshold
        self.threshold = threshold

    def load_model(self, model_path, num_labels):
        model = CustomResNet(num_labels)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        return model
    
    def open_image_from_url(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            image_data = BytesIO(response.content)
            image = Image.open(image_data)
            return image
        else:
            print(f"Failed to fetch image from URL: {url}")
            return None

    def predict(self, url):
        # Load and preprocess the new image
        new_image = self.open_image_from_url(url).convert('RGB')
        new_image = self.transform(new_image).unsqueeze(
            0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output = self.model(new_image)

        # Convert output to binary values based on a threshold
        predicted_labels = (output > self.threshold).float()

        return predicted_labels.tolist()[0]

    def predict_one(self, url):
        return {url: self.predict(url)}

    def predict_multiple(self, list_url):
        result = {}
        for i in list_url:
            result[i] = self.predict(i)
        return result


class CustomResNet(nn.Module):
    def __init__(self, num_labels):
        super(CustomResNet, self).__init__()
        self.inception_v3 = models.inception_v3(
            pretrained=False, aux_logits=False)
        in_features = self.inception_v3.fc.in_features
        self.inception_v3.fc = nn.Linear(in_features, num_labels)

    def forward(self, x):
        return self.inception_v3(x)
