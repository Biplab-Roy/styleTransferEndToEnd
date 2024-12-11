import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.optim import Adam
import torch.nn.functional as f
import numpy as np
from tqdm import tqdm

class StyleTransferRunner:

    def __init__(self):
        self.layers = ["0", "2", "5", "7", "10", "12", "14", "16", "19", "21", "23", "25", "28", "30", "32", "34"]
        self.model = torch.load("./objs/vgg_features.pt")
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        print("Running on: ", self.device)

    def imcnvt(self, image):
        x = image.to("cpu").clone().detach().numpy().squeeze()
        x = x.transpose(1, 2, 0)
        x = x * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
        return np.clip(x, 0, 1)

    def extract_features(self, image):
        features = {}
        x = image.to(self.device)
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.layers:
                features[name] = x
        return features

    def get_gram_matrix(self, activation):
        b, c, h, w = activation.shape
        gram_matrices = []
        for i in range(b):
            curr_activation = activation[i].reshape(c, h * w)
            gram_matrices.append(torch.matmul(curr_activation, curr_activation.T))
        return torch.stack(gram_matrices)

    def train_model_with_params(self,
                                content_tensor,
                                style_tensor,
                                content_features,
                                style_gram_features,
                                weights,
                                style_grams_weights,
                                learning_rate,
                                dir,
                                epochs = 1000,
                                logging_step = 100,
                                use_style_as_input = True):

        if use_style_as_input:
            x = style_tensor.clone().to(self.device)
        else:
            x = content_tensor.clone().to(self.device)
        x.requires_grad = True

        optimizer = Adam([x], lr = learning_rate)
        content_weight = weights[0]
        style_weight = weights[1]

        Images = []
        for epoch in tqdm(range(epochs)):
            activations = self.extract_features(x)
            content_loss, style_loss = 0, 0
            for layer in self.layers:
                b, c, h, w = activations[layer].shape
                current_grams = self.get_gram_matrix(activations[layer])
                style_loss += (f.mse_loss(current_grams, style_gram_features[layer]) / (c * h * w)) * style_grams_weights[layer]
                content_loss += f.mse_loss(activations[layer], content_features[layer])
            total_loss = content_weight * content_loss + style_weight * style_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (epoch + 1) % logging_step == 0:
                curr_image = self.imcnvt(x)
                Images.append(f"{dir}/{str(epoch + 1)}.png")
                plt.imsave(f"{dir}/{str(epoch + 1)}.png", curr_image, format='png')
        
        return Images


    def transferStyle(self, 
                    content_tensor,
                    style_tensor,
                    weights,
                    learning_rate,
                    samples,
                    dir = "../temp"):
        transform = transforms.Compose([transforms.ToTensor()])
        content_tensor = transform(content_tensor).unsqueeze(0)
        style_tensor = transform(style_tensor).unsqueeze(0)
        style_features = self.extract_features(style_tensor)
        style_gram_features = {}
        for name, activation in style_features.items():
            style_gram_features[name] = self.get_gram_matrix(activation)
        style_grams_weights = {layer : 1 / (int(layer) + 1) for layer in self.layers}
        step = 10

        return self.train_model_with_params(
            content_tensor,
            style_tensor,
            self.extract_features(content_tensor),
            style_gram_features,
            weights,
            style_grams_weights,
            learning_rate,
            dir,
            samples * step,
            step,
            use_style_as_input=False,
        )
    

        