import os
import torch
from torchvision.models import vgg19
from PIL import Image
import torchvision.transforms as transforms
import json
import matplotlib.pyplot as plt

vgg_model = vgg19()
content = Image.open("../data/content.jpg").resize((224, 224))
style = Image.open("../data/style.jpg").resize((224, 224))

transform = transforms.Compose([transforms.ToTensor()])
content_tensor = transform(content).unsqueeze(0)
style_tensor = transform(style).unsqueeze(0)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print("Running on: ", device)

vgg_model.to(device)
for name, tensor in vgg_model.named_parameters():
    tensor.requires_grad = False

def extract_features(layers, image, model):
    features = {}
    x = image.to(device)
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

def get_gram_matrix(activation):
    b, c, h, w = activation.shape
    gram_matrices = []
    for i in range(b):
        curr_activation = activation[i].reshape(c, h * w)
        gram_matrices.append(torch.matmul(curr_activation, curr_activation.T))
    return torch.stack(gram_matrices)

layers = ["0", "2", "5", "7", "10", "12", "14", "16", "19", "21", "23", "25", "28", "30", "32", "34"]
content_features = extract_features(layers, content_tensor, vgg_model.features)
style_features = extract_features(layers, style_tensor, vgg_model.features)
style_gram_features = {}
for name, activation in style_features.items():
    style_gram_features[name] = get_gram_matrix(activation)

from torch.optim import Adam
import torch.nn.functional as f
from tqdm import tqdm
import numpy as np

def imcnvt(image):
    x = image.to("cpu").clone().detach().numpy().squeeze()
    x = x.transpose(1, 2, 0)
    x = x * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    return np.clip(x, 0, 1)

def train_model_with_params(content_tensor,
                            style_tensor,
                            content_features,
                            style_gram_features,
                            weights,
                            style_grams_weights,
                            learning_rate,
                            output_path,
                            use_style_as_input = False):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if use_style_as_input:
        x = style_tensor.clone().to(device)
    else:
        x = content_tensor.clone().to(device)
    x.requires_grad = True
    optimizer = Adam([x], lr = learning_rate)
    epochs = 1000
    logging_step = 100
    content_weight = weights[0]
    style_weight = weights[1]

    losses = []
    for epoch in tqdm(range(epochs)):
        activations = extract_features(layers, x, vgg_model.features)
        content_loss, style_loss = 0, 0
        for layer in layers:
            b, c, h, w = activations[layer].shape
            current_grams = get_gram_matrix(activations[layer])
            style_loss += (f.mse_loss(current_grams, style_gram_features[layer]) / (c * h * w)) * style_grams_weights[layer]
            content_loss += f.mse_loss(activations[layer], content_features[layer])
        total_loss = content_weight * content_loss + style_weight * style_loss
        losses.append(total_loss.item())
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % logging_step == 0:
            plt.imsave(f"{output_path}/{str(epoch + 1)}.png", imcnvt(x), format='png')
    return losses

parameters = {
    "weights" : [
        (100, 1e4),
        (100, 1e6),
        (100, 1e8),
        (100, 1e10),
    ],
    "style_grams_weights" : [
        {layer : 1 / (int(layer) + 1) for layer in layers},
        {layer : 1 / (34 - int(layer) + 1) for layer in layers},
        {layer : abs(17 - int(layer)) / (int(layer) + 1) for layer in layers},
    ],
    "learning_rate" : [
        0.10,
        0.05,
        0.01,
    ]
}

folder_maps = {}
execution = 1
for weight in tqdm(parameters["weights"]):
    for style_gram_weight in tqdm(parameters["style_grams_weights"]):
        for learning_rate in tqdm(parameters["learning_rate"]):
            input_options = [True, False]
            output_folder = [f"../results/{execution}/style", f"../results/{execution}/content"]
            losses = []
            for input_option, output_folder in tqdm(zip(input_options, output_folder)):
                loss = train_model_with_params(
                    content_tensor,
                    style_tensor,
                    content_features,
                    style_gram_features,
                    weight,
                    style_gram_weight,
                    learning_rate,
                    output_folder,
                    use_style_as_input = input_option
                )
                losses.append(loss)

            folder_maps[execution] = {
                'weight' : weight,
                'style_gram_weight' : style_gram_weight,
                'learning_rate' : learning_rate,
                'losses_style_input' : losses[0],
                'losses_content_input' : losses[1]
            }
            execution += 1
            
with open("../results/folder_maps.json", "w") as fp:
    json.dump(folder_maps, fp)

