import torch
import torch.nn.functional as F
from torchvision.models.inception import inception_v3
from scipy.linalg import sqrtm
import numpy as np

# Load Inception model
inception_model = inception_v3(pretrained=True, transform_input=False)
inception_model.fc = torch.nn.Identity()  # Remove classification layer
inception_model.eval()

def get_inception_features(imgs):
    """Extract features using Inception-v3"""
    imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
    with torch.no_grad():
        features = inception_model(imgs)
    return features

def calculate_fid(real_imgs, fake_imgs):
    """Calculate FID score"""
    real_features = get_inception_features(real_imgs)
    fake_features = get_inception_features(fake_imgs)
    mu_real, sigma_real = real_features.mean(dim=0), torch.cov(real_features.T)
    mu_fake, sigma_fake = fake_features.mean(dim=0), torch.cov(fake_features.T)
    fid = torch.norm(mu_real - mu_fake) ** 2 + torch.trace(sigma_real + sigma_fake - 2 * sqrtm(sigma_real @ sigma_fake))
    return fid.item()

def calculate_inception_score(fake_imgs, splits=10):
    """Calculate Inception Score"""
    probs = torch.nn.functional.softmax(get_inception_features(fake_imgs), dim=1)
    scores = probs.mean(dim=0)
    kl_div = probs * (torch.log(probs) - torch.log(scores))
    kl_div = kl_div.sum(dim=1)
    is_score = torch.exp(kl_div.mean()).item()
    return is_score
