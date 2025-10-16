from __future__ import print_function

from src.misc.config import cfg, cfg_from_file
from src.dataset import TextDataset
from src.trainer import condGANTrainer as trainer

import time
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import streamlit as st
from torchvision.models.inception import inception_v3
from scipy.linalg import sqrtm

# Load Inception model
inception_model = inception_v3(pretrained=True, transform_input=False)
inception_model.fc = torch.nn.Identity()
inception_model.eval()

def get_inception_features(imgs):
    """Extract features using Inception-v3"""
    imgs = torch.nn.functional.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
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

def modify_description(text):
    """Modify user input to generate multiple birds in a single image."""
    text = text.lower()

    if "two birds" in text or "pair of birds" in text:
        text += ", two birds interacting"
    elif "three birds" in text:
        text += ", three birds together"
    elif "flock of birds" in text or "group of birds" in text:
        text += ", multiple birds in one scene"

    if "talking" in text:
        text += ", birds appearing to communicate"
    elif "flying" in text:
        text += ", birds flying together"
    elif "sitting" in text or "on a rock" in text:
        text += ", birds resting on a rock or ground"
    elif "swimming" in text or "water" in text:
        text += ", birds floating on water"

    return text

def gen_example(wordtoix, algo, text):
    """Generate an image based on user input and compute evaluation metrics."""
    from nltk.tokenize import RegexpTokenizer
    import torchvision.transforms as transforms
    from PIL import Image

    text = modify_description(text)
    data_dic = {}
    captions = []
    cap_lens = []

    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)

    rev = []
    for t in tokens:
        t = t.encode("ascii", "ignore").decode("ascii")
        if len(t) > 0 and t in wordtoix:
            rev.append(wordtoix[t])
    captions.append(rev)
    cap_lens.append(len(rev))
    max_len = np.max(cap_lens)

    sorted_indices = np.argsort(cap_lens)[::-1]
    cap_lens = np.asarray(cap_lens)
    cap_lens = cap_lens[sorted_indices]
    cap_array = np.zeros((len(captions), max_len), dtype="int64")
    for i in range(len(captions)):
        idx = sorted_indices[i]
        cap = captions[idx]
        c_len = len(cap)
        cap_array[i, :c_len] = cap
    name = "output"
    key = name[(name.rfind("/") + 1) :]
    data_dic[key] = [cap_array, cap_lens, sorted_indices]

    fake_img_path = algo.gen_example(data_dic)

    # Compute FID & Inception Score
    transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
    fake_img = transform(Image.open(fake_img_path)).unsqueeze(0)

    fid_score = calculate_fid(fake_img, fake_img)  # Ideally, use real images if available
    is_score = calculate_inception_score(fake_img)

    return fake_img_path, fid_score, is_score

def demo_gan():
    cfg_from_file("eval_bird.yml")
    cfg.CUDA = False
    manualSeed = 100
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    output_dir = "output/"
    split_dir = "test"
    bshuffle = True
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose(
        [
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip(),
        ]
    )

    @st.cache_data(ttl=10000)
    def load_dataset():
        return TextDataset(cfg.DATA_DIR, split_dir, base_size=cfg.TREE.BASE_SIZE, transform=image_transform)

    dataset = load_dataset()

    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True,
        shuffle=bshuffle,
        num_workers=int(cfg.WORKERS),
    )

    @st.cache_resource
    def load_trainer():
        return trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)

    algo = load_trainer()

    st.title("Text to Image Synthesis")

    user_input = st.text_input("Describe the birds")

    if user_input:
        start_t = time.time()
        fake_img_path, fid_score, is_score = gen_example(dataset.wordtoix, algo, text=user_input)
        end_t = time.time()

        print("Total time for training:", end_t - start_t)
        print(f"FID Score: {fid_score:.4f}, Inception Score: {is_score:.4f}")

        st.write(f"**Your input**: {user_input}")
        st.subheader("Generated Image")
        st.image(fake_img_path)

        st.write(f"**FID Score**: {fid_score:.4f}")
        st.write(f"**Inception Score**: {is_score:.4f}")

        with st.expander("First stage images"):
            st.image("models/bird_AttnGAN2/output/0_s_0_g1.png")
            st.image("models/bird_AttnGAN2/output/0_s_0_a0.png")
