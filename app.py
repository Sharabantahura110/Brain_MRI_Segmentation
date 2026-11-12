"""Local educational demo. Launch: streamlit run app.py"""
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from mri_seg.infer import predict_image
from mri_seg.utils import load_model, resolve_device
from mri_seg.visualize import make_overlay

st.set_page_config(page_title='Brain MRI Segmentation', page_icon='🧠', layout='wide')
st.title('Brain MRI FLAIR Abnormality Segmentation')
st.caption('Educational research demonstration — NOT for medical diagnosis, treatment, or clinical use.')
st.write('Upload an MRI slice in the same three-channel format as the TCGA-LGG dataset.')

checkpoint_path = st.text_input('Model checkpoint', 'outputs/unet/best.pt')
threshold = st.slider('Segmentation threshold', 0.05, 0.95, 0.50, 0.05)
upload = st.file_uploader('Upload MRI TIFF, PNG or JPEG', type=['tif', 'tiff', 'png', 'jpg', 'jpeg'])

@st.cache_resource(show_spinner='Loading trained model...')
def cached_model(path):
    device = resolve_device('auto')
    model, checkpoint = load_model(path, device)
    return model, checkpoint, device

if upload is not None:
    if not Path(checkpoint_path).is_file():
        st.error('No trained checkpoint found. First run `python -m mri_seg.train`, then enter its best.pt path.')
    else:
        try:
            model, checkpoint, device = cached_model(str(Path(checkpoint_path).resolve()))
            with Image.open(upload) as im:
                rgb, mask, _ = predict_image(model, checkpoint, im, device, threshold)
            a, b, c = st.columns(3)
            a.image(rgb, caption='Input MRI (three channels)', width="stretch")
            b.image(mask * 255, caption='Predicted binary mask', width="stretch", clamp=True)
            c.image(make_overlay(rgb, mask), caption='Overlay', width="stretch")
            st.caption(f"Model: {checkpoint['model_name']} | threshold: {threshold:.2f} | image size: {checkpoint['image_size']} px")
            if not np.any(mask):
                st.info('The model predicted no abnormal pixels at this threshold; this is not a medical finding.')
        except (RuntimeError, ValueError, OSError, KeyError) as exc:
            st.error(f'Unable to run inference: {exc}')
