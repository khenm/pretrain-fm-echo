"""
Copyright (c) 
EchoPrime: A Multi-Video View-Informed Vision-Language Model for Comprehensive Echocardiography Interpretation
Milos Vukadinovic, Xiu Tang, Neal Yuan, Paul Cheng, Debiao Li, Susan Cheng, Bryan He*, David Ouyang*
"""
# Standard library imports
import os
import math
import glob
import json
import pickle
import random
from src.utils.logging import get_logger

import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import pydicom
import sklearn
import sklearn.metrics
import transformers

import utils

logger = get_logger(__name__)

class EchoPrime:
    def __init__(self, device=None, lang='en'):
        """
        Initialize EchoPrime with a video encoder and view classifier.
        
        Args:
            device (str): Compute device ('cpu' or 'cuda')
            lang (str): Language ('en' for english, 'it' for italian, etc.)
        """
        utils.initialize_language(lang)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lang = lang

        self._init_echo_encoder()
        self._init_view_classifier()
        
        self.frames_to_take = 32
        self.frame_stride = 2
        self.video_size = 224
        
        self.mean = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
        self.std = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)

        self._load_metadata()

    def _init_echo_encoder(self):
        self.echo_encoder = torchvision.models.video.mvit_v2_s()
        self.echo_encoder.head[-1] = torch.nn.Linear(self.echo_encoder.head[-1].in_features, 512)
        
        checkpoint = torch.load("model_data/weights/echo_prime_encoder.pt", map_location=self.device)
        self.echo_encoder.load_state_dict(checkpoint)
        self.echo_encoder.eval().to(self.device)
        for param in self.echo_encoder.parameters():
            param.requires_grad = False

    def _init_view_classifier(self):
        self.view_classifier = torchvision.models.convnext_base()
        self.view_classifier.classifier[-1] = torch.nn.Linear(
            self.view_classifier.classifier[-1].in_features, 11
        )
        
        vc_state_dict = torch.load("model_data/weights/view_classifier.pt", map_location=self.device)
        self.view_classifier.load_state_dict(vc_state_dict)
        self.view_classifier.eval().to(self.device)
        for param in self.view_classifier.parameters():
            param.requires_grad = False

    def _load_metadata(self):
        self.MIL_weights = pd.read_csv("assets/MIL_weights.csv")
        self.non_empty_sections = self.MIL_weights['Section']
        self.section_weights = self.MIL_weights.iloc[:, 1:].to_numpy()
    
        self.candidate_studies = list(pd.read_csv("model_data/candidates_data/candidate_studies.csv")['Study'])
        candidate_embeddings_p1 = torch.load("model_data/candidates_data/candidate_embeddings_p1.pt", map_location='cpu')
        candidate_embeddings_p2 = torch.load("model_data/candidates_data/candidate_embeddings_p2.pt", map_location='cpu')
        
        self.candidate_embeddings = torch.cat((candidate_embeddings_p1, candidate_embeddings_p2), dim=0)
        
        candidate_reports = pd.read_pickle("model_data/candidates_data/candidate_reports.pkl")
        self.candidate_reports = [utils.phrase_decode(vec_phr) for vec_phr in tqdm(candidate_reports, desc="Decoding reports")]
        
        self.candidate_labels = pd.read_pickle("model_data/candidates_data/candidate_labels.pkl")
        self.section_to_phenotypes = pd.read_pickle("assets/section_to_phenotypes.pkl")

    def _pad_video(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """Pads the temporal dimension of a 4D video tensor if shorter than required."""
        current_frames = video_tensor.shape[1]
        if current_frames < self.frames_to_take:
            padding = torch.zeros(
                (3, self.frames_to_take - current_frames, self.video_size, self.video_size),
                dtype=torch.float
            )
            return torch.cat((video_tensor, padding), dim=1)
        return video_tensor

    def process_dicoms(self, directory_path: str) -> torch.Tensor:
        """
        Reads and preprocesses DICOM video data.
        
        Args:
            directory_path (str): Folder containing DICOM files.
            
        Returns:
            Tensor: Preprocessed videos (N, channels, frames, height, width).
        """
        dicom_paths = glob.glob(f'{directory_path}/**/*.dcm', recursive=True)
        processed_videos = []
        
        for dicom_path in tqdm(dicom_paths, desc="Processing DICOMs"):
            try:
                dcm = pydicom.dcmread(dicom_path)
                pixels = dcm.pixel_array
                
                if pixels.ndim < 3 or (pixels.ndim == 3 and pixels.shape[2] == 3 and pixels.shape[0] < self.frames_to_take):
                    pass # Handled differently based on downstream expectations or skip. 
                    # Original logic skips if it's purely a single frame RGB image.
                
                if pixels.ndim < 3 or pixels.shape[2] == 3:
                     continue 
                     
                if pixels.ndim == 3:
                    pixels = np.repeat(pixels[..., None], 3, axis=3)
                
                pixels = utils.mask_outside_ultrasound(pixels)
                
                processed_frames = np.zeros((len(pixels), self.video_size, self.video_size, 3))
                for i in range(len(processed_frames)):
                    processed_frames[i] = utils.crop_and_scale(pixels[i])
                
                video_tensor = torch.as_tensor(processed_frames, dtype=torch.float).permute([3, 0, 1, 2])
                video_tensor.sub_(self.mean).div_(self.std)
                video_tensor = self._pad_video(video_tensor)
                
                processed_videos.append(video_tensor[:, 0:self.frames_to_take:self.frame_stride, :, :])
                
            except Exception as e:
                logger.error(f"Failed to process {dicom_path}: {e}")

        return torch.stack(processed_videos) if processed_videos else torch.empty((0,))

    def process_mp4s(self, directory_path: str) -> torch.Tensor:
        """
        Reads and preprocesses MP4 video data.
        
        Args:
            directory_path (str): Folder containing MP4 files.
            
        Returns:
            Tensor: Preprocessed videos (N, channels, frames, height, width).
        """
        mp4_paths = glob.glob(f'{directory_path}/**/*.mp4', recursive=True)
        processed_videos = []
        
        for mp4_path in tqdm(mp4_paths, desc="Processing MP4s"):
            try:
                pixels, _, _ = torchvision.io.read_video(mp4_path)
                pixels = np.array(pixels)

                processed_frames = np.zeros((len(pixels), self.video_size, self.video_size, 3))
                for i in range(len(processed_frames)):
                    processed_frames[i] = utils.crop_and_scale(pixels[i])

                video_tensor = torch.as_tensor(processed_frames, dtype=torch.float).permute([3, 0, 1, 2])
                video_tensor.sub_(self.mean).div_(self.std)
                video_tensor = self._pad_video(video_tensor)

                processed_videos.append(video_tensor[:, 0:self.frames_to_take:self.frame_stride, :, :])

            except Exception as e:
                logger.error(f"Failed to process {mp4_path}: {e}")

        return torch.stack(processed_videos) if processed_videos else torch.empty((0,))

    def embed_videos(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        Embeds a batch of preprocessed videos into the EchoPrime latent space.
        
        Args:
            video_batch (Tensor): Videos shaped (N, channels, frames, height, width).
            
        Returns:
            Tensor: Latent embeddings (N, hidden_dim).
        """
        bin_size = 50
        n_bins = math.ceil(video_batch.shape[0] / bin_size)
        features_list = []
        
        with torch.no_grad():
            for bin_idx in range(n_bins):
                start_idx = bin_idx * bin_size
                end_idx = min((bin_idx + 1) * bin_size, video_batch.shape[0])
                batch_segment = video_batch[start_idx:end_idx].to(self.device)
                
                features = self.echo_encoder(batch_segment)
                features_list.append(features)
                
        return torch.cat(features_list, dim=0)

    def get_views(self, video_batch: torch.Tensor, visualize: bool = False, return_labels: bool = False) -> torch.Tensor:
        """
        Classifies echocardiogram views for a batch of videos.
        
        Args:
            video_batch (Tensor): Preprocessed videos.
            visualize (bool): Render debugging plot of views.
            return_labels (bool): Return string labels instead of one-hot tensors.
            
        Returns:
            Tensor | list: View probability tensors or string labels.
        """
        first_frames = video_batch[:, :, 0, :, :].to(self.device)
        
        with torch.no_grad():
            logits = self.view_classifier(first_frames)
            
        predicted_classes = torch.argmax(logits, dim=1)
        view_labels = [utils.COARSE_VIEWS[v] for v in predicted_classes]
        view_encodings = torch.nn.functional.one_hot(predicted_classes, num_classes=11).float()

        if visualize:
            self._visualize_views(first_frames, view_labels)

        if return_labels:
            return view_labels
            
        return view_encodings

    def _visualize_views(self, frames: torch.Tensor, labels: list[str]):
        """Helper to plot video frames alongside inferred view labels."""
        rows = math.ceil(len(labels) / 12)
        cols = 12
        fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
        axes = axes.flatten()
        
        for i, label in enumerate(labels):
            display_image = (frames[i].cpu().permute([1, 2, 0]) * 255).numpy()
            display_image = np.clip(display_image, 0, 255).astype('uint8')
            display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
            cv2.putText(display_image, label.replace("_", " "), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
            axes[i].imshow(display_image)
            axes[i].axis('off')

        for j in range(len(labels), len(axes)):
            axes[j].axis('off')
            
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()

    @torch.no_grad()
    def encode_study(self, video_batch: torch.Tensor, visualize: bool = False) -> torch.Tensor:
        """
        Encodes an entire study by combining visual features with classified views.
        
        Args:
            video_batch (Tensor): Preprocessed study videos.
            
        Returns:
            Tensor: Fused study representation combining visual and contextual features.
        """
        visual_features = self.embed_videos(video_batch)
        view_features = self.get_views(video_batch, visualize=visualize)
        return torch.cat((visual_features, view_features), dim=1)
    
    def translate_sections(self, report: str) -> str:
        translations = {}

        if self.lang == 'it':
            translations = {
                "Left Ventricle": "Ventricolo Sinistro",
                "Resting Segmental Wall Motion Analysis": "Cinetica Segmentaria a Riposo",
                "Right Ventricle": "Ventricolo Destro",
                "Left Atrium": "Atrio Sinistro",
                "Right Atrium": "Atrio Destro",
                "Atrial Septum": "Setto Inter-Atriale",
                "Mitral Valve": "Valvola Mitrale",
                "Aortic Valve": "Valvola Aortica",
                "Tricuspid Valve": "Valvola Tricuspide",
                "Pulmonic Valve": "Valvola Polmonare",
                "Pericardium": "Pericardio",
                "Aorta": "Aorta",
                "IVC": "Vena Cava Inferiore",
                "Pulmonary Artery": "Arteria Polmonare",
                "Pulmonary Veins": "Vene Polmonari",
                "Postoperative Findings": "Esiti Post-Operatori",
            }
        elif self.lang == 'bs':
            translations = {
                "Left Ventricle": "Lijeva komora",
                "Resting Segmental Wall Motion Analysis": "Analiza segmentalne pokretljivosti stijenke u mirovanju",
                "Right Ventricle": "Desna komora",
                "Left Atrium": "Lijeva pretkomora",
                "Right Atrium": "Desna pretkomora",
                "Atrial Septum": "Interatrijski septum",
                "Mitral Valve": "Mitralni zalisak",
                "Aortic Valve": "Aortni zalisak",
                "Tricuspid Valve": "Trikuspidalni zalisak",
                "Pulmonic Valve": "Pulmonalni zalisak",
                "Pericardium": "Perikard",
                "Aorta": "Aorta",
                "IVC": "Donja šuplja vena",
                "Pulmonary Artery": "Plućna arterija",
                "Pulmonary Veins": "Plućne vene",
                "Postoperative Findings": "Postoperativni nalazi",
            }

        for section, translated in translations.items():
            report = report.replace(section, translated)
        
        return report

    def generate_report(self, study_embedding: torch.Tensor) -> str:
        """
        Generates a text report mapping study embeddings to candidate report clauses.
        
        Args:
            study_embedding (Tensor): Feature array of shape (num_videos, 523).
            
        Returns:
            str: Generated medical report text.
        """
        study_embedding = study_embedding.cpu()
        generated_report = ""
        
        for section_idx, section_name in enumerate(self.non_empty_sections):
            active_weights = [
                self.section_weights[section_idx][torch.where(view == 1)[0]] 
                for view in study_embedding[:, 512:]
            ]
            
            weighted_embedding = study_embedding[:, :512] * torch.tensor(active_weights, dtype=torch.float).unsqueeze(1)
            averaged_embedding = torch.mean(weighted_embedding, dim=0)
            normalized_embedding = torch.nn.functional.normalize(averaged_embedding, dim=0)
            
            similarities = normalized_embedding @ self.candidate_embeddings.T
            extracted_section = "Section not found."
            
            while extracted_section == "Section not found.":
                best_match_idx = torch.argmax(similarities)
                predicted_text = self.candidate_reports[best_match_idx]
                extracted_section = utils.extract_section(predicted_text, section_name)
                
                if extracted_section != "Section not found.":
                    generated_report += extracted_section
                    
                similarities[best_match_idx] = float('-inf')

        if self.lang != 'en':
            generated_report = self.translate_sections(generated_report)
                
        return generated_report
    
    def predict_metrics(self, study_embedding: torch.Tensor, k: int = 50) -> dict:
        """
        Predicts continuous diagnostic phenotypes from study embeddings.
        
        Args:
            study_embedding (Tensor): Extracted video embeddings.
            k (int): Number of top candidate neighbors to average.
            
        Returns:
            dict: Phenotype metric names mapped to continuous predicted values.
        """
        num_sections = len(self.non_empty_sections)
        section_embeddings = torch.zeros(num_sections, 512)
        study_embedding = study_embedding.cpu()
        
        for section_idx, section_name in enumerate(self.non_empty_sections):
            active_weights = [
                self.section_weights[section_idx][torch.where(view == 1)[0]]
                for view in study_embedding[:, 512:]
            ]
            weighted_features = study_embedding[:, :512] * torch.tensor(active_weights, dtype=torch.float).unsqueeze(1)
            section_embeddings[section_idx] = torch.sum(weighted_features, dim=0)
            
        section_embeddings = torch.nn.functional.normalize(section_embeddings)
        similarities = section_embeddings @ self.candidate_embeddings.T
        top_indices = torch.topk(similarities, k=k, dim=1).indices
        
        predictions = {}
        for section_idx, section_name in enumerate(self.section_to_phenotypes.keys()):
            for phenotype in self.section_to_phenotypes[section_name]:
                neighbor_values = [
                    self.candidate_labels[phenotype][self.candidate_studies[idx]]
                    for idx in top_indices[section_idx]
                    if self.candidate_studies[idx] in self.candidate_labels[phenotype]
                ]
                predictions[phenotype] = np.nanmean(neighbor_values) if neighbor_values else np.nan
        
        return predictions

class EchoPrimeTextEncoder(torch.nn.Module):
    """Encodes text reports into the joint embedding space aligned with EchoPrime videos."""
    
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
        
        config = transformers.AutoConfig.from_pretrained(model_name)
        self.backbone = transformers.AutoModelForMaskedLM.from_config(config)
        self.text_projection = torch.nn.Linear(768, 512)
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.max_length = 512
        
        self.to(self.device)

    def forward(self, report: str) -> torch.Tensor:
        """
        Tokenizes and embeds an input report text.
        
        Args:
            report (str): Raw report text.
            
        Returns:
            Tensor: Single embedding sequence representation (1, 512).
        """
        encoded_input = self.tokenizer(
            report,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        
        if encoded_input["input_ids"].shape[1] > 512:
            encoded_input = self._truncate_long_report(encoded_input)

        with torch.no_grad():
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            outputs = self.backbone(**encoded_input, output_hidden_states=True)
            sequence_features = outputs.hidden_states[-1][:, 0, :]
            embeddings = self.text_projection(sequence_features)
            
        return embeddings

    def _truncate_long_report(self, encoded_input: dict) -> transformers.BatchEncoding:
        """Selects a sub-segment of tokens preserving internal separators if input exceeds 512."""
        sep_positions = list(torch.where(encoded_input["input_ids"].squeeze(0) == self.tokenizer.sep_token_id)[0].numpy())
        max_start = sep_positions[-1] - 512 if sep_positions else 0
        
        possible_starts = [0] + [pos for pos in sep_positions if pos < max_start]
        start_idx = possible_starts[random.randint(0, max(0, len(possible_starts) - 1))]
        
        max_end = start_idx + 512
        end_idx = max_end
        
        for pos in reversed(sep_positions):
            if pos <= max_end:
                end_idx = pos
                break
                
        return transformers.BatchEncoding(data={k: v[:, start_idx:end_idx] for k, v in encoded_input.items()})
