import torch
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


# TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs["VQ_Configs"])

        self.num_image_tokens = configs["num_image_tokens"]
        self.mask_token_id = configs["num_codebook_vectors"]
        self.choice_temperature = configs["choice_temperature"]
        self.gamma = self.gamma_func(configs["gamma_type"])
        self.transformer = BidirectionalTransformer(configs["Transformer_param"])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs["VQ_config_path"], "r"))
        model = VQGAN(cfg["model_param"])
        model.load_state_dict(torch.load(configs["VQ_CKPT_path"]), strict=True)
        model = model.eval()
        return model

    ##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        self.vqgan.eval()
        zq, z_indices, _ = self.vqgan.encode(x)
        B = zq.size(0)
        z_indices = z_indices.view(B, -1)
        return zq, z_indices

    ##TODO2 step1-2:
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1].
        During training, the input ratio is uniformly sampled;
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.

        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":

            def linear_gamma(ratio):
                return 1 - ratio

            return linear_gamma
        elif mode == "cosine":

            def cosine_gamma(ratio):
                return math.cos((ratio * math.pi) / 2)

            return cosine_gamma
        elif mode == "square":

            def square_gamma(ratio):
                return 1 - ratio**2

            return square_gamma
        else:
            raise ValueError(f"Unsupported gamma type: {mode}")

    ##TODO2 step1-3:
    def forward(self, x):
        _, z_indices = self.encode_to_z(x)  # (B, 256) — ground truth indices
        # Create masked version with per-sample mask ratio
        masked_indices = z_indices.clone()
        B = z_indices.size(0)
        for i in range(B):
            r = torch.rand(1).item()
            mask_rate = self.gamma(r)
            num_mask = max(1, math.ceil(mask_rate * self.num_image_tokens))
            mask_pos = torch.randperm(self.num_image_tokens)[:num_mask]
            masked_indices[i, mask_pos] = self.mask_token_id  # 1024

        logits = self.transformer(masked_indices)  # (B, 256, 1025)
        return logits, z_indices

    ##TODO3 step1-1: define one iteration decoding
    @torch.no_grad()
    def inpainting(self, z_indices, mask_bc, mask_b, ratio, mask_num):
        # z_indices: (B, 256) current tokens — originals at unmasked positions, mask_token_id at masked
        # mask_bc: (B, 256) current boolean mask — True = still masked (shrinks each iteration)
        # mask_b:  (B, 256) original boolean mask — True = needs prediction (fixed)
        # ratio:   current step ratio (step+1)/total_iter
        # mask_num: total number of originally masked tokens

        logits = self.transformer(z_indices)
        logits = torch.softmax(logits, dim=-1)

        # Find MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = logits.max(dim=-1)

        # Predicted probabilities add temperature annealing gumbel noise as confidence
        g = -torch.log(-torch.log(torch.rand_like(z_indices_predict_prob)))
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g

        # If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        confidence[~mask_b] = float("inf")

        # Sort the confidence for the rank
        # Define how much the iteration remain predicted tokens by mask scheduling
        n = math.ceil(self.gamma(ratio) * mask_num)

        # lowest confidence positions get re-masked
        _, sorted_indices = confidence.sort(dim=-1)
        mask_bc = torch.zeros_like(mask_b)
        mask_bc.scatter_(1, sorted_indices[:, :n], True)

        # Where still masked: replace prediction with mask_token_id
        z_indices_predict[mask_bc] = self.mask_token_id

        # At the end of the decoding process, add back the original(non-masked) token values
        z_indices_predict[~mask_b] = z_indices[~mask_b]

        return z_indices_predict, mask_bc


__MODEL_TYPE__ = {"MaskGit": MaskGit}
