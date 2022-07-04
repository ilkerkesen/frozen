from xml.sax.handler import property_declaration_handler
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from .custom_opt import CustomOPTCausalLM


class OPTCaptioningModel(nn.Module):
    def __init__(self, config=dict()):
        super().__init__()
        self.config = config
        self.init_text_encoder()
        self.init_image_encoder()

        if self.frozen_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            
        if self.frozen_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        
        self.add_image_token()

    @property
    def d_text_encoder(self):
        return self.text_encoder.config.hidden_size
    
    @property
    def d_image_encoder(self):
        return self.image_encoder.config.hidden_sizes[-1]
    
    @property
    def device(self):
        return self.text_encoder.get_input_embeddings().weight.device

    @property
    def frozen_text_encoder(self):
        return self.config.get('frozen_text_encoder', False)
    
    @property
    def frozen_image_encoder(self):
        return self.config.get('frozen_image_encoder', False)

    def add_image_token(self):
        num_embeddings = self.text_encoder.get_input_embeddings().num_embeddings
        self.text_encoder.resize_token_embeddings(num_embeddings+1)

    def init_text_encoder(self):
        architecture = self.config.get('text_encoder', 'facebook/opt-1.3b')
        pretrained = self.config.get('pretrained_text_encoder', True)

        if pretrained:
            text_encoder = CustomOPTCausalLM.from_pretrained(architecture)
        else:
            config = AutoConfig.from_pretrained(architecture)
            text_encoder = CustomOPTCausalLM(config)

        self.text_encoder = text_encoder

    def init_image_encoder(self):
        architecture = self.config.get('image_encoder', 'microsoft/resnet-50')
        num_image_tokens = self.config.get('num_image_tokens', 2)

        self.image_encoder = AutoModel.from_pretrained(architecture)
        self.proj_image_features = nn.Linear(
            in_features=self.d_image_encoder,
            out_features=num_image_tokens * self.d_text_encoder,
        )
    
    def encode_images(self, pixel_values):
        batch_size = pixel_values.shape[0]
        visual = self.image_encoder(pixel_values).pooler_output
        visual = visual.reshape(batch_size, self.d_image_encoder)
        visual = self.proj_image_features(visual)
        return visual

    def forward(self, *args, **kwargs):
        if 'pixel_values' in kwargs:
            pixel_values = kwargs.pop('pixel_values')
            kwargs['image_features'] = self.encode_images(pixel_values)
        return self.text_encoder.forward(*args, **kwargs,
            return_dict=True)
        
    def generate(self, *args, **kwargs):
        if 'pixel_values' in kwargs:
            pixel_values = kwargs.pop('pixel_values')
            kwargs['image_features'] = self.encode_images(pixel_values)
        return self.text_encoder.generate(*args, **kwargs,
            return_dict_in_generate=True)