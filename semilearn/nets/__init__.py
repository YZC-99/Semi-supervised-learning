# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from .densnet import densenet121
from .resnet import resnet50,resnet50_clinical
from .wrn import wrn_28_2, wrn_28_8, wrn_var_37_2
from .vit import vit_base_patch16_224, vit_small_patch16_224,deep_prompt_vit_base_patch16_224,deep_prompt_vit_small_patch16_224,shallow_prompt_vit_small_patch16_224,vit_small_patch2_32, vit_tiny_patch2_32, vit_base_patch16_96,vit_large_patch16_224
from .vit import deep_prompt_vit_large_patch16_224
from .bert import bert_base_cased, bert_base_uncased
from .wave2vecv2 import wave2vecv2_base
from .hubert import hubert_base
