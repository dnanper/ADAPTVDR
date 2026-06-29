from transformers import PaliGemmaForConditionalGeneration, PaliGemmaConfig
import torch, warnings; warnings.filterwarnings('ignore')

path = '/data2/cmdir/home/test01/longvnu/stable_diff/models/vidore/colpaligemma-3b-mix-448-base'
cfg = PaliGemmaConfig.from_pretrained(path)
cfg.architectures = ['PaliGemmaForConditionalGeneration']

model = PaliGemmaForConditionalGeneration.from_pretrained(
    path, config=cfg, ignore_mismatched_sizes=True, torch_dtype=torch.bfloat16
)
print('Loaded OK. hidden_size:', cfg.text_config.hidden_size)
print('Model type:', type(model).__name__)