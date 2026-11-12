# -*- coding: utf-8 -*-
import json, sys
# 用 Python 侧的 transformers 取真实 tokenizer 的 all_special_tokens 与 id
sys.path.insert(0, r'Python 环境/Lib/site-packages')
from transformers import AutoTokenizer
import os
os.environ['HF_HOME'] = r'外部素材目录/hf'
snap = r'外部素材目录/hf/hub/models--ku-nlp--deberta-v2-base-japanese-char-wwm/snapshots/29498bbcaa6e3220d9e74345cd6d963e7c00fee4'
t = AutoTokenizer.from_pretrained(snap)
print('class', type(t).__name__)
print('all_special_tokens', t.all_special_tokens)
print('all_special_ids', t.all_special_ids)
for p in ['bos','eos','sep','pad','cls','mask','unk']:
    print(' ', p, getattr(t, p+'_token', None), getattr(t, p+'_token_id', None))
