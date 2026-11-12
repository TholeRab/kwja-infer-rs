# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, r'Python 环境/Lib/site-packages')
from transformers import AutoTokenizer
os.environ['HF_HOME'] = r'外部素材目录/hf'
snap = r'外部素材目录/hf/hub/models--ku-nlp--deberta-v2-base-japanese-char-wwm/snapshots/29498bbcaa6e3220d9e74345cd6d963e7c00fee4'
t = AutoTokenizer.from_pretrained(snap)
special_ids = {getattr(t, f'{p}_token_id') for p in ['bos','eos','sep','pad','cls','mask']}
print('special_ids 原样 =', special_ids)
print('排序 =', sorted(special_ids, key=lambda x: (x is None, x)))
# 注意 bos/eos 是 None，会被算进集合
print('None in set:', None in special_ids)
print('用于过滤的集合大小 =', len(special_ids))
