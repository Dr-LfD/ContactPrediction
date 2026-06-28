import os
from typing import Dict, Iterable

import numpy as np
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def get_text_embedding_cache_path(cache_dir: str, model_name: str) -> str:
    return os.path.join(
        cache_dir,
        f"{_sanitize_model_name(model_name)}_text_embeddings.npy",
    )


def _load_cached_embeddings(cache_path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(cache_path):
        return {}

    cached = np.load(cache_path, allow_pickle=True).item()
    return {
        key: np.asarray(value, dtype=np.float32)
        for key, value in cached.items()
    }


def _compute_text_embeddings(
    texts,
    cache_dir: str,
    model_name: str,
    max_length: int,
) -> Dict[str, np.ndarray]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Text embeddings require the sdp_dmg environment with transformers installed."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    model.eval()

    embeddings = {}
    with torch.no_grad():
        for text in texts:
            tokens = tokenizer(
                text=text,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            embedding = model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )["pooler_output"][0]
            embeddings[text] = embedding.detach().cpu().numpy().astype(np.float32)
    return embeddings


def load_or_create_text_embeddings(
    texts: Iterable[str],
    cache_dir: str,
    model_name: str = "bert-base-cased",
    max_length: int = 25,
) -> Dict[str, np.ndarray]:
    unique_texts = list(dict.fromkeys(texts))
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = get_text_embedding_cache_path(cache_dir, model_name)
    cached_embeddings = _load_cached_embeddings(cache_path)

    missing_texts = [text for text in unique_texts if text not in cached_embeddings]
    if missing_texts:
        cached_embeddings.update(
            _compute_text_embeddings(
                texts=missing_texts,
                cache_dir=cache_dir,
                model_name=model_name,
                max_length=max_length,
            )
        )
        np.save(cache_path, cached_embeddings, allow_pickle=True)

    return {
        text: np.asarray(cached_embeddings[text], dtype=np.float32)
        for text in unique_texts
    }
