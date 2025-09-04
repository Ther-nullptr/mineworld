import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
from streaming_llm.kv_cache import StartRecentKVCache


class H2OKVCache:
    """
    H2O (Heavy-Hitter Oracle) KV Cache implementation for MineWorld
    """
    def __init__(
        self,
        hh_ratio: float = 0.1,
        k_seq_dim: int = 2,
        v_seq_dim: int = 2,
        device: str = "cuda"
    ):
        self.hh_ratio = hh_ratio
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.device = device
        self.attention_scores = None
        self.hh_size = 0
        self.recent_size = 0
        self.total_cache_size = 0
        
    def initialize_cache_sizes(self, prompt_len: int):
        """Initialize cache sizes based on prompt length"""
        self.hh_size = max(1, int(prompt_len * self.hh_ratio))
        self.recent_size = self.hh_size - 1
        self.total_cache_size = self.hh_size + self.recent_size
        
    def update_attention_scores(self, attn_weights: torch.Tensor):
        """Update attention scores for heavy hitter identification"""
        if self.attention_scores is None:
            self.attention_scores = attn_weights.sum(dim=1)  # Aggregate across heads
        else:
            self.attention_scores += attn_weights.sum(dim=1)
    
    def get_heavy_hitters(self, k_cache: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Identify heavy hitter tokens based on attention scores"""
        if self.attention_scores is None or seq_len <= self.total_cache_size:
            return k_cache
            
        # Get top heavy hitters (excluding recent tokens)
        recent_start = max(0, seq_len - self.recent_size)
        eligible_scores = self.attention_scores[:recent_start]
        
        if eligible_scores.numel() == 0:
            return k_cache
            
        # Get top-k heavy hitters
        k_hh = min(self.hh_size, eligible_scores.numel())
        _, topk_indices = eligible_scores.topk(k_hh)
        
        # Extract heavy hitter keys and values
        hh_k = k_cache[:, topk_indices]
        recent_k = k_cache[:, recent_start:]
        
        # Combine heavy hitters with recent tokens
        return torch.cat([hh_k, recent_k], dim=1)
    
    def __call__(self, past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Apply H2O cache management"""
        if past_key_values is None:
            return None
            
        k, v = past_key_values
        seq_len = k.size(self.k_seq_dim)
        
        if seq_len <= self.total_cache_size:
            return (k, v)
        else:
            # Apply H2O strategy
            k_h2o = self.get_heavy_hitters(k, seq_len)
            v_h2o = self.get_heavy_hitters(v, seq_len)
            return (k_h2o, v_h2o)
    
    def evict_for_space(self, past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]], num_coming: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Evict tokens to make space for new tokens"""
        if past_key_values is None:
            return None
            
        k, v = past_key_values
        seq_len = k.size(self.k_seq_dim)
        
        if seq_len + num_coming <= self.total_cache_size:
            return (k, v)
        else:
            # Need to evict: keep heavy hitters and most recent tokens
            available_space = self.total_cache_size - num_coming
            keep_recent = min(self.recent_size, available_space)
            keep_hh = max(0, available_space - keep_recent)
            
            if keep_hh > 0:
                hh_k = self.get_heavy_hitters(k[:, :-self.recent_size], seq_len - self.recent_size)[:, :keep_hh]
                recent_k = k[:, -keep_recent:]
                k_evicted = torch.cat([hh_k, recent_k], dim=1)
                
                hh_v = self.get_heavy_hitters(v[:, :-self.recent_size], seq_len - self.recent_size)[:, :keep_hh]
                recent_v = v[:, -keep_recent:]
                v_evicted = torch.cat([hh_v, recent_v], dim=1)
            else:
                k_evicted = k[:, -keep_recent:]
                v_evicted = v[:, -keep_recent:]
                
            return (k_evicted, v_evicted)


class StreamingKVCache(StartRecentKVCache):
    """
    Streaming-LLM KV Cache wrapper for MineWorld
    """
    def __init__(self, start_size: int = 4, recent_size: int = 512, **kwargs):
        super().__init__(start_size=start_size, recent_size=recent_size, **kwargs)


class AdaptiveKVCache:
    """
    Adaptive KV Cache that can switch between different strategies
    """
    def __init__(self, strategy: str = "streaming", **kwargs):
        self.strategy = strategy
        self.kwargs = kwargs
        
        if strategy == "h2o":
            # H2OKVCache only needs specific parameters
            h2o_kwargs = {k: v for k, v in kwargs.items() 
                         if k in ['hh_ratio', 'k_seq_dim', 'v_seq_dim', 'device']}
            self.cache = H2OKVCache(**h2o_kwargs)
        elif strategy == "streaming":
            self.cache = StreamingKVCache(**kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def __call__(self, past_key_values):
        return self.cache(past_key_values)
    
    def evict_for_space(self, past_key_values, num_coming):
        if hasattr(self.cache, 'evict_for_space'):
            return self.cache.evict_for_space(past_key_values, num_coming)
        return past_key_values
    
    def update_attention_scores(self, attn_weights):
        if hasattr(self.cache, 'update_attention_scores'):
            return self.cache.update_attention_scores(attn_weights)


class KVCacheManager:
    """
    Main KV Cache Manager for MineWorld with multiple strategies
    """
    def __init__(self, strategy: str = "none", **kwargs):
        self.strategy = strategy
        self.cache = None
        self.attention_scores = []
        
        if strategy != "none":
            self.cache = AdaptiveKVCache(strategy=strategy, **kwargs)
    
    def apply_cache_management(self, past_key_values, attention_weights=None):
        """Apply cache management strategy"""
        if self.cache is None:
            return past_key_values
            
        if attention_weights is not None and self.strategy == "h2o":
            self.cache.update_attention_scores(attention_weights)
            
        return self.cache(past_key_values)
    
    def evict_for_space(self, past_key_values, num_coming):
        """Evict tokens to make space"""
        if self.cache is None:
            return past_key_values
            
        return self.cache.evict_for_space(past_key_values, num_coming)
    
    def initialize_for_prompt(self, prompt_len: int):
        """Initialize cache sizes based on prompt length"""
        if self.cache is not None and hasattr(self.cache, 'initialize_cache_sizes'):
            self.cache.initialize_cache_sizes(prompt_len)