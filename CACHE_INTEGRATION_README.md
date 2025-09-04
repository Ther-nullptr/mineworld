# MineWorld KV Cache Management Integration

This document describes the integration of H2O and Streaming-LLM algorithms into MineWorld for efficient KV cache management.

## Overview

The integration provides two advanced KV cache management strategies:

1. **H2O (Heavy-Hitter Oracle)**: Identifies and preserves tokens with the highest attention scores (heavy hitters) while maintaining recent tokens.
2. **Streaming-LLM**: Maintains a fixed-size cache containing initial tokens (start) and recent tokens, enabling efficient streaming inference.

## Features

### Cache Management Strategies

#### 1. H2O Algorithm
- **Principle**: Preserves tokens that contribute most to attention computation
- **Key Parameters**:
  - `hh_ratio`: Ratio of prompt length to determine heavy hitter count (default: 0.1)
  - Dynamic attention score tracking and aggregation
  - Automatic eviction of least valuable tokens

#### 2. Streaming-LLM Algorithm
- **Principle**: Maintains initial tokens and recent tokens in a sliding window
- **Key Parameters**:
  - `start_size`: Number of initial tokens to preserve (default: 4)
  - `recent_size`: Number of recent tokens to preserve (default: 512)

## Usage

### Command Line Interface

#### For Inference (`inference.py`)

```bash
# Use H2O cache management
python inference.py \
    --data_root /path/to/data \
    --model_ckpt /path/to/model.pt \
    --config /path/to/config.yaml \
    --output_dir /path/to/output \
    --cache-strategy h2o \
    --hh-ratio 0.1 \
    --frames 100 \
    --demo_num 10

# Use Streaming-LLM cache management
python inference.py \
    --data_root /path/to/data \
    --model_ckpt /path/to/model.pt \
    --config /path/to/config.yaml \
    --output_dir /path/to/output \
    --cache-strategy streaming \
    --start-size 4 \
    --recent-size 512 \
    --frames 100 \
    --demo_num 10

# Disable cache management (default)
python inference.py \
    --data_root /path/to/data \
    --model_ckpt /path/to/model.pt \
    --config /path/to/config.yaml \
    --output_dir /path/to/output \
    --cache-strategy none \
    --frames 100 \
    --demo_num 10
```

#### For Interactive Interface (`mineworld.py`)

```bash
# Use H2O cache management
python mineworld.py \
    --scene /path/to/scene.mp4 \
    --model_ckpt /path/to/model.pt \
    --config /path/to/config.yaml \
    --cache-strategy h2o \
    --hh-ratio 0.1

# Use Streaming-LLM cache management
python mineworld.py \
    --scene /path/to/scene.mp4 \
    --model_ckpt /path/to/model.pt \
    --config /path/to/config.yaml \
    --cache-strategy streaming \
    --start-size 4 \
    --recent-size 512

# Disable cache management (default)
python mineworld.py \
    --scene /path/to/scene.mp4 \
    --model_ckpt /path/to/model.pt \
    --config /path/to/config.yaml \
    --cache-strategy none
```

### Parameter Descriptions

| Parameter | Description | Default Value | Valid Range |
|-----------|-------------|---------------|-------------|
| `--cache-strategy` | Cache management strategy | `none` | `none`, `streaming`, `h2o` |
| `--hh-ratio` | H2O heavy hitter ratio | `0.1` | `0.0 - 1.0` |
| `--start-size` | Streaming-LLM start tokens | `4` | `≥ 1` |
| `--recent-size` | Streaming-LLM recent tokens | `512` | `≥ 1` |

## Implementation Details

### Architecture

The integration consists of three main components:

1. **`kv_cache_manager.py`**: Core cache management implementations
2. **`lvm.py`**: Modified attention layer with cache management support
3. **`inference.py`** & **`mineworld.py`**: Command line interface integration

### Key Classes

#### `H2OKVCache`
- Implements heavy hitter identification and preservation
- Dynamic attention score tracking
- Automatic cache eviction based on token importance

#### `StreamingKVCache`
- Wrapper around Streaming-LLM's `StartRecentKVCache`
- Maintains start and recent token windows
- Efficient sliding window management

#### `KVCacheManager`
- Unified interface for different cache strategies
- Automatic strategy selection and parameter management
- Integration with MineWorld's transformer layers

### Performance Considerations

#### H2O Benefits
- **Memory Efficiency**: Reduces cache size by preserving only important tokens
- **Performance**: Maintains model quality while using less memory
- **Adaptability**: Dynamically adjusts to token importance during generation

#### Streaming-LLM Benefits
- **Deterministic Behavior**: Fixed cache size provides predictable memory usage
- **Simplicity**: Easy to understand and configure
- **Compatibility**: Works well with streaming inference scenarios

#### Memory Usage Comparison

| Strategy | Cache Size | Memory Usage | Quality Impact |
|----------|------------|--------------|---------------|
| `none` | Full context | High | None (baseline) |
| `streaming` | `start_size + recent_size` | Low | Minimal for recent contexts |
| `h2o` | Dynamic based on `hh_ratio` | Medium | Minimal for most tasks |

## Integration Examples

### Programmatic Usage

```python
from kv_cache_manager import KVCacheManager

# Setup H2O cache manager
cache_manager = KVCacheManager(
    strategy="h2o",
    hh_ratio=0.1,
    prompt_len=1000
)

# Setup Streaming-LLM cache manager
cache_manager = KVCacheManager(
    strategy="streaming",
    start_size=4,
    recent_size=512
)

# Apply to model
model.transformer.setup_cache_manager("h2o", hh_ratio=0.1)
```

### Custom Cache Strategies

```python
# Implement custom cache strategy
class CustomKVCache:
    def __init__(self, **kwargs):
        # Initialize custom parameters
        pass
    
    def __call__(self, past_key_values):
        # Implement custom cache logic
        return processed_cache
    
    def evict_for_space(self, past_key_values, num_coming):
        # Implement custom eviction logic
        return evicted_cache
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce cache sizes (`hh_ratio`, `recent_size`)
2. **Quality Degradation**: Increase cache sizes or try different strategy
3. **Performance Issues**: Monitor cache hit rates and adjust parameters

### Debug Mode

Enable debug output by setting environment variable:
```bash
export DEBUG_CACHE=1
```

## Future Enhancements

1. **Adaptive Parameters**: Automatic parameter tuning based on input characteristics
2. **Hybrid Strategies**: Combination of H2O and Streaming-LLM approaches
3. **Memory Monitoring**: Real-time cache usage statistics
4. **Quality Metrics**: Quantitative quality assessment for different strategies

## References

- H2O Paper: [Heavy-Hitter Oracle: Efficient Long-Context Inference for LLMs](https://arxiv.org/abs/2306.14048)
- Streaming-LLM Paper: [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)

## License

This integration follows the same license as the original MineWorld project.
