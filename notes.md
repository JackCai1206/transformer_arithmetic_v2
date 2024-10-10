# LG Inheritance

## Addition
1. RoPE > other pos encodings
  - Test NoPE
  - Test LPE
2. Ablations 
  - Train on just 1 aux task
3. Custom LR scheduler
  1e-3, then 5e-4, then decay
4. task ratio scheduling

## Smaller tasks
- rotate-reverse
  - verify this works

## Scaling studies
- Given a min length, whats the max length that can be generalized to? 
  - poor performance for 4x and 8x
    - Conclusion is that max_len must be 2 * min_len
    - small + 20k steps
    - Retry with same data seed across tasks
  - have to try dropping attention masks
- Vary only the model size and keep compute the same
- Sample complexity - reduce the number of samples for main task
  - There could be better ways to sample - right now its just random
    - sapmle only 10-16 digits, only 16

## COT "Chain-length generalization"
- Show that TF can auto-compose the two reasoning steps
  - maybe even have length generalization inheritance from one task to another
- Use LEGO tasks: reasoning length generalization

## Why models cannot length generalize in normal settings
- they cannot discover relative positional information? 
- they cannot learn to use OOD positional information? 

## "Why" does this happen
- because position encoding is trained? 

## Other
- backtrack token
- top p/non greedy decoding/supress eos
