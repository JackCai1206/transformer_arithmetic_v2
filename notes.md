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
- Vary only the model size and keep compute the same
- Sample complexity
  - Compare multi-task vs single task full length
    - Do we limit the sample size of aux tasks?
  - There could be better ways to sample - right now its just random
    - sapmle only 10-16 digits, only 16
- Number of tokens seen (efficiency)
  - Compare with directly training on full length (probably is more)
  - What is the minimum tokens seen to learn the task? 
- Task similarity
  - Interleave copy, xor, a_i == b_i
  - add 1
  - Alternative format: map to characters
  - Add characters to the task id

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
