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
    - actually, it seems to work on l=16,64, so max length probably plays a role
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
  - keep input format the same
    - pair-wise operations
      - pair-wise addition (with bias)
      - a_i == b_i, a_i != b_i, (a>=5) xor (b>=5), etc
    - carry chain operations
      - carry identification
      - generate and propagate identification
      - running sum
      - modified carry rule (e.g. a_i + b_i + c_i >= 2)
    - add 1 (at any position)
  - Change input format
    - swap numbers to characters

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
