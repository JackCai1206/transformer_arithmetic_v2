# LG Inheritance

## Addition
- Standardize amount of compute
  - try 20k steps for small, right now might be undertrained
- Redo NoPE
  - does not work, so can claim that RoPE is special
- redo previous ablations (!)
  - hope is that max LG is lower for the partial ones
  - doing 5 runs, and try to conclude from there
- Result noisy is kinda confirmed, how to stabilize? 
  - Retry with same data seed across tasks
  - NanoGPT + cosin lr sweep + small batch
- train on composed task, and test on LG for individual tasks
  - Does not work well/ more noisy
- compare with abacus 
  - should work if I just train longer?

## Smaller tasks
- rotate-reverse
  - verify this works

## Scaling studies
- Given a min length, whats the max length that can be generalized to? 
  - poor performance for 4x and 8x
    - Conclusion is that max_len must be 2 * min_len
    - small + 20k steps
    - Retry with same data seed across tasks (WIP ida 1)
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
