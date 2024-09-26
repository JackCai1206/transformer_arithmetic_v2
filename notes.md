# LG Inheritance

## Addition
- Standardize amount of compute
- redo previous ablations (!)
  - hope is that max LG is lower for the partial ones
  - doing 5 runs, and try to conclude from there (WIP)
- Result noisy is kinda confirmed, how to stabilize? 
  - random padding 5 runs (WIP)
- train on composed task, and test on LG for individual tasks
- compare with abacus (WIP)

## Smaller tasks
- rotate-reverse
  - verify this works

## Scaling studies
- Given a min length, whats the max length that can be generalized to? 
  - poor performance for 4x and 8x
    - max length needs to scale? (64 --> 16)
    - random padding?
    - partial auxillary tasks? 
    - right padding? (WIP)
- Vary only the model size and keep compute the same
- Sample complexity

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
