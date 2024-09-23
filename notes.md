# LG Inheritance

## Addition
- Standardize amount of compute
- redo previous ablations (!)
  - running 10k steps makes SDA perform equally well
  - run 5k steps? 
  - sample complexity? 
- train on composed task, and test on LG for individual tasks

## Smaller tasks
- rotate-reverse
  - verify this works

## Scaling studies
- Given a min length, whats the max length that can be generalized to? 
- Vary only the model size and keep compute the same

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
