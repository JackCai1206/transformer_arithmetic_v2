## Addition
- redo previous ablations (!)
- train on composed task, and test on LG for individual tasks

## Smaller tasks
- rotate-reverse
  - verify this works
  - grid search max-len vs min-len

## COT "Chain-length generalization"
- Show that TF can auto-compose the two reasoning steps
  - maybe even have length generalization inheritance from one task to another

## Why models cannot length generalize in normal settings
- they cannot discover relative positional information? 
- they cannot learn to use OOD positional information? 

## "Why" does this happen
- because position encoding is trained? 

## Other
- backtrack token
- top p/non greedy decoding/supress eos
