## Addition
- Abacus long run?
  - abacus sanity run
  - May be a bug with inference code
- How short can composed task be? 
- Is abacus optimization hyper params actually good?
  - Loss less spikes? 
  - Use this for now, no big diff
- No task IDs? 
  - Doesn't work
- forward addition
- force composed task id to be the sum
  - no big diff
- try adding positional encoding back

# Weaker version
- train on composed task, and test on LG for individual tasks

## Smaller tasks
- 3SUM and parity
- Duplicate and reverse
- rotate and reverse
- sort
- mode?

## COT
- Automata format
  - kinda works really well?

## Other 
- Does LG scale with compute
- curriculum learning
- backtrack token
- top p/non greedy decoding/supress eos
- pad to fixed length
