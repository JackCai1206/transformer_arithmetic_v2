# Data-Driven Priors Can Improve Generalization
## Copy & MQAR
- [ ] Show finetuning gpt2 can length gen on either tasks
    - Accuracy ascend and then descend
    - [ ] freezing layers help? (maybe too hard to test)
    - [ ] Scaling help? 
    - [ ] Lora help? 
- [ ] Does pretraining on synthetic N-gram text help length gen copying? 
## Addition
- Show pretrained models can solve length gen addition
    - [ ] Use a COT format based on copying
    - [ ] Make the COT format match orginal text distribution as much as possible --> sneak in numbers in the text
    - [ ] Logit distillation??
    - [ ] RLHF?? This might be able to provide proper feedback when the length is too short
- [Deprecated] Non-COT: Curriculum learning interleaved copy --> add + carry hint --> add
## Multiplication
- What is the COT format
- What is the non-COT format
## Compare with other methods
- Index hint, abacus, CAT, HAlibi --> both facilitate length gen copying --> LG addition
    - [ ] Compare CAT with current SOTA abacus
        - What is the best k for abacus? tested 300


TODO:
- [ ] Fix use_cache=True
