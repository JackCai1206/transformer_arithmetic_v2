from random import random

# Main task: Reverse Addition
def get_reverse_add(a, b):
    s = str(int(a) + int(b))[::-1]
    l = max(len(a), len(b))
    s = s.ljust(l+1, '0')
    return f'{a[::-1]}+{b[::-1]}=', s, None

# Auxiliary 1: Pairwise digit operations
def get_pairwise_digit_op(a, b, op='mod_add', bias=0):
    def bin_op(ai, bi):
        if op == 'mod_add':
            return str((ai + bi + bias) % 10)
        elif op == 'mod_sub':
            return str((ai - bi + bias) % 10)
        elif op == 'generate_propagate':
            if a + b == bias:
                return '0', None
            elif a + b > bias:
                return '1', None
            elif a + b < bias:
                return '2', None
        elif op == 'eq':
            return str(int(ai == bi))
        elif op == 'lt':
            return str(int(ai < bi))
        elif op == 'gt':
            return str(int(ai > bi))
        else:
            raise ValueError(f'Unknown op: {op}')

    l = max(len(a), len(b))
    a = a[::-1]
    b = b[::-1]
    s = ''.join(bin_op(int(ai), int(bi)) for ai, bi in zip(a.ljust(l, '0'), b.ljust(l, '0')))
    s += '0'

    return f'{a}+{b}=', s, None

# Auxiliary 2: State Tracking Operations
def get_state_tracking(a, b, op='carry', generate_at=10, propagate_at=9):
    def bin_op(a, b, prev_c):
        if op == 'carry':
            if a + b == propagate_at:
                return '1' if prev_c else '0', prev_c
            elif a + b > generate_at:
                prev_c = True
                return '1', prev_c
            else: # a + b < carry_at
                prev_c = False
                return '0', prev_c
        elif op == 'running_sum':
            s = (a + b + prev_c) % 10
            return str(s), s
        else:
            raise ValueError(f'Unknown op: {op}')
    l = max(len(a), len(b))
    a = a[::-1]
    b = b[::-1]
    s = '0'
    prev_c = False
    for ai, bi in zip(a.ljust(l, '0'), b.ljust(l, '0')):
        si, prev_c = bin_op(int(ai), int(bi), prev_c) 
        s += si

    return f'{a}+{b}=', s, None

# Auxiliary 3: Other Operations
def get_add1(a, b):
    if random() < 0.5:
        return f'{a[::-1]}+{10**len(b)}=', str(int(a) + 10**len(b))[::-1], None
    else:
        return f'{10**len(b)}+{a[::-1]}=', str(int(a) + 10**len(b))[::-1], None

def get_reverse_sub(a, b):
    s = str(int(a) - int(b))[::-1]
    l = max(len(a), len(b))
    s = s.ljust(l+1, '0')
    return f'{a[::-1]}-{b[::-1]}=', s, None
