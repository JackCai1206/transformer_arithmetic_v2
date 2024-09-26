from random import randint, choice, sample, shuffle, random

def split_digits(a, b):
    a_digits = []
    b_digits = []
    carries = [0]
    while True:
        a_digits.append(a % 10)
        b_digits.append(b % 10)
        carries.append((a % 10 + b % 10) // 10)
        a //= 10
        b //= 10
        if a == 0 and b == 0:
            break
    return a_digits, b_digits, carries

def get_COT(a, b):
    a_digits, b_digits, carries = split_digits(int(a), int(b))
    a_str = ''.join(map(str, a_digits))
    b_str = ''.join(map(str, b_digits))

    prompt = ''
    cot = ''
    ans = str(int(a) + int(b))[::-1] # left justify means padding on the right
    
    abc_list = list(enumerate(zip(a_digits, b_digits, carries)))
    k = len(abc_list)
    j = 0
    for i, (a_i, b_i, c_i) in abc_list[j:j+k]:
        if i == j:
            if i - 1 >= 0:
                c_prev = carries[i-1]
                ans_prev = ans[:i]
            else:
                c_prev = 0
                ans_prev = ''
            # prompt = f'D{ans_prev}\nC{c_prev}P{a_str[i:]}+{b_str[i:]}='
            prompt = f'Q{str(a)}+{str(b)}='
            cot += f'\nC0P{a_str[i:]}+{b_str[i:]}='
            cot += f'A{a_i}B{b_i}S{(a_i + b_i) % 10}D{ans[:i+1]}\nC{c_i}'            
        else:
            cot += f'P{a_str[i:]}+{b_str[i:]}=A{a_i}B{b_i}S{(a_i + b_i) % 10}D{ans[:i+1]}\nC{c_i}'
    cot += f'P =A B S D{ans}'
    # if train:
    #     print(prompt + cot)
    #     breakpoint()
    return prompt, cot, None

def get_add1(a, b):
    if random() < 0.5:
        return f'A{a[::-1]}+1{"0"*(len(b)-1)}=', str(int(a) + 1)[::-1], None
    else:
        return f'A1{"0"*(len(b)-1)}+{a[::-1]}=', str(int(a) + 1)[::-1], None

def get_reverse_add_automata(a, b, type='A'):
    c = 0
    s = '0'
    loss_mask = [0]
    for ai, bi in zip(a[::-1], b[::-1]):
        s += f'{ai}{bi}'
        s += str((int(ai) + int(bi) + c) % 10)
        c = (int(ai) + int(bi) + c) // 10
        s += f'{c}'
        if type == 'A':
            loss_mask += [0, 0, 1, 1]
        elif type == 'B':
            loss_mask += [1, 1, 0, 0]
        else:
            loss_mask += [1, 1, 1, 1]
    
    return f'{a[::-1]}+{b[::-1]}=', s, loss_mask

def get_reverse_add(a, b):
    s = str(int(a) + int(b))[::-1]
    l = max(len(a), len(b))
    s = s.ljust(l+1, '0')
    return f'C{a[::-1]}+{b[::-1]}=', s, None

def get_reverse_add_cont(a, b):
    s = list(str(int(a) + int(b)))[::-1]
    cis = sample(range(len(s)), randint(0, round(len(s) / 5)))
    cis.sort()
    for i, ci in enumerate(cis):
        s.insert(ci + i, 'X')
    s = ''.join(s)
    return f'{a[::-1]}+{b[::-1]}=', s, None

def get_forward(a, b):
    s = str(int(a) + int(b))
    return f'A{a}+{b}=', s, None

def get_reverse_no_carry(a, b, randomize=False):
    def bin_op(a, b):
        # return chr(ord('a') + (a + b) % 10)
        return str((a + b) % 10)
    l = max(len(a), len(b))
    a = a[::-1]
    b = b[::-1]
    s = ''.join(bin_op(int(ai), int(bi)) for ai, bi in zip(a.ljust(l, '0'), b.ljust(l, '0')))
    s += '0'
    
    if not randomize:
        return f'A{a}+{b}=', s, None
    else:
        s = list(s)
        shuffle(s)
        s = ''.join(s)
        return f'A{a}+{b}=', s, None

def get_forward_no_carry(a, b, randomize=False):
    def bin_op(a, b):
        # return chr(ord('a') + (a + b) % 10)
        return str((a + b) % 10)
    l = max(len(a), len(b))
    a = a[::-1]
    b = b[::-1]
    s = ''.join(bin_op(int(ai), int(bi)) for ai, bi in zip(a.ljust(l, '0'), b.ljust(l, '0')))
    s += '0'
    
    if not randomize:
        return f'A{a[::-1]}+{b[::-1]}=', s[::-1], None
    else:
        s = list(s)
        shuffle(s)
        s = ''.join(s)
        return f'A{a[::-1]}+{b[::-1]}=', s[::-1], None

def get_reverse_carry_only(a, b, randomize=False):
    def bin_op(a, b, prev_c):
        if a + b == 9:
            # return '.' if prev_c else '_', prev_c
            return '1' if prev_c else '0', prev_c
        elif a + b > 9:
            prev_c = True
            return '1', prev_c
            # return '.', prev_c
        else: # a + b < 9
            prev_c = False
            return '0', prev_c
            # return '_', prev_c
    l = max(len(a), len(b))
    a = a[::-1]
    b = b[::-1]
    s = '0'
    prev_c = False
    for ai, bi in zip(a.ljust(l, '0'), b.ljust(l, '0')):
        si, prev_c = bin_op(int(ai), int(bi), prev_c) 
        s += si
        
    if not randomize:
        return f'B{a}+{b}=', s, None
    else:
        s = list(s)
        shuffle(s)
        s = ''.join(s)
        return f'B{a}+{b}=', s, None

def get_forward_carry_only(a, b, randomize=False):
    def bin_op(a, b, prev_c):
        if a + b == 9:
            # return '.' if prev_c else '_', prev_c
            return '1' if prev_c else '0', prev_c
        elif a + b > 9:
            prev_c = True
            return '1', prev_c
            # return '.', prev_c
        else: # a + b < 9
            prev_c = False
            return '0', prev_c
            # return '_', prev_c
    l = max(len(a), len(b))
    a = a[::-1]
    b = b[::-1]
    s = '0'
    prev_c = False
    for ai, bi in zip(a.ljust(l, '0'), b.ljust(l, '0')):
        si, prev_c = bin_op(int(ai), int(bi), prev_c) 
        s += si
        
    if not randomize:
        return f'B{a[::-1]}+{b[::-1]}=', s[::-1], None
    else:
        s = list(s)
        shuffle(s)
        s = ''.join(s)
        return f'B{a[::-1]}+{b[::-1]}=', s[::-1], None

def get_nar(a, n=5):
    i = randint(0, len(a) - n)

    prompt = f'{a}[SEP]{a[i:i+n-1]}'
    target = a[i+n-1]
    
    return prompt, target, None

def get_sort(a, reverse=False):
    x = []
    for ai in a:
        x.append(f'{str(ai).zfill(2)}')
    y = sorted(x, reverse=reverse)

    return ','.join(x) + 'A[SEP]', ','.join(y), None

def get_set_diff(a):
    x = []
    for ai in a:
        x.append(f'{str(ai).zfill(2)}')
    j = choice(range(len(x)))
    si = sorted(range(len(x)), key=lambda k: x[k])
    y = [x[k] for k in sorted(si[j+1:])]
    x_sort= [x[k] for k in si[:j+1]]

    return ','.join(x) + 'B[SEP]' + ','.join(x_sort) + ',', ','.join(y), None

def get_minimum(a):
    x = []
    for ai in a:
        x.append(f'{str(ai).zfill(2)}')
    j = choice(range(len(x)))
    si = sorted(range(len(x)), key=lambda k: x[k])[j:]
    y = x[si[0]]
    x = [x[k] for k in si]

    return ','.join(x) + 'C[SEP]', str(y), None

def get_rotate1(a):
    y = [str((int(ai) + 1) % 10) for ai in a]
    return ''.join(a) + 'A[SEP]', ''.join(y), None

def get_reverse(a):
    return a + 'B[SEP]', a[::-1], None

def get_rot1rev(a):
    y = [str((int(ai) + 1) % 10) for ai in a]
    return a + 'C[SEP]', ''.join(y[::-1]), None

def get_interleave_copy(a, b):
    a_digits, b_digits, carries = split_digits(int(a), int(b))
    # b_digits = [chr(ord('a') + i) for i in b_digits][::-1]
    a_digits = a_digits[::-1]
    b = ''.join(b_digits)

    prompt = f'A{str(a)},{str(b)}[SEP]'
    target = ''.join(f'{da}{db}' for da, db in zip(a_digits, b_digits))

    return prompt, target, None

def get_reverse_2op(a, b):
    # b = ''.join([chr(ord('a') + int(i)) for i in b])
    return f'B{a},{b}[SEP]', b[::-1]+','+a[::-1], None

def get_itcopy_rev(a, b):
    prompt, target, _ = get_interleave_copy(a, b)
    return prompt.replace('A', 'C'), target[::-1], None
    # return prompt.replace('A', 'C'), get_reverse_2op(a, b)[1] + 'A[SEP]' + target[::-1]

def get_sd_mult(a, b):
    a_rev = a[::-1]
    i = int(choice(range(len(b))))
    task_id = chr(ord('C') + i)
    b_rev = b[::-1]
    # b_rev = ''.join(['0' if i != j else b_rev[i] for j in range(len(b))])
    return f'{a_rev}*{b_rev}{task_id}=', '0' * i + str(int(a) * int(b_rev[i]))[::-1], None
    # cot = []
    # for i, bi in enumerate(b_rev):
    #     cot.append('0' * i + str(int(a) * int(bi))[::-1])
    # cot = '+'.join(cot)
    # return f'{a_rev}*{b_rev}A=', f'{cot}B='

def get_mult(a, b):
    a_rev = a[::-1]
    b_rev = b[::-1]
    cot = []
    for i, bi in enumerate(b_rev):
        cot.append('0' * i + str(int(a) * int(bi))[::-1])
    cot = '+'.join(cot)
    return f'{a_rev}*{b_rev}AB=', f'{cot}B=' + str(int(a) * int(b))[::-1], None

def get_cumsum(a):
    a_rev = a[::-1]
    cot = ''
    s = 0
    for ai in a_rev:
        s += int(ai)
        s = s % 10
        cot += str(s)
    return f'A{a_rev}=', cot, None

def get_gt5(a):
    a_rev = a[::-1]
    cot = ''
    for ai in a_rev:
        if int(ai) > 5:
            cot += '1'
        else:
            cot += '0'
    return f'B{a_rev}=', cot, None

def get_cumsum_gt5(a):
    a_rev = a[::-1]
    cot = ''
    s = 0
    for ai in a_rev:
        s += int(ai)
        s = s % 10
        if s >= 5:
            cot += '1'
        else:
            cot += '0'
    return f'C{a_rev}=', cot, None

def get_copy(a):
    return f'D{a}=', a, None

def get_3sum(a):
    s = ''
    for i in range(0, len(a), 2):
        s += str(a[i] or a[i+1])
    a_str = ''.join(map(str, a))
    return f'A{a_str}=', s, None

def get_parity(a):
    # s = sum(a)
    # a_str = ''.join(map(str, a))
    # return f'B{a_str}=', str(s), None
    s = ''
    for i in range(0, len(a), 2):
        s += str(a[i] and a[i+1])
    a_str = ''.join(map(str, a))
    return f'B{a_str}=', s, None

def get_3parity(a):
    # s = 0
    # for i in range(0, len(a), 3):
    #     s = s + int(a[i] or a[i+1] or a[i+2])
    # # s = s % 2
    # a_str = ''.join(map(str, a))
    # return f'C{a_str}=', str(s), None
    or_str = get_3sum(a)[1]
    and_str = get_parity(a)[1]
    s = ''
    for ai, bi in zip(or_str, and_str):
        s += str((int(ai) + int(bi)) % 2)
    a_str = ''.join(map(str, a))
    return f'C{a_str}=', s, None


def get_random_truncated_number(a, b):
    # for string a, b (numbers), return a length-1 truncated version of a or b (or both)
    if len(a) == 1 and len(b) == 1:
        orig_a, orig_b = a, b
        # re-pick both numbers until a != a and b != b
        while a == orig_a and b == orig_b:
            a = str(randint(0, 9))
            b = str(randint(0, 9))
    elif len(a) == 1:
        # randomly drop one digit from b from any position of b
        i = randint(0, len(b) - 1)
        b = b[:i] + b[i+1:]
    elif len(b) == 1:
        # randomly drop one digit from a from any position of a
        i = randint(0, len(a) - 1)
        a = a[:i] + a[i+1:]
    else:
        # randomly drop one digit from a or b (or both)
        if random() < 0.5:
            i = randint(0, len(a) - 1)
            a = a[:i] + a[i+1:]
            i = randint(0, len(b) - 1)
            b = b[:i] + b[i+1:]
        else:
            if random() < 0.5:
                i = randint(0, len(a) - 1)
                a = a[:i] + a[i+1:]
            else:
                i = randint(0, len(b) - 1)
                b = b[:i] + b[i+1:]
    return a, b


def get_truncated_cot(l):
    # for dpo: get randomly truncated cot for l
    # l: string of prompt ex. [BOS]Q412661+1941=
    a, b = l.split('+')
    a = a[6:]
    b = b[:-1]
    a, b = get_random_truncated_number(a, b)

    a_digits, b_digits, carries = split_digits(int(a), int(b))
    a_str = ''.join(map(str, a_digits))
    b_str = ''.join(map(str, b_digits))

    prompt = ''
    cot = ''
    ans = str(int(a) + int(b))[::-1] # left justify means padding on the right
    
    abc_list = list(enumerate(zip(a_digits, b_digits, carries)))
    k = len(abc_list)
    j = 0
    for i, (a_i, b_i, c_i) in abc_list[j:j+k]:
        if i == j:
            if i - 1 >= 0:
                c_prev = carries[i-1]
                ans_prev = ans[:i]
            else:
                c_prev = 0
                ans_prev = ''
            # prompt = f'D{ans_prev}\nC{c_prev}P{a_str[i:]}+{b_str[i:]}='
            prompt = f'Q{str(a)}+{str(b)}='
            cot += f'\nC0P{a_str[i:]}+{b_str[i:]}='
            cot += f'A{a_i}B{b_i}S{(a_i + b_i) % 10}D{ans[:i+1]}\nC{c_i}'            
        else:
            cot += f'P{a_str[i:]}+{b_str[i:]}=A{a_i}B{b_i}S{(a_i + b_i) % 10}D{ans[:i+1]}\nC{c_i}'
    cot += f'P =A B S D{ans}'
    # if train:
    #     print(prompt + cot)
    #     breakpoint()
    return cot


if __name__ == '__main__': 
    print(get_reverse_carry_only('122343', '994499'))
    print(get_reverse_no_carry('122343', '994499'))
    print(get_reverse_add('122343', '994499'))

