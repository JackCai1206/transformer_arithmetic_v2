from random import randint

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
    a_digits, b_digits, carries = split_digits(a, b)
    a_str = ''.join(map(str, a_digits))
    b_str = ''.join(map(str, b_digits))

    prompt = ''
    cot = ''
    ans = str(a + b)[::-1] # left justify means padding on the right
    
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
            prompt = f'D{ans_prev}\nC{c_prev}P{a_str[i:]}+{b_str[i:]}='
            cot += f'A{a_i}B{b_i}S{(a_i + b_i) % 10}D{ans[:i+1]}\nC{c_i}'
        else:
            cot += f'P{a_str[i:]}+{b_str[i:]}=A{a_i}B{b_i}S{(a_i + b_i) % 10}D{ans[:i+1]}\nC{c_i}'

    # if train:
    #     print(prompt + cot)
    #     breakpoint()
    return prompt, cot

def get_reverse(a, b):
    a = int(a)
    b = int(b)
    return f'{str(a)[::-1]}+{str(b)[::-1]}=', f'{str(a+b)[::-1]}'

def get_nar(a, n=5):
    i = randint(0, len(a) - n)

    prompt = f'{a}[SEP]{a[i:i+n-1]}'
    target = a[i+n-1]
    
    return prompt, target

def get_copy(a):
    return f'{a}[SEP]', a

def get_interleave_copy(a, b):
    a_digits, b_digits, carries = split_digits(int(a), int(b))
    
    prompt = f'{str(a)[::-1]}+{str(b)[::-1]}='
    target = ''.join(f'{da}{db}' for da, db in zip(a_digits, b_digits))

    return prompt, target
