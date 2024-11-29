from tokenizer import tokenize
from io import BytesIO
from nltk.tokenize import word_tokenize

def tokenize_code(code):
    tokens = []
    code_bytes = code.encode('utf-8')
    code_io = BytesIO(code_bytes)

    for token in tokenize.tokenize(code_io.readline):
        if token.line != '':
            tokens.append(token)
    
    return tokens

def detokenize_code(tokens):
    detokenized_code = ''
    prev_end_line = 0
    prev_end_col = 0
    
    for token in tokens:
        start_line, start_col = token.start
        end_line, end_col = token.end
        
        if start_line > prev_end_line+1:
            detokenized_code += '\n' * (start_line - prev_end_line+1)
            
        if start_col > prev_end_col:
            detokenized_code += ' ' * (start_col - prev_end_col)
        
        detokenized_code += token.string
        
        prev_end_line = end_line
        prev_end_col = end_col
        
    return detokenized_code


def get_tokenstr_list(tokens):
    token_list = []
    for token in tokens:
        if token.string!= '':
            # print(token.type )
            if token.type == 3 and len(token.string)>20:
                token_list.extend(tokenize_nl(token.string))
            else:
                token_list.append(token.string)
    return token_list

def tokenize_nl(nl):
    tokens = word_tokenize(nl)
    return tokens

def subtokenize_nl(nl, tokenizer):
    tokens = tokenizer.tokenize(nl)
    return tokens

def subtokenize_code(nl, tokenizer):
    tokens = tokenizer.tokenize(nl)
    return tokens