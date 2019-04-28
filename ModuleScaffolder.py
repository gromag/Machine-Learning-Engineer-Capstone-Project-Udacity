import random
import os
import regex


PATH = './notebooks/library/'

def build(output_module_name = None, local_model_name = None):
    
    files = os.listdir(PATH)
    separator = '\n\n' +\
                '# ----------------- SECTION AUTO ASSEMBLED ----------------------\n' +\
                '# ---------------------------------------------------------------\n' 
    
    if output_module_name is None:
        output_module_name = 'modelbuild' + str(random.randint(1, 100000)) + '.py'

    def read_file(file):
        f = open(file, "r")
        c = f.read()
        f.close()
        return c

    def replace_imports(content):
        return regex.sub(r'from library\.[^\n]+', '', content)

    def load_model(content):
        if local_model_name is not None:
            m = read_file(local_model_name)
            m = m.split('# AUTO-REMOVE-ABOVE')[1]
            content += separator + m
        return content


    content = ''

    for file in files:
        if(not file.endswith('.py')):
            continue
        content +=  separator + read_file(PATH + file)


    content += separator + read_file('./settings/kaggle.py')

    content = load_model(content)

    content = replace_imports(content)

    out = open(output_module_name, 'w')
    out.write(content)
    out.close()


    return output_module_name

