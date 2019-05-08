"""
Module responsible for collating all source files into a single file.
This is a utility in order to upload the source code to Kaggle's Kernel
as a single file, since the competition does not allow custom module beyond
what is in the Kernel editor. 
"""

import random
import os
import regex


PATH = './library/'
DOTTED_LINE = '# ---------------------------------------------------------------' 
LF = "\n"
DBL_LF = LF + LF

def build(output_module_name = None, local_model_name = None):
    """
    Compiles a single file containing all source codes.

    Parameters
    --------------

    output_module_name: (optional) the name of the file being outputed
    local_model_name: (required) the name of the model orchestrator file
    """

    # Closures

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
            content += m
        return content

    # Start function body

    if output_module_name is None:
        output_module_name = 'modelbuild' + str(random.randint(1, 100000)) + '.py'
    
    content = "# SETTINGS SECTION" + LF + DOTTED_LINE + LF

    content += read_file('../settings/kaggle.py')

    content +=  DBL_LF + LF +  "# AUTO ASSEMBLED CLASS IMPORT SECTION " + LF + DOTTED_LINE + LF

    files = os.listdir(PATH)

    for file in files:
        if(not file.endswith('.py')):
            continue
        content += DBL_LF + '# ' + file + LF + DOTTED_LINE + DBL_LF + read_file(PATH + file) + LF


    content = load_model(content)

    content = replace_imports(content)

    out = open(output_module_name, 'w')
    out.write(content)
    out.close()


    return output_module_name

