# -*- coding: utf-8 -*-
def print_variable_info(variable, print_in_one_line=True, print_value=True):
    """
    打印变量的相关信息, 若不是变量, 请先赋予一个变量, 再传入
    :param variable: 变量
    :param print_in_one_line: 真, 一行打印           假, 分多行打印
    :param print_value: 真, 打印变量的值           假, 不打印变量的值      当变量的值过长时, 最好不要打印
    """
    import inspect
    from inspect import stack
    line_no = str(stack()[1][2])
    some_info = list(inspect.currentframe().f_back.f_locals.items())
    variable_name_value_list = []
    for a_info in some_info:
        if a_info[1] is variable:
            variable_name_value_list.append(a_info)
    if len(variable_name_value_list) == 1:
        variable_name = str(variable_name_value_list[0][0])
        variable_value = str(variable_name_value_list[0][1])
        variable_type = str(type(variable))
        variable_len = ''
        variable_key = ''
        try:
            len(variable)
        except:
            pass
        else:
            variable_len = str(len(variable))
        try:
            list(variable.keys())
        except:
            pass
        else:
            variable_key = str(list(variable.keys()))
        if print_in_one_line:
            output = '---------- 行号: ' + line_no + ' , 变量名: ' + variable_name + ' , 类型: ' + variable_type
            if variable_len and variable_key:
                output = output + ' , 长度: ' + variable_len + ' , 键: ' + variable_key
            elif variable_len:
                output = output + ' , 长度: ' + variable_len
            elif variable_key:
                output = output + ' , 键: ' + variable_key
            if print_value:
                output = output + ' , 值: ' + variable_value
        else:
            output = '--------------------------------------------------------------------------\n' + \
                     '---------- 行号: ' + line_no + '\n' + \
                     '---------- 变量名: ' + variable_name + '\n' + \
                     '---------- 类型: ' + variable_type + '\n'
            if variable_len and variable_key:
                output = output + '---------- 长度: ' + variable_len + '\n' + '---------- 键: ' + variable_key + '\n'
            elif variable_len:
                output = output + '---------- 长度: ' + variable_len + '\n'
            elif variable_key:
                output = output + '---------- 键: ' + variable_key + '\n'
            if print_value:
                output = output + '---------- 值: ' + variable_value + '\n'
            output = output + '--------------------------------------------------------------------------'
        print(output)
    elif not variable_name_value_list:
        print('函数 print_variable_info 出故障了')
    else:
        output = '--------------------------------------------------------------------------\n' + \
                 '---------- 行号: ' + line_no + '\n' + \
                 '---------- 有以下几种可能: \n'
        for a_variable_name_value in variable_name_value_list:
            output = output + '-------------------- 变量名: ' + str(a_variable_name_value[0])
            if print_value:
                output = output + ', 值:' + str(a_variable_name_value[1])
            output = output + '\n'
        output = output + '---------- 类型: ' + str(type(variable)) + \
                 '\n--------------------------------------------------------------------------'
        print(output)


def get_code_info(mode='current_info', to_print=True, complete_file_path=False, a_self=None):
    """
    获取代码信息, 若模式是 current_info, 只返回当前函数信息; 若模式是 stack_info, 则返回各层函数信息
    :param mode: 模式, 可选: 'current_info', 'stack_info'
    :param to_print: 是否打印, 可选: True/False
    :param complete_file_path: 是否使用完整的文件路径, 若为否, 只取文件名
    :param a_self: 若想要类名, 请传入类方法的 self
    :return: 包含代码信息的字符串
    """
    import os
    assert mode in ['current_info', 'stack_info'], 'mode 参数输入错误'
    assert to_print in [True, False], 'to_print 参数输入错误'
    assert complete_file_path in [True, False], 'complete_file_path 参数输入错误'
    from inspect import stack
    stack_info_frame = stack()

    def get_a_layer_info_str(a_layer_info_frame, get_class_name=False):
        file_name_str = '文件名: ' + a_layer_info_frame[1] + '\n' if complete_file_path else '文件名: ' + os.path.basename(
            a_layer_info_frame[1]) + '\n'
        line_no_str = '行号: ' + str(a_layer_info_frame[2]) + '\n'
        class_name_str = ''
        if (a_self is not None) and get_class_name:
            try:
                a_self.__class__
            except:
                pass
            else:
                class_name_str = '****** 类名: ' + str(a_self.__class__) + ' ******\n'
        function_name_str = '函数名: ' + a_layer_info_frame[3] + '\n'
        code_context_str = '代码内容: ' + str(a_layer_info_frame[4]) + '\n'
        a_layer_info_str = '------------------------------------------------\n' + \
                           file_name_str + line_no_str + class_name_str + function_name_str + code_context_str + \
                           '------------------------------------------------\n'
        return a_layer_info_str

    if mode == 'current_info':
        a_layer_info_str = get_a_layer_info_str(stack_info_frame[1], get_class_name=True)
        if to_print:
            print(a_layer_info_str)
        return a_layer_info_str
    elif mode == 'stack_info':
        stack_info_str = ''
        for i in range(len(stack_info_frame) - 1, 0, -1):
            if i >= 2:
                stack_info_str = stack_info_str + get_a_layer_info_str(stack_info_frame[i])
                stack_info_str = stack_info_str + '    ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓    \n'
            else:
                stack_info_str = stack_info_str + get_a_layer_info_str(stack_info_frame[i], get_class_name=True)
        if to_print:
            print(stack_info_str)
        return stack_info_str


def delete_file_fistline(file_path):
    """
    删除文件的第一行, 其他行上移
    :param file_path: 文件路径
    :return: 成功删除返回 True, 否则返回 False
    """
    import os
    import fileinput
    try:
        if not os.path.exists(file_path):
            print('文件', file_path, '不存在')
            return False
        if os.path.getsize(file_path) == 0:
            print('文件', file_path, '为空, 无法删除第一行')
            return False
        for line in fileinput.input(file_path, inplace=1):
            if not fileinput.isfirstline():
                print(line.replace('\n', ''))
        return True
    except:
        print('函数 delete_file_fistline 发生故障')
        return False


def file_row_num_count(file_path):
    """
    计算大文件行数
    :param file_path: 文件路径, 类型: 字符串
    :return: 行数, 类型: 整型
    """
    count = 0
    with open(file_path, encoding='utf-8', mode='r') as f:
        for count, _ in enumerate(f, 1):
            pass
    return count


def is_str(str):
    """
    判断是否是字符串
    :param str: 参数
    :return: 真假
    """
    try:
        str + 'aaa'
    except:
        return False
    else:
        return True


class MyLogger(object):
    """
    用于重定向标准输出和错误输出(控制台的输出不会变, 只是将控制台的输出复制一份到文件中
    使用方法: 先导入本类, 再在程序中写入:
    sys.stdout = MyLogger(文件路径1, sys.stdout)
    sys.stderr = MyLogger(文件路径2, sys.stderr)
    位于上面两句之后的程序语句, 如果有输出到控制台的, 都会额外输出到文件中
    """
    import sys
    def __init__(self, filename, stream=sys.stdout, mode='w'):
        self.terminal = stream
        self.log = open(filename, mode=mode, encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# 下面三个是组合的:
# def init_logger():
#     """
#     初始化 logger (需要自行改代码中的log文件路径)
#     :return: logger
#     """
#     import logging
#     logger = logging.getLogger()
#     file_handler = logging.FileHandler(filename='log文件路径', mode='a', encoding='utf-8')
#     file_handler.setFormatter(logging.Formatter("%(message)s - %(asctime)s - %(levelname)s"))
#     logger.addHandler(file_handler)
#     logger.setLevel(logging.INFO)
#     return logger
#
#
# logger = init_logger()
#
#
# def print_and_log(describe_str):
#     logger.info(describe_str)
#     print(describe_str)


def my_stop():
    print(11111111111111111111111)
    print(11111111111111111111111)
    print(11111111111111111111111)
    print(11111111111111111111111)
    exit(0)


def my_dividing_line(num=100):
    print('\n' + '#'*num + '\n')
