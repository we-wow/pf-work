"""
Data为自定义的数据处理类，用于对输入数据进行处理，并获取节点导纳矩阵。
初始化时需要指定数据文件路径和节点导纳矩阵大小即节点数量即节点导纳矩阵大小即节点数量即:
Data(path=path, shape=shape)

PowerEquation为自定义潮流方程计算类
属性：
    各个电压节点的幅值：u
    角度：angle
    导纳矩阵：admittance
方法:
    计算△PQ的方法：get_delta_pq()
    获取雅克比矩阵的方法:get_jacobian_matrix()

初始值设置为平启动/平直电压法
"""
import math
from readData import Data
from matrix import PowerEquation
from functools import wraps
import numpy as np
import time
import copy

def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" %
              (function.__name__, str(t1 - t0))
              )
        return result
    return function_timer


def data_initial(branch_path, power_path):
    data = Data(path_admittance=branch_path, path_power=power_path)
    net_work = PowerEquation(
        known_pq=data.power_data['known_pq'],
        voltage_diagonal=data.power_data['voltage'],
        voltage_angle=data.power_data['angle'],
        admittance_matrix=data.admittance_matrix,  # 导纳矩阵
        shape=data.shape,  # 节点数，导纳矩阵维数
        num_of_pq=data.num_of_pq,  # PQ节点数
    )
    return data, net_work


def data_return(shape, no2no, net_work):
    voltage = {}  # 返回节点电压
    angle = {}  # 返回节点相角
    active_power = {}  # 返回节点有功
    reactive_power = {}  # 返回节点无功
    get_p = []
    get_q = []
    for i in range(shape):
        get_p.append(net_work.P(i))
        get_q.append(net_work.Q(i))
    for i in range(1, 10):
        index = no2no[i]  # 获取节点对应运算编号
        radian2angle = []
        for radian in net_work.angle:
            radian2angle.append(radian / 2 / math.pi * 360)  # 弧度转换为角度
        voltage[i] = net_work.u[index]
        active_power[i] = get_p[index]
        reactive_power[i] = get_q[index]
        angle[i] = radian2angle[index]
    return {'voltage': voltage,
            'angle': angle,
            'active_power': active_power,
            'reactive_power': reactive_power,
            }


@fn_timer
def newton_method(brach_path, power_path):
    # 计算，初始化网络方程类
    data, net_work = data_initial(branch_path=brach_path, power_path=power_path)
    x = data.power_data['angle'][:data.shape-1] + data.power_data['voltage'][:data.num_of_pq]
    count = 0
    while count < 50:
        # X(k) <---- X(k-1)
        count += 1
        # 获取△PQ和雅克比矩阵
        delta_pq = net_work.get_delta_pq()
        if max(abs(delta_pq)) <= 1e-9:
            break  # 设置退出条件
        jacobian_matrix = net_work.get_jacobian_matrix()
        # 计算△X
        delta_x = np.linalg.solve(jacobian_matrix, delta_pq)
        delta_x[data.shape-1:] *= net_work.u[:data.num_of_pq]
        # 计算X(k+1)
        x += delta_x
        # 更新数据，角度和电压幅值
        net_work.angle[:data.shape-1] = x[:data.shape-1]
        net_work.u[:data.num_of_pq] = x[data.shape-1:]
    if count == 50:
        print("Data divergence...")
    data4return = data_return(shape=data.shape, no2no=data.no2no, net_work=net_work)
    return data4return, count, data


@fn_timer
def decoupling_method(brach_path, power_path):
    data, net_work = data_initial(branch_path=brach_path, power_path=power_path)
    x = data.power_data['angle'][:data.shape - 1] + data.power_data['voltage'][:data.num_of_pq]
    count = 0
    while count < 50:
        # X(k) <---- X(k-1)
        count += 1
        # 获取△PQ和雅克比矩阵
        delta_pq = net_work.get_delta_pq()

        delta_p = delta_pq[:data.shape-1]
        delta_q = delta_pq[data.shape-1:]
        delta_p /= net_work.u[:data.shape-1]
        delta_q /= net_work.u[:data.num_of_pq]
        if max(abs(np.concatenate([delta_p, delta_q], axis=0))) <= 1e-9:
            break  # 设置退出条件
        b1 = data.pq_jacobian_matrix
        b2 = data.admittance_matrix[:data.num_of_pq, :data.num_of_pq].imag
        # 计算△X
        delta_angle = np.linalg.solve(-b1, delta_p)
        delta_voltage = np.linalg.solve(-b2, delta_q)
        # 计算X(k+1)
        x += np.concatenate([delta_angle, delta_voltage], axis=0)
        # 更新数据，角度和电压幅值
        net_work.angle[:data.shape-1] = x[:data.shape-1]
        net_work.u[:data.num_of_pq] = x[data.shape-1:]
    if count == 50:
        print("Data divergence...")
    data4return = data_return(shape=data.shape, no2no=data.no2no, net_work=net_work)
    return data4return, count, data


# 精确计算短路电流与电压
def accurate_short_circuit_param(result, source, short_circuit_no):
    """
    :param result:A dict
    Form1: a dict which includes calculated voltage list(key's name='voltage'),calculated
    angle list(key's name='angle')calculated active power list(key's name='activate_power'),
    and calculated reactive power list(key's name='activate_power').
    Form2: the first return value of function newton_method/decoupling_method
    :param source:A dict
    The third return value of function newton_method/decoupling_method
    :param short_circuit_no: An integer. the number of short circuit node.
    :return: three dicts
    Including short circuit voltage and current
    The third dict is some information about this calculation, and has nothing to do with the result.
    """
    result = copy.deepcopy(result)
    source = copy.deepcopy(source)
    voltage = result['voltage']
    angle = result['angle']
    active_power = result['active_power']
    reactive_power = result['reactive_power']
    info = {}
    # 形成带发电机负荷的节点阻抗矩阵
    changed_admittance = {}
    for i in range(len(source.input_file_list_power)):
        impedance = source.input_file_list_power[i][8]
        if impedance == 'l':  # 负荷节点
            impedance = complex(abs(active_power[source.reno2no[i]]),
                                - abs(reactive_power[source.reno2no[i]])) / (voltage[source.reno2no[i]] ** 2)
            source.admittance_matrix[i][i] += impedance
            changed_admittance[source.reno2no[i]] = source.admittance_matrix[i][i]
        elif type(source.input_file_list_power[i][8]) == float:  # 发电机节点
            source.admittance_matrix[i][i] += (1/complex(0, impedance))
            changed_admittance[source.reno2no[i]] = source.admittance_matrix[i][i]

    info['changed_admittance'] = changed_admittance
    impedance_matrix = np.linalg.inv(source.admittance_matrix)  # 求逆，建立阻抗矩阵
    info['impedance_matrix'] = impedance_matrix
    # 矩阵建立完毕

    f = short_circuit_no  # 输入短路节点编号
    Zf = 0  # 默认金属性三相短路，短路阻抗等于0
    short_circuit_voltage = {}
    short_circuit_current = []

    # ---------------精确计算短路电流-------------------
    Zff = impedance_matrix[source.no2no[f]][source.no2no[f]]
    If = complex(voltage[f]*math.cos(angle[f]/180*math.pi), voltage[f]*math.sin(angle[f]/180*math.pi)) / (Zff + Zf)
    info['If'] = If
    for no, vol in voltage.items():
        vol = complex(vol*math.cos(angle[no]/180*math.pi), vol*math.sin(angle[no]/180*math.pi))
        Zif = impedance_matrix[source.no2no[no]][source.no2no[f]]
        short_circuit_voltage[no] = vol - Zif * If

    for branch in source.input_file_list_admittance:
        if type(branch[7]) == float:
            k = branch[7]
            head = branch[1] - 1
            foot = branch[3] - 1
            i = source.reno2no[head]
            j = source.reno2no[foot]
            current = (short_circuit_voltage[i] - short_circuit_voltage[j] / k) / impedance_matrix[head][foot]
            short_circuit_current.append({'head': i, 'foot': j, 'current': current, 'type': '变压器支路'})
        else:
            head = branch[1] - 1
            foot = branch[3] - 1
            i = source.reno2no[head]
            j = source.reno2no[foot]
            current = (short_circuit_voltage[i] - short_circuit_voltage[j]) / complex(branch[4], branch[5])
            short_circuit_current.append({'head': i, 'foot': j, 'current': current, 'type': ''})
    return short_circuit_voltage, short_circuit_current, info


def rough_short_circuit_param(source, short_circuit_no=4):
    """
    :param source:A dict
    The third return value of function newton_method/decoupling_method
    :param short_circuit_no: An integer. the number of short circuit node.
    :return: Two dict
    Including roughly calculated short-circuit voltage.
    And the second dict is just some information about calculation, and has nothing to do with the result.
    """
    source = copy.deepcopy(source)
    info = {}
    changed_admittance = {}
    for i in range(len(source.input_file_list_power)):
        impedance = source.input_file_list_power[i][8]
        if type(source.input_file_list_power[i][8]) == float:  # 发电机节点
            source.admittance_matrix[i][i] += (1 / complex(0, impedance))
            changed_admittance[source.reno2no[i]] = source.admittance_matrix[i][i]

    info['changed_admittance'] = changed_admittance
    impedance_matrix = np.linalg.inv(source.admittance_matrix)  # 求逆，建立阻抗矩阵
    info['impedance_matrix'] = impedance_matrix
    f = short_circuit_no  # 输入短路节点编号
    Zf = 0  # 默认金属性三相短路，短路阻抗等于0

    # ---------------粗略计算短路电流-------------------
    Zff = impedance_matrix[source.no2no[f]][source.no2no[f]]
    If = 1 / (Zff + Zf)
    info['If'] = If
    short_circuit_voltage = {}
    for no in range(source.shape):
        Zif = impedance_matrix[source.no2no[no+1]][source.no2no[f]]
        short_circuit_voltage[no+1] = 1 - Zif * If

    return short_circuit_voltage, info


def write2txt(result, source, info, rough_info, sc_voltage, sc_current, rough_sc_voltage, short_circuit_no):
    with open('./result.txt', 'w', encoding='utf-8') as res:
        voltage = result['voltage']
        angle = result['angle']

        # ------------------写入稳态计算结果--------------------
        res.write("\n#################电压幅值与相角（单位：度）#################\n")
        for key, value in voltage.items():
            res.write('U_{}:{}∠{}°\n'.format(key, round(value, 5), round(angle[key], 5)))

        # ------------------写入节点导纳矩阵非零元素--------------
        res.write("\n#################不包括发电机与负荷节点的节点导纳矩阵非零元素#################\n")
        for i in range(source.shape):
            for j in range(source.shape):
                if source.admittance_matrix[i][j] != 0:
                    res.write('Y{}{}:{}\n'.format(source.reno2no[i], source.reno2no[j], round(source.admittance_matrix[i][j], 5)))

        # ------------------写入修正后的导纳--------------------
        res.write("\n#################精确计算时，修正后的导纳#################\n")
        for no, val in info['changed_admittance'].items():
            res.write('Y{}{}:{}\n'.format(no, no, round(val, 5)))

        # ------------------写入短路列的阻抗--------------------
        res.write("\n#################第{}列的阻抗#################\n".format(short_circuit_no))
        j = source.no2no[short_circuit_no]
        impedance_matrix = info['impedance_matrix']
        for i in range(len(impedance_matrix)):
            res.write('Z{}{}:{}\n'.format(source.reno2no[i],short_circuit_no, round(impedance_matrix[i][j], 5)))

        # ------------------写入并分析短路电流--------------------
        res.write("\n#################短路电流对比#################\n")

        short_current = info['If']
        modulus = abs(short_current)
        angle = math.atan(short_current.imag / short_current.real)/2/math.pi*360
        res.write('精确计算时的短路电流：{}∠{}°\n'.format(round(modulus, 5), round(angle, 5)))

        short_current = rough_info['If']
        modulus_2 = abs(short_current)
        angle_2 = math.atan(short_current.imag / short_current.real)/2/math.pi*360
        res.write('粗略计算时的短路电流：{}∠{}°\n'.format(round(modulus_2, 5), round(angle_2, 5)))

        res.write('相比于精确计算，模值误差为：{}%\n'.format(round(abs(modulus - modulus_2) / abs(modulus)*100), 5))
        res.write('相比于精确计算，角度误差为：{}%\n'.format(round(abs(angle - angle_2) / abs(angle) * 100), 5))

        # ------------------写入并分析短路电压--------------------
        res.write("\n#################短路电压对比#################\n")
        for no, val in sc_voltage.items():
            if abs(val) >= 1e-10:
                res.write('节点{}：精确计算电压：{}/粗略计算电压：{}/相比于精确值的误差为{}%\n'.format(
                    no, round(abs(val), 5), round(abs(rough_sc_voltage[no]), 5),
                    round(abs(abs(val)-abs(rough_sc_voltage[no]))/abs(val)*100, 5)
                ))
            else:
                res.write('节点{}：精确计算电压：{}/粗略计算电压：{}/相比于精确值的误差为{}%\n'.format(
                    no, round(abs(val), 5), round(abs(rough_sc_voltage[no]), 5),
                    0.0
                ))

        # ------------------写入并分析短路电流--------------------
        res.write("\n#################短路电流计算#################\n")
        for branch in sc_current:
            head = branch['head']
            foot = branch['foot']
            current = branch['current']
            line_type = branch['type']
            modulus = abs(current)
            angle = math.atan(current.imag / current.real) / 2 / math.pi * 360
            res.write('支路电流{}---->>{}：{}∠{}°   {}\n'.format(head, foot, round(modulus, 5), round(angle, 5), line_type))

    print('文件已经写入到result.txt')