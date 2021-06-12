"""
project: Load Flow Calculation
author: @魏明江
time: 2020/02/22
attention：readData.py定义了数据类完成了从txt读取文档，然后建立节点导纳矩阵的过程
           Data类包含的属性有 ：path, input_file_list, admittance_matrix分别是
           源文件路径，读取完成并经过数据转换后的输入列表，以及节点导纳矩阵
           Data类包含的可用方法有read_data(self), get_admittance_matrix(self)
           分别是读取并转换数据和计算节点导纳矩阵
"""


class Data:
    def __init__(self, path_admittance, path_power):
        self.path_admittance = path_admittance
        self.path_power = path_power
        self.bus_type = {}
        self.input_file_list_admittance = self.read_admittance_data()
        self.input_file_list_power = self.read_power_data()
        self.shape = len(self.read_power_data())
        rename_data = self.rename_bus()  # 由于对节点进行了重编号，建立索引字典
        self.no2no = rename_data[0]
        self.reno2no = rename_data[2]
        self.num_of_pq = rename_data[1]
        self.admittance_matrix = self.get_admittance_matrix()['data_array']
        self.pq_jacobian_matrix = self.get_admittance_matrix()['b_array']
        self.power_data = self.get_power_data()

    # 读取输入文件列表
    def read_admittance_data(self):
        data_txt = []
        with open(self.path_admittance, 'r', encoding='utf-8') as data_file:
            for each_line in data_file.readlines():
                data_each_line = each_line.split(' ')
                # 对应字符串类型转换
                try:
                    for i in [1, 3]:
                        data_each_line[i] = int(data_each_line[i])
                    for i in range(4, 8):
                        if data_each_line[i] != '/' and data_each_line[i] != '/\n':
                            data_each_line[i] = float(data_each_line[i])
                        else:
                            data_each_line[i] = '/'
                except ValueError:
                    print('wrong input format!')
                # 转换完毕，得到列表
                data_txt.append(data_each_line)
        return data_txt

    # 读取功率输入表
    def read_power_data(self):
        data_txt = []
        with open(self.path_power, 'r', encoding='utf-8') as data_file:
            for each_line in data_file.readlines():
                data_each_line = each_line.split(' ')
                data_each_line[-1] = data_each_line[-1].replace('\n', '')
                try:
                    data_each_line[1] = int(data_each_line[1])
                    for i in range(2, 10):
                        if data_each_line[i] not in ['/', '/\n', 'l']:
                            data_each_line[i] = float(data_each_line[i])
                except ValueError:
                    print('wrong input format!')
                    # 转换完毕，得到列表
                data_txt.append(data_each_line)
        data_txt.sort(key=lambda x: (x[1]))
        return data_txt

    # 节点重编号
    def rename_bus(self):
        pq_bus = []
        pv_bus = []
        slack_bus = 0
        input_power_data = self.input_file_list_power
        input_admittance_data = self.input_file_list_admittance
        for branch in input_power_data:
            i = branch[1]
            if branch[7] != '/':  # 参考节点
                slack_bus = i
            elif branch[6] != '/':  # PV节点
                pv_bus.append(i)
            else:  # PQ节点
                pq_bus.append(i)
        no2no = {}
        reno2no = {}
        search_list = pq_bus + pv_bus + [slack_bus]
        value = 0
        for i in search_list:
            no2no[i] = value
            reno2no[value] = i
            input_power_data[i-1][1] = value
            value += 1
        input_power_data.sort(key=lambda x: (x[1]))
        for branch in input_admittance_data:
            branch[1] = no2no[branch[1]] + 1
            branch[3] = no2no[branch[3]] + 1
        return [no2no, len(pq_bus), reno2no]

    # 将输入文件列表转换为节点导纳矩阵/快速解耦潮流算法的雅可比矩阵
    def get_admittance_matrix(self):
        import numpy as np
        # 初始化节点导纳矩阵，即建立零矩阵
        data_array = np.zeros((self.shape, self.shape), dtype=complex)
        b_array = np.zeros((self.shape-1, self.shape-1), dtype=float)
        input_data = self.input_file_list_admittance  # 获取输入文件列表

        # 建立节点导纳矩阵
        for branch in input_data:
            i = branch[1] - 1
            j = branch[3] - 1
            if i < self.shape-1 and j < self.shape-1:
                b_array[i][j] = 1.0 / branch[5]
                b_array[j][i] = 1.0 / branch[5]
                b_array[i][i] -= 1.0 / branch[5]
                b_array[j][j] -= 1.0 / branch[5]
            elif i == self.shape-1:
                b_array[j][j] -= 1.0 / branch[5]
            else:
                b_array[i][i] -= 1.0 / branch[5]
            if branch[3] == 0:  # 判断是否为接地节点
                data_array[i][i] += 1.0/complex(branch[4], branch[5])
            elif branch[7] != '/':  # 判断是否为变压器节点
                data_array[i][i] += 1.0/complex(branch[4], branch[5])
                data_array[j][j] += 1.0/((branch[7]**2) * complex(branch[4], branch[5]))
                mutual_admittance = 1/(branch[7] * complex(branch[4], branch[5]))
                data_array[i][j] -= mutual_admittance
                data_array[j][i] -= mutual_admittance
            else:
                self_admittance = complex(0, branch[6]) + 1.0/complex(branch[4], branch[5])
                data_array[i][i] += self_admittance
                data_array[j][j] += self_admittance
                mutual_admittance = 1.0/complex(branch[4], branch[5])
                data_array[i][j] -= mutual_admittance
                data_array[j][i] -= mutual_admittance
        # 节点导纳矩阵建立完毕
        return {'data_array': data_array, 'b_array': b_array}

    # 读取输入功率的
    def get_power_data(self):
        input_power = self.input_file_list_power
        known_pq = [0.0] * (self.shape-1 + self.num_of_pq)
        voltage = [1.0] * self.shape
        angle = [0.0] * self.shape
        for branch in input_power:
            if branch[1] == (self.shape-1):
                angle[branch[1]] = branch[7]  # 读取参考角度
            if branch[1] < self.shape-1:
                known_pq[branch[1]] = branch[2] - branch[4]  # 读取有功功率
            if branch[1] < self.num_of_pq:
                known_pq[self.shape - 1 + branch[1]] = branch[3] - branch[5]  # 读取无功功率
            else:
                voltage[branch[1]] = branch[6]  # 读取电压
        return {
            'known_pq': known_pq,
            'voltage': voltage,
            'angle': angle,
        }


if __name__ == '__main__':
    data = Data(path_admittance='./data.txt', path_power='./data_power.txt')
    for i in range(9):
        for j in range(9):
            k = data.admittance_matrix[i][j]
            if k != 0:
                print('Y{}{}:{}+j{}'.format(i+1,j+1,round(k.real, 4), round(k.imag, 4)))
    print(data.pq_jacobian_matrix)
    print(data.pq_jacobian_matrix[:data.num_of_pq, :data.num_of_pq])
    """print(data.input_file_list_admittance)
    print(data.power_data['known_pq'])  
    print(data.power_data['voltage'])
    print(data.power_data['angle'])
    print(data.admittance_matrix)"""