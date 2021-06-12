from tools import *

result, times, source = decoupling_method(brach_path='./data_network.txt', power_path='./data_power.txt')

short_circuit_bus = 4  # 短路点编号
sc_voltage,  sc_current, info = accurate_short_circuit_param(result=result, source=source,
                                                             short_circuit_no=short_circuit_bus)
rough_sc_voltage, rough_info = rough_short_circuit_param(source=source, short_circuit_no=short_circuit_bus)

write2txt(result, source, info, rough_info, sc_voltage,
          sc_current, rough_sc_voltage, short_circuit_no=short_circuit_bus)  # 将结果写入到txt文件中