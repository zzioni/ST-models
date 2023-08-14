import pandas as pd
import math

def Lactin_Funtion(RHO, TM, DT, T):
    result = math.exp(RHO*T) - math.exp((RHO*TM) - ((TM-T) / DT))
    return result


def Weibull_Funtion(a, b, px):
    result = 1 - math.exp(-((px/a)**b))
    return result


def sigmoid_tem(MIN, MAX, L, TH):
    y_list = []
    for x in range(1, 25):
        y = ((math.sin((L*x) + TH))*(MAX - MIN) + (MAX + MIN))/2
        y_list.append(y)
    return y_list


def LntoLnn(data, RHO, TM, DT, a, b, source, target):

    min_tem_list = data['Minimum']
    max_tem_list = data['Maximum']

    physiological_age_list = []
    for min, max in zip(min_tem_list, max_tem_list):
        tem_list = sigmoid_tem(MIN=min, MAX=max, L=0.2618, TH=2.3562)
        pre_physiological_age_list = []

        for T in tem_list:
            result = Lactin_Funtion(RHO=RHO, TM=TM, DT=DT, T=T)
            pre_physiological_age_list.append(result)
        physiological_age_list.append((sum(pre_physiological_age_list))/24)
        print(physiological_age_list)


    data['Physiological age'] = physiological_age_list

    cohort_true = data[source][data[source] != 0]

    cohort_true_index = cohort_true.index.to_list()
    cohort_true_value = cohort_true.to_list()

    for index, value in zip(cohort_true_index, cohort_true_value):
        data['Physiological age cul'] = data['Physiological age'][index:]
        data['Physiological age cul'] = data['Physiological age cul'].fillna(0)
        data['Physiological age cul'] = data['Physiological age cul'].expanding().sum()

        target_list = []
        pre_rate = 0
        for i, px in enumerate(data['Physiological age cul'].to_list()):
            rate = Weibull_Funtion(a=a, b=b, px=px)
            target_list.append((rate - pre_rate) * value)
            pre_rate = rate
        data[target] = data[target] + target_list

    data.drop(['Physiological age', 'Physiological age cul'], axis=1, inplace=True)
    new_data = data.copy()

    return new_data


if __name__ == '__main__':

    #read_excel_file
    file_dir = 'Tem_data.xlsx'
    data = pd.read_excel(file_dir)

    #making_column
    data[['L2', 'L3', 'L4', 'L5', 'Pupae', 'Adult']] = 0

    #L1toL2
    dataL1toL2 = LntoLnn(data=data, RHO=0.15979, TM=38.27030, DT=6.24120, a=0.9797, b=6.202, source='Cohort', target='L2')

    #L2toL3
    dataL2toL3 = LntoLnn(data=dataL1toL2, RHO=0.139947, TM=42.6512, DT=7.113, a=0.8811, b=3.5949, source='L2', target='L3')

    #L3toL4
    dataL3toL4 = LntoLnn(data=dataL2toL3, RHO=0.182115, TM=37.6041, DT=5.4812, a=0.9177, b=3.1233, source='L3', target='L4')

    #L4toL5
    dataL4toL5 = LntoLnn(data=dataL3toL4, RHO=0.164026, TM=38.3008, DT=6.0811, a=0.982, b=4.0727, source='L4', target='L5')

    #L5toPupae
    dataL5toPupae = LntoLnn(data=dataL4toL5, RHO=0.167294, TM=38.8958, DT=5.9728, a=1.0002, b=4.7824, source='L5', target='Pupae')

    #PupaetoAdult
    dataPupaetoAdult = LntoLnn(data=dataL5toPupae, RHO=0.15468, TM=41.2146, DT=6.4608, a=0.9924, b=7.8764, source='Pupae', target='Adult')

    #result_excel
    dataPupaetoAdult.to_excel('result_2.xlsx', index=False)