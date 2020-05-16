from Methodology import COST_FUNCTION, LETAL_COEFF, MIN_DTP_AMOUNT, THRESHHOLD
from Data_preparation import get_data


if __name__ == '__main__':
    data = get_data(columns=['infoDtp'])

    stats = []
    for i in range(data.shape[0]):
        results = [[K_UCH['S_T'] for K_UCH in ts['ts_uch']] for ts in data['infoDtp'][i]['ts_info']]
        for r in results:
            for r_s in r:
                if r_s not in stats:
                    stats.append(r_s)


    for s in stats:
        print(s)

