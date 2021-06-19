def main(actual, predicted):

    cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  # 3x3 confusion matrix
    
    a = open(actual).readlines()
    p = open(predicted).readlines()

    for i in range(len(a)):
        act = int(a[i].rstrip())
        pred = int(p[i].rstrip())
        cm[pred][act] += 1

    # print(cm[0])
    # print(cm[1])
    # print(cm[2])

    prec = [0, 0, 0]
    rec = [0, 0, 0]

    for i in range(3):
        prec[i] = cm[i][i] / (cm[i][0] + cm[i][1] + cm[i][2])
        rec[i] = cm[i][i] / (cm[0][i] + cm[1][i] + cm[2][i])

    # print(prec)
    # print(rec)

    f1_list = [0, 0, 0]

    for i in range(3):
        f1_list[i] = 2 * (prec[i] * rec[i]) / (prec[i] + rec[i])

    # print(f1_list)
    
    f1 = sum(f1_list) / 3

    # print(f1)

    accuracy = (cm[1][1] + cm[2][2] + cm[0][0]) / len(a)

    print('accuracy = ', accuracy)
    print('f1 score = ', f1)


if __name__ == '__main__':
    actual = 'reference/random_order_data/testing2_label.txt'
    predicted = 'result2'
    main(actual, predicted)

