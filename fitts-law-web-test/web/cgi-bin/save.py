#!/usr/bin/env python

import cgi
# import cgitb
# cgitb.enable()  # for troubleshooting

from datetime import datetime, timedelta

try:
    print "Content-Type: text/plain\n\n"
    # get data
    data = cgi.FieldStorage()

    # list of keys
    labels = data.keys()

    userID = str(data.getvalue('userID'))
    filename = str(data.getvalue('filename'))
    summaryFilename = '/home/pgrice/fitts-law-data/' + filename[:len(filename)-4] + '-' + userID + '-summary.csv'
    dataFilename = '/home/pgrice/fitts-law-data/' + filename[:len(filename)-4] + '-' + userID + '-data.csv'

    # make summary
    summaryList = ['a', 'b', 'throughput', 'dwellTime', 'startTime', 'endTime']
    s1 = ''
    s2 = ''
    for element in summaryList:
        s1 += element + ', '
        s2 += data.getvalue(element) + ', '

    s1 += 'total duration'
    s1 += '\n'

    # calculate total time duration
    startTime = str(data.getvalue('startTime'))
    endTime = str(data.getvalue('endTime'))
    startTime = startTime[0:24]
    endTime = endTime[0:24]

    time1 = datetime.strptime(startTime, '%a %b %d %Y %X')
    time2 = datetime.strptime(endTime, '%a %b %d %Y %X')

    delta = time2 - time1
    seconds = timedelta.total_seconds(delta)

    s2 += str(seconds)

    # write to file and close
    with open(summaryFilename, 'w') as f:
        f.write(s1)
        f.write(s2)
    f.close()

    dataList = ['round', 'target', 'startX', 'startY', 'endX', 'endY', 'goalX', 'goalY', 'startTime', 'endTime', 'duration', 'Diameter', 'setWidth', 'width', 'calculated distance']

    # get round number and target number
    rounds = 0
    n = 0
    for key in labels:
        if 'dataSets' in key:
            key = key.replace('[', ']')
            key = key.split(']')
            for part in key:
                if part == '':
                    key.remove('')
            currentRound = int(key[1])
            currentTarget = int(key[2])
            if currentRound > rounds:
                rounds = currentRound
            if currentTarget > n:
                n = currentTarget

    rounds += 1
    n += 1

    # intialize nested list
    fields = len(dataList)
    total = [[[None]*fields for i in range(0, n)] for i in range(0, rounds)]

    # get index values
    diameterInd = dataList.index('Diameter')
    actualWidthInd = dataList.index('setWidth')
    startXInd = dataList.index('startX')
    startYInd = dataList.index('startY')
    endXInd = dataList.index('endX')
    endYInd = dataList.index('endY')
    goalXInd = dataList.index('goalX')
    goalYInd = dataList.index('goalY')
    distanceInd = dataList.index('calculated distance')

    # add everything except the calculated distance into the list
    for label in labels:
        tf1 = 'dataSets' in label
        tf2 = 'setParameters' in label

        if tf1:
            value = data.getvalue(label)
            label = label.replace('[', ']')
            label = label.split(']')
            for part in label:
                if part == '':
                    label.remove('')
            roundCount = int(label[1])
            targetCount = int(label[2])

            if 'XY' in label[3]:
                x = value[0]
                y = value[1]
                if 'start' in label[3]:
                    total[roundCount][targetCount][startXInd] = str(x)
                    total[roundCount][targetCount][startYInd] = str(y)
                if 'end' in label[3]:
                    total[roundCount][targetCount][endXInd] = str(x)
                    total[roundCount][targetCount][endYInd] = str(y)
                if 'goal' in label[3]:
                    total[roundCount][targetCount][goalXInd] = str(x)
                    total[roundCount][targetCount][goalYInd] = str(y)
            else:
                fieldCount = dataList.index(label[3])
                total[roundCount][targetCount][fieldCount] = str(value)
                total[roundCount][targetCount][0] = str(roundCount+1)
                total[roundCount][targetCount][1] = str(targetCount+1)

        if tf2:
            value = data.getvalue(label)
            d = value[0]
            w = value[1]
            label = label.replace('[', ']')
            label = label.split(']')
            for part in label:
                if part == '':
                    label.remove('')
            roundCount = int(label[1])
            for i in range(0, n):
                total[abs(roundCount-rounds+1)][i][diameterInd] = str(d)
                total[abs(roundCount-rounds+1)][i][actualWidthInd] = str(w)

    # make first line of data file
    toText = ', '.join(dataList) + '\n'

    # flatten nested list
    newTotal = []
    for i in range(0, rounds):
        for j in range(0, n):
            sublist = total[i][j]
            newTotal.append(sublist)

    # calculate distance and write in flattened list
    for i in range(0, rounds*n):
        if i == 0:
            newTotal[i][distanceInd] = '0'
        else:
            x1 = float(newTotal[i][goalXInd])
            y1 = float(newTotal[i][goalYInd])
            x2 = float(newTotal[i-1][goalXInd])
            y2 = float(newTotal[i-1][goalYInd])
            d = ((x2-x1)**2 + (y2-y1)**2)**0.5
            newTotal[i][distanceInd] = str(d)

    # make list into string
    for x in range(0, rounds*n):
        l = newTotal[x]
        line = ", ".join(l)
        line += '\n'
        toText += line

    # write data to file
    with open(dataFilename, 'w') as f2:
        f2.write(toText)
    f2.close()

    # success message
    msg = "<head>"
    msg += "<style>"
    msg += "button.startButton {"
    msg += "display: block;"
    msg += "position: relative;"
    msg += "left: 38%;"
    msg += "width: 24%;"
    msg += "height: 8%;"
    msg += "bottom: 0%;"
    msg += "}"
    msg += "</style>"
    msg += "</head>"
    msg += "<div style='background-color:green'>"
    msg += "<h2>Thank you! Your data has been successfully submitted.</h2>"
    msg += "<h2>If you would like to take the test again, please click the button below.</h2>"
    msg += "</br>"
    msg += "<button class='startButton' onclick='history.go(0)'>Start over</button>"
    msg += "</div>"
    print msg

except Exception as e:
        # error message
        msg = "<head>"
        msg += "<style>"
        msg += "button.startButton {"
        msg += "display: block;"
        msg += "position: relative;"
        msg += "left: 38%;"
        msg += "width: 24%;"
        msg += "height: 8%;"
        msg += "bottom: 1%;"
        msg += "}"
        msg += "</style>"
        msg += "</head>"
        msg += "<div style='background-color:yellow'>"
        msg += "<h1>An error occured while sending data. Please click the button below to take the test again.</h1>"
        msg += "<h2>If error persists, please contact Phillip Grice at phillip.grice@gatech.edu</h2>"
        msg += "<button class='startButton' onclick='history.go(0)'>Start over</button>"
        msg += "</div>"
        print msg
