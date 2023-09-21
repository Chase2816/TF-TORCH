def drawBox(frame, classId, conf, left, top, right, bottom, classMask):
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                 (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    print(classMask.shape)
    classMask = cv.resize(classMask, (right - left + 1, bottom - top + 1))
    print('============4============')
    print(classMask.shape)
    print(classMask)
    mask = (classMask > maskThreshold)
    print('============5============')
    print(mask)

    roi = frame[top:bottom + 1, left:right + 1][mask]
    # print(roi)
    # color = colors[classId%len(colors)]
    colorIndex = random.randint(0, len(colors) - 1)
    color = colors[colorIndex]

    frame[top:bottom + 1, left:right + 1][mask] = ([0.3 * color[0], 0.3 * color[1], 0.3 * color[2]] + 0.7 * roi).astype(
        np.uint8)

    mask = mask.astype(np.uint8)
    print('[[[[[[[[[[[[[[[[[[[[[[[[')
    print(mask)
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print("========6==============")
    print(contours)
    print(np.array(contours).shape)
    print(hierarchy)
    exit()
    cv.drawContours(frame[top:bottom + 1, left:right + 1], contours, -1, color, 3, cv.LINE_8, hierarchy, 100)
