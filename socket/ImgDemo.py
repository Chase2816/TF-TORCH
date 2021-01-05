# import numpy as np
# from scipy import stats
# import matplotlib.pyplot as plt
# import matplotlib.font_manager
# from pyod.models.abod import ABOD
# from pyod.models.knn import KNN
# from pyod.utils.data import generate_data,get_outliers_inliers
#
#
# X_train,Y_train = generate_data(n_train=200,train_only=True,n_features=2)
# outlier_fraction = 0.1
# x_outliers,x_inliers = get_outliers_inliers(X_train,Y_train)
# n_inliers = len(x_inliers)
# n_outliers = len(x_outliers)
#
# F1 = X_train[:,[0]].reshape(-1,1)
# F2 = X_train[:,[1]].reshape(-1,1)
#
# xx,yy = np.meshgrid(np.linspace(-10,10,200),np.linspace(-10,10,200))
#
# plt.scatter(F1,F2)
# plt.xlabel('F1')
# plt.ylabel('F2')
#
# classifiers = {
# 'Angle-based Outlier Detector (ABOD)' : ABOD(contamination=outlier_fraction),
# 'K Nearest Neighbors (KNN)' : KNN(contamination=outlier_fraction)
# }
#
# plt.figure(figsize=(10,10))
# for i,(clf_name,clf) in enumerate(classifiers.items()):
#     clf.fit(X_train)
#     scores_pred = clf.decision_function(X_train)*-1
#     y_pred = clf.predict(X_train)
#     n_errors = (y_pred != Y_train).sum()
#     print("No of Erroes:",clf_name,n_errors)
#     threshold = stats.scoreatpercentile(scores_pred,100*outlier_fraction)
#
#     Z = clf.decision_function(np.c_[xx.ravel(),yy.ravel()])*-1
#     Z = Z.reshape(xx.shape)
#     subplot = plt.subplot(1,2,i+1)
#     subplot.contourf(xx,yy,Z,levels=np.linspace(Z.min(),10),cmap=plt.cm.Blues_r)
#     a = subplot.contour(xx,yy,Z,levels=[threshold],linewidths=2,colors='red')
#     subplot.contourf(xx,yy,Z,levels=[threshold,Z.max()],colors='orange')
#     b = subplot.scatter(X_train[:-n_outliers,0],X_train[:-n_outliers,1],c='white',s=20,edgecolor='k')
#     c = subplot.scatter(X_train[-n_outliers:,0],X_train[-n_outliers:,1],c='black',s=20,edgecolor='k')
#     subplot.axis('tight')
#     subplot.legend([a.collections[0],b,c],['learned decision function', 'true inliers', 'true outliers'],prop=matplotlib.font_manager.FontProperties(size=10),loc='lower right')
#     subplot.set_title(clf_name)
#     subplot.set_xlim((-10,10))
#     subplot.set_ylim((-10,10))
#     plt.show()

import socketserver
#
class myTCPhandler(socketserver.BaseRequestHandler):
    def handle(self):
        while True:
            self.data = self.request.recv(1024).decode('UTF-8', 'ignore').strip()
            if not self.data : break
            print(self.data)
            self.feedback_data =("回复\""+self.data+"\":\n\t你好，我是Server端").encode("utf8")
            print("发送成功")
            self.request.sendall(self.feedback_data)

host = '127.0.0.1'
port = 9007
server = socketserver.ThreadingTCPServer((host,port),myTCPhandler)
server.serve_forever()

from socket import *
import numpy as np
import cv2
import base64

def main():
    HOST = '127.0.0.1'
    PORT = 9999
    BUFSIZ = 1024*20
    ADDR = (HOST, PORT)
    tcpSerSock = socket(AF_INET, SOCK_STREAM)
    tcpSerSock.bind(ADDR)
    tcpSerSock.listen(5)
    while True:
        rec_d = bytes([])
        print('waiting for connection...')
        tcpCliSock, addr = tcpSerSock.accept()
        print('...connected from:', addr)
        while True:
            data = tcpCliSock.recv(BUFSIZ)
            if not data or len(data) == 0:
                break
            else:
                rec_d = rec_d + data
        rec_d = base64.b64decode(rec_d)
        np_arr = np.fromstring(rec_d, np.uint8)
        image = cv2.imdecode(np_arr, 1)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        tcpCliSock.send("0001".encode())
        # tcpCliSock.send("返回值")
        tcpCliSock.close()
    tcpSerSock.close()

if __name__ == "__main__":
    main()
