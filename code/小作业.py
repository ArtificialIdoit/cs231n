class Matrix:
    def mul(x,y): #self?
        if(len(x[0]) != len(y)):
            print('ValueError')
        else:
            result = []
            for i in range(len(x)):
                temp = []
                for j in range(len(y[0])) :
                    temp2 = 0
                    for k in range(len(y)) :
                        temp2 += x[i][k]*y[k][j]
                    temp.append(temp2)
                result.append(temp)
            print('success!')
            return result
l = [[1,2,3],[4,5,6]]
m = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
print(Matrix.mul(l,m))