Y = []
filepath = 'newdatay.csv'
with open(filepath) as fp:
   line = fp.readline()
   cnt = 1
   while line:
       print("Line {}: {}".format(cnt, line.strip()))
       Y.append(float(line.strip()))
       line = fp.readline()
       cnt += 1
print (Y)
