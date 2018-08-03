# Name 		: Kuldeep Singh Bhandari
# Roll No.  : 111601009


f = open("test.txt", "r")
begin = False
c = 0
while(True) :
	x = f.read(1)
	if(x == '') :
		break
	if(not begin and x != '\n' and x != ' ') :
		begin = True
	if(begin and (x == ' ' or x == '\n')) :	
		c += 1
		begin = False

print((c+1) if begin else c)