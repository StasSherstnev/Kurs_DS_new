def x(t): 
    return 5*t + 17
def y(t): 
    return 3*t*t + 2
t = 0  #int(input())

print(f'x= {x(t)}, y= {y(t)}')

##   Еще решение, которе работает в validator
def x(t): 
    x = 5*t + 17
    return x
def y(t): 
    y = 3*t*t + 2
    return y
t = 0  #int(input())
temp = 'x = {}, y = {}'
print(temp.format(x, y))